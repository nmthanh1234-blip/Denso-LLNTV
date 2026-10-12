"""Evolution Agent — produce the next generation of hypotheses.

Operations: combination / grounding / simplification / out_of_box.
"""
from __future__ import annotations
from typing import List
from pydantic import BaseModel, Field

from app.llm.provider import get_smart_llm
from app.llm.structured import run_with_schema
from app.agents.state import CoScientistState
from app.utils.schemas import Hypothesis, EvolutionResult
from app.utils.logger import get_logger, log_event

logger = get_logger(__name__)


class EvolutionBatch(BaseModel):
    items: List[EvolutionResult] = Field(default_factory=list)


EVOL_SYS = (
    "You are the Evolution Agent. Take the top-ranked hypotheses and produce "
    "the next generation using ONE of: combination, grounding, simplification, "
    "out_of_box. Each new hypothesis must cite chunk ids and be self-contained. "
    "Output JSON only."
)


def evolution_node(state: CoScientistState) -> CoScientistState:
    with log_event(logger, "Evolution", {"round": state.get("round", 0)}):
        ranked = state.get("ranked", [])
        if state.get("round", 0) >= state.get("_cfg", {}).get("max_rounds", 2):
            return {}  # stop evolving
        top = ranked[:2]
        if not top:
            return {}

        # lookup full hypotheses
        hyp_map = {h.hyp_id: h for h in state.get("hypotheses", [])}
        parent_text = "\n\n".join(
            f"PARENT {h.hyp_id} ({r.title}):\n{hyp_map.get(h.hyp_id, Hypothesis(hyp_id=h.hyp_id, title=r.title, body='')).body}"
            for r in top for h in [r]
            if h.hyp_id in hyp_map
        )

        llm = get_smart_llm()
        prompt = (
            f"Goal: {state['goal']}\n"
            f"Round: {state.get('round', 0)}\n"
            f"Context: {state.get('context_text', '')[:2000]}\n\n"
            f"Top-ranked parents:\n{parent_text}\n\n"
            "Produce 2 next-generation hypotheses."
        )
        batch: EvolutionBatch = run_with_schema(
            llm, EvolutionBatch, prompt, agent_name="evolution", tier="smart"
        )
        # ensure unique ids
        for ev in batch.items:
            ev.new_hypothesis.hyp_id = ev.new_hypothesis.hyp_id or str(uuid.uuid4())[:8]
        # append to hypotheses
        new_hyps = [ev.new_hypothesis for ev in batch.items]
        return {
            "evolutions": batch.items,
            "hypotheses": state.get("hypotheses", []) + new_hyps,
        }
