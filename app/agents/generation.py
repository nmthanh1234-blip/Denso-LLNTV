"""Generation Agent — produces N diverse hypotheses from retrieved context.

Uses a fast LLM (flash) — Generation is high-volume, low-stakes.
"""
from __future__ import annotations
import uuid
from typing import List
from pydantic import BaseModel, Field

from app.llm.provider import get_fast_llm
from app.llm.structured import run_with_schema
from app.agents.state import CoScientistState
from app.utils.schemas import Hypothesis
from app.utils.logger import get_logger, log_event

logger = get_logger(__name__)


class HypothesisBatch(BaseModel):
    items: List[Hypothesis] = Field(default_factory=list)


GEN_SYS = (
    "You are the Generation Agent of a co-scientist system. "
    "Given a research goal and document context, propose diverse, specific, "
    "testable hypotheses. Each hypothesis must cite at least one chunk id "
    "(e.g. 'src:file.pdf p3'). Output JSON only."
)


def generation_node(state: CoScientistState) -> CoScientistState:
    with log_event(logger, "Generation", {"n": state.get("_cfg", {}).get("num_hypotheses", 4)}):
        llm = get_fast_llm()
        n = state.get("_cfg", {}).get("num_hypotheses", 4)
        focus = state.get("_cfg", {}).get("focus", "general")
        prompt = (
            f"Goal: {state['goal']}\n"
            f"Focus: {focus}\n"
            f"Context (chunks, structured):\n{state.get('context_text', '')[:3500]}\n\n"
            f"Generate exactly {n} hypotheses. Make them diverse, testable, and well-cited."
        )
        # ensure unique hyp_ids
        batch: HypothesisBatch = run_with_schema(
            llm, HypothesisBatch, prompt, agent_name="generation", tier="fast"
        )
        for h in batch.items:
            h.hyp_id = h.hyp_id or str(uuid.uuid4())[:8]
    return {"hypotheses": batch.items}
