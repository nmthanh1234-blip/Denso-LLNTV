"""Meta-Review Agent — synthesises the final answer + Research Overview."""
from __future__ import annotations
from typing import List
from pydantic import BaseModel, Field

from app.llm.provider import get_smart_llm
from app.llm.structured import run_with_schema
from app.agents.state import CoScientistState
from app.utils.schemas import Hypothesis, MetaReview
from app.utils.logger import get_logger, log_event

logger = get_logger(__name__)


class MetaDecision(BaseModel):
    final_hyp_id: str
    report_markdown: str
    recurring_failure_modes: List[str] = Field(default_factory=list)


META_SYS = (
    "You are the Meta-Review Agent. You have observed every co-scientist round. "
    "Pick the final hypothesis and write a Research Overview (markdown) with: "
    "Background, Selected Hypothesis, Evidence, Limitations, Next Steps. "
    "Output JSON only."
)


def _best_hyp(state: CoScientistState) -> Hypothesis:
    ranked = state.get("ranked", [])
    hyp_map = {h.hyp_id: h for h in state.get("hypotheses", [])}
    if ranked:
        return hyp_map.get(ranked[0].hyp_id) or Hypothesis(
            hyp_id=ranked[0].hyp_id, title=ranked[0].title, body="(body missing)"
        )
    return state.get("hypotheses", [Hypothesis(hyp_id="?", title="?", body="?")])[0]


def meta_review_node(state: CoScientistState) -> CoScientistState:
    with log_event(logger, "MetaReview", {"hypotheses": len(state.get("hypotheses", []))}):
        # recurring failure modes
        all_failures: List[str] = []
        for r in state.get("reflections", []):
            all_failures.extend(r.failure_modes)
        # most common failure modes
        from collections import Counter
        common = [f for f, _ in Counter(all_failures).most_common(5)]

        ranked = state.get("ranked", [])
        top3 = ranked[:3]
        hyp_map = {h.hyp_id: h for h in state.get("hypotheses", [])}
        top3_text = "\n".join(
            f"id={r.hyp_id} elo={r.elo:.0f}\n"
            f"title={r.title}\nbody={hyp_map.get(r.hyp_id, Hypothesis(hyp_id=r.hyp_id,title=r.title,body='')).body[:600]}"
            for r in top3
        )
        llm = get_smart_llm()
        prompt = (
            f"Goal: {state['goal']}\n"
            f"Top candidates:\n{top3_text}\n\n"
            f"Recurring failure modes observed: {common}\n\n"
            "Pick the best one (final_hyp_id) and write the Research Overview."
        )
        decision: MetaDecision = run_with_schema(
            llm, MetaDecision, prompt, agent_name="meta_review", tier="smart"
        )
        final = hyp_map.get(decision.final_hyp_id) or _best_hyp(state)
        meta = MetaReview(
            final_hypothesis=final,
            report_markdown=decision.report_markdown,
            recurring_failure_modes=common,
        )
    return {"meta": meta}
