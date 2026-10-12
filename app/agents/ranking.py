"""Ranking Agent — Elo tournament between hypotheses, with simulated debate.

Pattern: Schema-contracted (one batched ranking call) — no multi-turn debate
in the demo, to control cost and latency.
"""
from __future__ import annotations
from typing import List
from pydantic import BaseModel, Field

from app.llm.provider import get_smart_llm
from app.llm.structured import run_with_schema
from app.agents.state import CoScientistState
from app.utils.schemas import Hypothesis, ReflectionReport, RankedHypothesis
from app.utils.logger import get_logger, log_event

logger = get_logger(__name__)


class RankingDecision(BaseModel):
    items: List[RankedHypothesis] = Field(default_factory=list)
    rationale: str = ""


RANK_SYS = (
    "You are the Ranking Agent. You run a simulated scientific debate among "
    "hypotheses and produce an Elo-style ranked list. Score criteria: "
    "feasibility, novelty, rigor, safety. Output JSON only."
)


def _elo_pair(a: RankedHypothesis, b: RankedHypothesis, score_a: float) -> tuple[float, float]:
    """Standard Elo update (K=24)."""
    K = 24
    ea = 1 / (1 + 10 ** ((b.elo - a.elo) / 400))
    eb = 1 - ea
    a.elo += K * (score_a - ea)
    b.elo += K * ((1 - score_a) - eb)
    return a.elo, b.elo


def ranking_node(state: CoScientistState) -> CoScientistState:
    with log_event(logger, "Ranking", {"n": len(state.get("hypotheses", []))}):
        hyps: List[Hypothesis] = state.get("hypotheses", [])
        refs: List[ReflectionReport] = state.get("reflections", [])

        # initialise Elo from reflection scores
        ref_map = {r.hyp_id: r for r in refs}
        ranked: List[RankedHypothesis] = []
        for h in hyps:
            r = ref_map.get(h.hyp_id)
            base = 1200 + 200 * ((r.novelty + r.correctness + r.feasibility) / 3 if r else 0)
            ranked.append(RankedHypothesis(hyp_id=h.hyp_id, title=h.title, elo=base))

        # one batched LLM decision to "debate" and order
        llm = get_smart_llm()
        prompt = (
            "Re-rank these hypotheses after a simulated debate. "
            "Use feasibility, novelty, rigor, safety. Return the list in best-first order.\n\n"
            + "\n".join(
                f"id={r.hyp_id}\ntitle={r.title}\nelo={r.elo:.0f}\n---"
                for r in ranked
            )
        )
        decision: RankingDecision = run_with_schema(
            llm, RankingDecision, prompt, agent_name="ranking", tier="smart"
        )
        # merge Elo from LLM ordering
        if decision.items:
            ordered_titles = [it.hyp_id for it in decision.items]
            for pos, hid in enumerate(ordered_titles):
                for r in ranked:
                    if r.hyp_id == hid:
                        r.elo += 100 * (len(ranked) - pos)
                        r.rationale = decision.rationale[:200]
                        break
        ranked.sort(key=lambda x: -x.elo)
    return {"ranked": ranked, "round": state.get("round", 0) + 1}
