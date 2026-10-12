"""Reflection Agent — 4 sub-reviews (Initial/Full/Deep/Simulation) run in PARALLEL.

This implements the Asynchronous / Fan-out / Fan-in pattern from DENSO.pdf
to minimise wall-clock latency.
"""
from __future__ import annotations
import asyncio
from typing import List
from pydantic import BaseModel, Field

from app.llm.provider import get_smart_llm, get_fast_llm
from app.llm.structured import run_with_schema
from app.agents.state import CoScientistState
from app.utils.schemas import Hypothesis, ReflectionReport
from app.utils.logger import get_logger, log_event

logger = get_logger(__name__)


# ----- Schema for each sub-review (4 specialised outputs) -----

class InitialReview(BaseModel):
    hyp_id: str
    safety_ok: bool = True
    obvious_flaw: bool = False
    notes: str = ""


class FullReview(BaseModel):
    hyp_id: str
    novelty: float = 0.0         # 0..1
    correctness: float = 0.0     # 0..1
    notes: str = ""


class DeepReview(BaseModel):
    hyp_id: str
    assumptions: list[str] = Field(default_factory=list)
    invalid_assumptions: list[str] = Field(default_factory=list)


class SimReview(BaseModel):
    hyp_id: str
    failure_modes: list[str] = Field(default_factory=list)
    feasibility: float = 0.0    # 0..1


# ----- 4 parallel sub-workers -----

async def _initial_review(h: Hypothesis, ctx: str) -> InitialReview:
    return await asyncio.to_thread(
        run_with_schema,
        get_fast_llm(),
        InitialReview,
        f"Hypothesis: {h.title}\n{h.body}\n\nContext: {ctx[:1500]}\nQuick safety + obvious-flaw check.",
        agent_name="reflection.initial",
    )


async def _full_review(h: Hypothesis, ctx: str) -> FullReview:
    return await asyncio.to_thread(
        run_with_schema,
        get_smart_llm(),
        FullReview,
        f"Hypothesis: {h.title}\n{h.body}\n\nContext: {ctx[:2500]}\n"
        "Rate novelty and correctness 0..1. Be strict.",
        agent_name="reflection.full",
    )


async def _deep_review(h: Hypothesis, ctx: str) -> DeepReview:
    return await asyncio.to_thread(
        run_with_schema,
        get_smart_llm(),
        DeepReview,
        f"Hypothesis: {h.title}\n{h.body}\n\nContext: {ctx[:2500]}\n"
        "List hidden assumptions, then flag which are invalid.",
        agent_name="reflection.deep",
    )


async def _sim_review(h: Hypothesis, ctx: str) -> SimReview:
    return await asyncio.to_thread(
        run_with_schema,
        get_smart_llm(),
        SimReview,
        f"Hypothesis: {h.title}\n{h.body}\n\nContext: {ctx[:2500]}\n"
        "Simulate execution: list failure modes and rate feasibility 0..1.",
        agent_name="reflection.sim",
    )


async def _review_one(h: Hypothesis, ctx: str) -> ReflectionReport:
    init, full, deep, sim = await asyncio.gather(
        _initial_review(h, ctx),
        _full_review(h, ctx),
        _deep_review(h, ctx),
        _sim_review(h, ctx),
    )
    return ReflectionReport(
        hyp_id=h.hyp_id,
        safety_ok=init.safety_ok and not init.obvious_flaw,
        novelty=full.novelty,
        correctness=full.correctness,
        feasibility=sim.feasibility,
        assumptions=deep.assumptions,
        failure_modes=sim.failure_modes,
        notes=(
            f"init={init.notes[:60]}; full={full.notes[:60]}; "
            f"invalid_assumps={len(deep.invalid_assumptions)}"
        ),
    )


async def _review_all(hypotheses: List[Hypothesis], ctx: str) -> List[ReflectionReport]:
    sem = asyncio.Semaphore(4)  # bound parallelism

    async def _bounded(h):
        async with sem:
            return await _review_one(h, ctx)

    return await asyncio.gather(*[_bounded(h) for h in hypotheses])


def reflection_node(state: CoScientistState) -> CoScientistState:
    """Reflection node — fans out 4 sub-reviews × N hypotheses, fans in."""
    with log_event(logger, "Reflection", {"n": len(state.get("hypotheses", []))}):
        reflections = asyncio.run(
            _review_all(state.get("hypotheses", []), state.get("context_text", ""))
        )
    return {"reflections": reflections}
