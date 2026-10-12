"""LangGraph state shared across the 7 co-scientist agents.

This state is intentionally compact (Fetch-Once / Process-Locally pattern):
each agent adds only what the next agent needs, no prose duplication.
"""
from __future__ import annotations
from typing import List, Optional, TypedDict

from app.utils.schemas import (
    DocumentChunk,
    Hypothesis,
    ReflectionReport,
    RankedHypothesis,
    EvolutionResult,
    MetaReview,
)


class CoScientistState(TypedDict, total=False):
    # input
    goal: str
    language: str                      # 'en' | 'vi'
    retrieved_chunks: List[DocumentChunk]   # set once, reused (fetch-once)
    context_text: str                  # pre-compressed context string

    # generation
    hypotheses: List[Hypothesis]

    # reflection
    reflections: List[ReflectionReport]

    # ranking
    ranked: List[RankedHypothesis]
    round: int

    # proximity
    clusters: List[List[str]]          # list of hyp_id clusters

    # evolution
    evolutions: List[EvolutionResult]

    # meta-review (final)
    meta: Optional[MetaReview]

    # meta
    notes: str
