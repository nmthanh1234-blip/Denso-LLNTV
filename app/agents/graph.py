"""LangGraph definition: Co-Scientist cycle.

Flow:
  Supervisor -> Generation -> Reflection -> Ranking -> Proximity
                                                              |
                                                              v
                                                         (loop? -- Evolution)
                                                              |
                                                              v
                                                       Meta-Review -> END
"""
from __future__ import annotations
from langgraph.graph import StateGraph, END

from app.agents.state import CoScientistState
from app.agents.supervisor import supervisor_node
from app.agents.generation import generation_node
from app.agents.reflection import reflection_node
from app.agents.ranking import ranking_node
from app.agents.proximity import proximity_node
from app.agents.evolution import evolution_node
from app.agents.meta_review import meta_review_node
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _should_evolve(state: CoScientistState) -> str:
    """Conditional edge: if more rounds remain, evolve; else finalise."""
    r = state.get("round", 0)
    max_r = state.get("_cfg", {}).get("max_rounds", 2)
    return "evolution" if r < max_r else "meta_review"


def build_graph():
    g = StateGraph(CoScientistState)

    g.add_node("supervisor", supervisor_node)
    g.add_node("generation", generation_node)
    g.add_node("reflection", reflection_node)
    g.add_node("ranking", ranking_node)
    g.add_node("proximity", proximity_node)
    g.add_node("evolution", evolution_node)
    g.add_node("meta_review", meta_review_node)

    g.set_entry_point("supervisor")
    g.add_edge("supervisor", "generation")
    g.add_edge("generation", "reflection")
    g.add_edge("reflection", "ranking")
    g.add_edge("ranking", "proximity")
    g.add_conditional_edges(
        "proximity",
        _should_evolve,
        {"evolution": "evolution", "meta_review": "meta_review"},
    )
    g.add_edge("evolution", "reflection")  # loop
    g.add_edge("meta_review", END)

    return g.compile()


# pre-built graph
GRAPH = build_graph()


def run_scientist(
    goal: str,
    context_text: str,
    *,
    language: str = "en",
    retrieved_chunks=None,
    config: dict | None = None,
) -> dict:
    """Convenience runner used by Streamlit / CLI."""
    initial: CoScientistState = {
        "goal": goal,
        "language": language,
        "context_text": context_text,
        "retrieved_chunks": retrieved_chunks or [],
    }
    if config:
        initial["_cfg"] = config
    final = GRAPH.invoke(initial)
    return final
