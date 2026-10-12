"""Supervisor Agent — parses goal, builds research config, allocates resources.

Pattern: One-time routing step (Aragog-style) + Schema-contracted prompt.
"""
from __future__ import annotations
from pydantic import BaseModel, Field

from app.llm.provider import get_fast_llm
from app.llm.structured import run_with_schema
from app.agents.state import CoScientistState
from app.utils.logger import get_logger, log_event

logger = get_logger(__name__)


class ResearchConfig(BaseModel):
    language: str = "en"
    num_hypotheses: int = Field(default=4, ge=2, le=8)
    max_rounds: int = Field(default=2, ge=1, le=5)
    focus: str = "general"
    sub_questions: list[str] = Field(default_factory=list)


SUPERVISOR_SYS = (
    "You are the Supervisor Agent of a multi-agent co-scientist. "
    "Given a user goal, decide research configuration: language, "
    "number of hypotheses to generate, max co-scientist rounds, and a 1-line focus. "
    "Output JSON only."
)


def supervisor_node(state: CoScientistState) -> CoScientistState:
    with log_event(logger, "Supervisor", {"goal": state.get("goal", "")[:80]}):
        llm = get_fast_llm()
        prompt = (
            f"User goal: {state['goal']}\n"
            f"Detected document language hint: {state.get('language', 'auto')}\n"
            "Produce a ResearchConfig."
        )
        # raw structured call (no need for full schema hint here)
        cfg: ResearchConfig = run_with_schema(
            llm, ResearchConfig, prompt, agent_name="supervisor", tier="fast"
        )
    return {
        "language": cfg.language,
        "round": 0,
        "notes": f"config: focus={cfg.focus}, max_rounds={cfg.max_rounds}, n={cfg.num_hypotheses}",
        "_cfg": cfg.model_dump(),  # internal-only
    }


def route_after_supervisor(state: CoScientistState) -> str:
    # Always go to Generation first.
    return "generation"
