"""Structured-output helper enforcing Pydantic schema (Schema-Contracted Prompts)."""
from __future__ import annotations
import json
import time
from typing import Any, Type, TypeVar
from pydantic import BaseModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from app.utils.logger import get_logger, METRICS

logger = get_logger(__name__)
T = TypeVar("T", bound=BaseModel)


SYSTEM_INSTRUCTIONS = (
    "You are a precise scientific assistant. "
    "Respond ONLY with valid JSON that matches the requested schema. "
    "Do not include commentary, markdown fences, or extra text."
)


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _coerce_to_text(content: Any) -> str:
    """Gemini 2.5+ may return content as list[dict] (multi-part). Flatten to str."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                # multimodal: {'type': 'text', 'text': '...'}
                t = p.get("text")
                if t:
                    parts.append(t)
        return "\n".join(parts)
    return str(content)


def _strip_code_fence(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.startswith("json"):
            s = s[4:]
        return s.strip()
    return s


def run_with_schema(
    llm: BaseChatModel,
    schema: Type[T],
    user_prompt: str,
    *,
    agent_name: str = "unknown",
    tier: str = "fast",
) -> T:
    """Invoke the LLM and parse the result into `schema`.

    Schema-contracted prompt: we append the schema name + JSON shape so the
    model is forced to return a single JSON object.
    """
    schema_hint = (
        f"\n\nReturn JSON matching this schema:\n"
        f"schema_name: {schema.__name__}\n"
        f"fields: {list(schema.model_fields.keys())}\n"
        f"example: {schema.model_json_schema()}"
    )
    full_prompt = user_prompt + schema_hint

    messages = [
        SystemMessage(content=SYSTEM_INSTRUCTIONS),
        HumanMessage(content=full_prompt),
    ]

    t0 = time.perf_counter()
    raw_obj = llm.invoke(messages).content
    raw = _coerce_to_text(raw_obj)
    elapsed = (time.perf_counter() - t0) * 1000

    cleaned = _strip_code_fence(raw)

    try:
        obj = schema.model_validate_json(cleaned)
    except Exception as e:
        logger.warning(f"[{agent_name}] JSON parse failed, retrying once. err={e}")
        fix_msg = _coerce_to_text(
            llm.invoke(
                [
                    SystemMessage(content="Fix the following so it is valid JSON. Output only JSON."),
                    HumanMessage(content=cleaned),
                ]
            ).content
        )
        obj = schema.model_validate_json(_strip_code_fence(fix_msg))

    METRICS.record(
        agent=agent_name,
        elapsed_ms=elapsed,
        tokens_in=_approx_tokens(full_prompt),
        tokens_out=_approx_tokens(raw),
    )
    logger.info(
        f"[{agent_name}|{tier}] {elapsed:.0f}ms "
        f"tokens≈in{_approx_tokens(full_prompt)}/out{_approx_tokens(raw)}"
    )
    return obj

