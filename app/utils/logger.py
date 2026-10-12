"""Lightweight logger + token/latency accounting helpers."""
from __future__ import annotations
import logging
import sys
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

# Force UTF-8 stdout on Windows (cp1252 cannot print Vietnamese)
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(name)-25s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


@contextmanager
def log_event(
    logger: logging.Logger,
    event: str,
    extra: Optional[Dict[str, Any]] = None,
):
    """Context manager: logs start/end of an event with elapsed ms."""
    start = time.perf_counter()
    meta = {"event": event, **(extra or {})}
    logger.info(f"[START] {event} | {meta}")
    payload: Dict[str, Any] = {"meta": meta}
    try:
        yield payload
    except Exception as exc:
        elapsed = (time.perf_counter() - start) * 1000
        logger.exception(f"[FAIL ] {event} after {elapsed:.1f}ms | err={exc}")
        raise
    else:
        elapsed = (time.perf_counter() - start) * 1000
        payload["elapsed_ms"] = elapsed
        logger.info(f"[END  ] {event} in {elapsed:.1f}ms")


# --- Simple in-memory token/latency registry (used by Streamlit dashboard) ---
class Metrics:
    """Process-local metrics aggregator."""

    def __init__(self) -> None:
        self.events: list[dict] = []
        self.tokens_in = 0
        self.tokens_out = 0

    def record(self, agent: str, elapsed_ms: float, tokens_in: int = 0, tokens_out: int = 0) -> None:
        self.events.append(
            {"agent": agent, "elapsed_ms": elapsed_ms, "tokens_in": tokens_in, "tokens_out": tokens_out}
        )
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out

    def summary(self) -> dict:
        if not self.events:
            return {"total_calls": 0, "total_ms": 0.0, "total_tokens": 0}
        return {
            "total_calls": len(self.events),
            "total_ms": round(sum(e["elapsed_ms"] for e in self.events), 1),
            "total_tokens": self.tokens_in + self.tokens_out,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
        }


METRICS = Metrics()
