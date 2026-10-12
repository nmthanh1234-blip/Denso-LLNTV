"""LLM factory — model tiering (cheap flash for fan-out, smart pro for review).

Includes a runtime fallback: if the "smart" model hits quota / not-found /
any transient error, we transparently fall back to the "fast" model so
the pipeline never breaks (Token-Aware Fallback Chains pattern from DENSO.pdf).
"""
from __future__ import annotations
from functools import lru_cache
import warnings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.messages import BaseMessage

from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_FALLBACK_FLAG = "_co_scientist_fell_back"


class FallbackLLM(BaseChatModel):
    """Wraps smart LLM, falls back to fast LLM on transient errors.

    Implements the Token-Aware Fallback Chains idea: don't retry the same
    expensive model — switch to the cheaper one and keep going.
    """

    smart: BaseChatModel
    fast: BaseChatModel
    fell_back: bool = False

    @property
    def _llm_type(self) -> str:  # required by langchain
        return "google-genai-fallback"

    def _generate(self, messages, stop=None, **kwargs):
        try:
            return self.smart._generate(messages, stop=stop, **kwargs)
        except Exception as e:
            err = str(e).lower()
            transient = any(
                token in err
                for token in (
                    "429",
                    "resource_exhausted",
                    "quota",
                    "rate_limit",
                    "not_found",
                    "unavailable",
                    "deadline",
                    "timeout",
                )
            )
            if not transient:
                raise
            self.fell_back = True
            logger.warning(
                f"Smart LLM failed ({type(e).__name__}: {str(e)[:120]}) → falling back to FAST model"
            )
            return self.fast._generate(messages, stop=stop, **kwargs)

    async def _agenerate(self, messages, stop=None, **kwargs):
        try:
            return await self.smart._agenerate(messages, stop=stop, **kwargs)
        except Exception as e:
            err = str(e).lower()
            transient = any(
                token in err
                for token in (
                    "429",
                    "resource_exhausted",
                    "quota",
                    "rate_limit",
                    "not_found",
                    "unavailable",
                    "deadline",
                    "timeout",
                )
            )
            if not transient:
                raise
            self.fell_back = True
            logger.warning(
                f"Smart LLM failed (async) → falling back to FAST model"
            )
            return await self.fast._agenerate(messages, stop=stop, **kwargs)


@lru_cache(maxsize=1)
def _fast_llm_inner() -> BaseChatModel:
    logger.info(f"Init FAST LLM: {settings.gemini_model_fast}")
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model_fast,
        google_api_key=settings.google_api_key,
        temperature=0.4,
        max_output_tokens=2048,
    )


@lru_cache(maxsize=1)
def _smart_llm_inner() -> BaseChatModel:
    logger.info(f"Init SMART LLM: {settings.gemini_model_smart}")
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model_smart,
        google_api_key=settings.google_api_key,
        temperature=0.2,
        max_output_tokens=2048,
    )


def get_fast_llm() -> BaseChatModel:
    return _fast_llm_inner()


def get_smart_llm() -> BaseChatModel:
    """Smart model wrapped with auto-fallback to fast on quota/404 errors."""
    return FallbackLLM(smart=_smart_llm_inner(), fast=_fast_llm_inner())


def get_llm(tier: str = "fast") -> BaseChatModel:
    if tier == "smart":
        return get_smart_llm()
    return get_fast_llm()


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """Multilingual embeddings via Gemini (gemini-embedding-001 supports 100+ langs)."""
    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.google_api_key,
    )

