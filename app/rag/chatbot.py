"""RAG chatbot — answers user queries with citations.

Uses the meta-review report as the primary context, with the retrieved
chunks as ground truth fallback. Answer is constrained to the context.
"""
from __future__ import annotations
from langchain_core.messages import SystemMessage, HumanMessage

from app.llm.provider import get_fast_llm
from app.utils.logger import get_logger
from app.rag.retriever import RAGRetriever
from app.utils.schemas import MetaReview

logger = get_logger(__name__)


ANSWER_SYS = (
    "You are a document-grounded RAG assistant. "
    "Answer the user's question ONLY using the provided CONTEXT. "
    "If the answer is not in the context, say you don't know. "
    "Always cite sources like [source p.X]."
)


def answer_query(
    query: str,
    retriever: RAGRetriever,
    *,
    meta: MetaReview | None = None,
    k: int = 5,
) -> str:
    """Generate a grounded answer."""
    ctx = retriever.retrieve_context(query, k=k, max_chars=3500)
    if meta:
        ctx = (
            "## Research overview (from co-scientist)\n"
            + meta.report_markdown[:2000]
            + "\n\n---\n\n## Retrieved chunks\n"
            + ctx
        )

    llm = get_fast_llm()
    resp = llm.invoke(
        [
            SystemMessage(content=ANSWER_SYS),
            HumanMessage(content=f"CONTEXT:\n{ctx}\n\nQUESTION:\n{query}\n\nANSWER:"),
        ]
    )
    # Gemini 2.5+ may return content as list[dict]
    if isinstance(resp.content, str):
        return resp.content
    if isinstance(resp.content, list):
        return "\n".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in resp.content
        )
    return str(resp.content)
