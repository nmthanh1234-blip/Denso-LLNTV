"""RAG retriever — query -> relevant chunks (fetch-once, process-locally)."""
from __future__ import annotations
from typing import List, Tuple
import numpy as np

from app.rag.store import VectorStore
from app.utils.schemas import DocumentChunk
from app.llm.provider import get_embeddings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RAGRetriever:
    def __init__(self, store: VectorStore) -> None:
        self.store = store
        self.embedder = get_embeddings()

    def embed_query(self, text: str) -> np.ndarray:
        vec = np.array(self.embedder.embed_query(text), dtype="float32")
        return vec

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        q = self.embed_query(query)
        return self.store.search(q, k=k)

    def retrieve_context(self, query: str, k: int = 5, max_chars: int = 4000) -> str:
        """Build a compressed context string (Context Stratification)."""
        results = self.retrieve(query, k=k)
        blocks = []
        used = 0
        for c, score in results:
            tag = f"[{c.source} p{c.page or '-'} | {c.modality}/{c.lang} | sim={score:.2f}]"
            piece = f"{tag}\n{c.text}"
            if used + len(piece) > max_chars:
                piece = piece[: max_chars - used]
            blocks.append(piece)
            used += len(piece)
            if used >= max_chars:
                break
        return "\n\n---\n\n".join(blocks)
