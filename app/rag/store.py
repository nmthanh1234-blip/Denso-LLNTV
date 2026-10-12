"""FAISS-backed vector store with persistent on-disk index."""
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

from app.utils.config import settings
from app.utils.schemas import DocumentChunk
from app.utils.logger import get_logger

logger = get_logger(__name__)


class VectorStore:
    """Thin wrapper over numpy + FAISS-lite (cosine via L2-normalised vectors)."""

    def __init__(self, dim: int = 3072) -> None:
        self.dim = dim
        self.vectors: np.ndarray = np.zeros((0, dim), dtype="float32")
        self.chunks: List[DocumentChunk] = []
        self.path = Path(settings.vector_store_dir)
        self.path.mkdir(parents=True, exist_ok=True)

    # ---------- ingest ----------
    def add(self, chunks: List[DocumentChunk], embeddings: np.ndarray) -> None:
        assert embeddings.shape[1] == self.dim
        # normalise for cosine
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
        embeddings = embeddings / norms
        self.vectors = np.vstack([self.vectors, embeddings.astype("float32")])
        self.chunks.extend(chunks)
        logger.info(f"VectorStore now has {len(self.chunks)} chunks")

    # ---------- query ----------
    def search(self, query_vec: np.ndarray, k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        if len(self.chunks) == 0:
            return []
        q = query_vec.reshape(1, -1).astype("float32")
        q = q / (np.linalg.norm(q) + 1e-9)
        scores = (self.vectors @ q.T).ravel()
        top_k = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top_k]

    # ---------- persistence ----------
    def save(self) -> None:
        np.save(self.path / "vectors.npy", self.vectors)
        with open(self.path / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(c.model_dump_json() + "\n")
        logger.info(f"VectorStore saved to {self.path}")

    def load(self) -> bool:
        v = self.path / "vectors.npy"
        c = self.path / "chunks.jsonl"
        if not (v.exists() and c.exists()):
            return False
        self.vectors = np.load(v)
        self.chunks = []
        with open(c, "r", encoding="utf-8") as f:
            for line in f:
                self.chunks.append(DocumentChunk.model_validate_json(line))
        self.dim = self.vectors.shape[1] if self.vectors.size else self.dim
        logger.info(f"VectorStore loaded {len(self.chunks)} chunks from {self.path}")
        return True
