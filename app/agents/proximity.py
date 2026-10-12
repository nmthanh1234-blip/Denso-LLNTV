"""Proximity Agent — semantic similarity clustering of hypotheses.

Lightweight: uses Gemini embeddings of hypothesis titles + bodies.
"""
from __future__ import annotations
import numpy as np
from typing import List
from sklearn.cluster import AgglomerativeClustering

from app.llm.provider import get_embeddings
from app.agents.state import CoScientistState
from app.utils.logger import get_logger, log_event

logger = get_logger(__name__)


def proximity_node(state: CoScientistState) -> CoScientistState:
    with log_event(logger, "Proximity", {"n": len(state.get("hypotheses", []))}):
        hyps = state.get("hypotheses", [])
        if len(hyps) < 2:
            return {"clusters": [[h.hyp_id for h in hyps]]}

        embedder = get_embeddings()
        texts = [f"{h.title}. {h.body}" for h in hyps]
        vecs = np.array(embedder.embed_documents(texts), dtype="float32")
        # cosine distance
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        vecs = vecs / norms
        sim = vecs @ vecs.T
        dist = 1.0 - sim
        np.fill_diagonal(dist, 0.0)

        n_clusters = max(1, min(len(hyps) // 2, 3))
        cl = AgglomerativeClustering(
            n_clusters=n_clusters, metric="precomputed", linkage="average"
        )
        labels = cl.fit_predict(dist)
        clusters: List[List[str]] = [[] for _ in range(n_clusters)]
        for h, lab in zip(hyps, labels):
            clusters[int(lab)].append(h.hyp_id)
    return {"clusters": clusters}
