"""Streamlit demo UI: upload -> ingest -> co-scientist -> chatbot."""
from __future__ import annotations
import os
import tempfile
import time
from pathlib import Path
import numpy as np
import streamlit as st

from app.ingest.pipeline import ingest_paths
from app.llm.provider import get_embeddings
from app.rag.store import VectorStore
from app.rag.retriever import RAGRetriever
from app.agents.graph import run_scientist
from app.rag.chatbot import answer_query
from app.utils.logger import METRICS, get_logger

logger = get_logger(__name__)
st.set_page_config(page_title="Co-Scientist RAG", page_icon="🧪", layout="wide")


# ----------------- session state -----------------
def _init_state() -> None:
    st.session_state.setdefault("store", VectorStore())
    if not st.session_state.store.chunks:
        st.session_state.store.load()
    st.session_state.setdefault("retriever", RAGRetriever(st.session_state.store))
    st.session_state.setdefault("history", [])


_init_state()


# ----------------- helpers -----------------
def _ingest_uploads(files) -> int:
    paths = []
    for f in files:
        suffix = Path(f.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(f.read())
            paths.append(tmp.name)
    chunks = ingest_paths(paths)
    if not chunks:
        return 0
    embedder = get_embeddings()
    # batch embed
    vecs = np.array(embedder.embed_documents([c.text for c in chunks]), dtype="float32")
    st.session_state.store.add(chunks, vecs)
    st.session_state.store.save()
    return len(chunks)


# ----------------- UI -----------------
st.title("🧪 Co-Scientist Multi-Agent RAG")
st.caption(
    "Diverse documents (PDF / ảnh / bảng / song ngữ) → 7 co-scientist agents → câu trả lời có trích dẫn"
)

with st.sidebar:
    st.header("📁 Ingest")
    files = st.file_uploader(
        "Tải tài liệu (PDF / ảnh / CSV / TXT / MD)",
        type=["pdf", "png", "jpg", "jpeg", "csv", "xlsx", "txt", "md"],
        accept_multiple_files=True,
    )
    if st.button("Ingest & Index", disabled=not files):
        with st.spinner("Đang ingest + embed..."):
            n = _ingest_uploads(files)
        st.success(f"Indexed {n} chunks. Total: {len(st.session_state.store.chunks)}")
    st.divider()
    st.metric("Chunks", len(st.session_state.store.chunks))
    st.metric("LLM calls", METRICS.summary().get("total_calls", 0))
    st.metric("Tokens (approx)", METRICS.summary().get("total_tokens", 0))

tabs = st.tabs(["💬 Chatbot (RAG)", "🧬 Co-Scientist run", "📊 Metrics"])

# --- tab 1: chatbot ---
with tabs[0]:
    st.subheader("Hỏi đáp có trích dẫn")
    q = st.text_input("Câu hỏi của bạn", key="q")
    use_meta = st.checkbox("Dùng Research Overview từ Co-Scientist", value=False)
    if st.button("Hỏi", key="ask") and q:
        with st.spinner("Đang truy vấn..."):
            t0 = time.perf_counter()
            meta = st.session_state.get("last_meta")
            ans = answer_query(q, st.session_state.retriever, meta=meta if use_meta else None)
            dt = (time.perf_counter() - t0) * 1000
        st.session_state.history.append((q, ans, dt))
        st.markdown(ans)
        with st.expander("Top-K retrieved"):
            for c, s in st.session_state.retriever.retrieve(q, k=5):
                st.markdown(
                    f"- **[{c.source} p{c.page or '-'} | {c.modality}/{c.lang} | sim={s:.2f}]** {c.text[:200]}..."
                )

    for q_, a_, dt_ in reversed(st.session_state.history[:-1]):
        with st.expander(f"Q: {q_[:80]}  •  {dt_:.0f}ms"):
            st.markdown(a_)

# --- tab 2: co-scientist ---
with tabs[1]:
    st.subheader("Chạy vòng Co-Scientist (Generation → Reflection → Ranking → Evolution → Meta-Review)")
    goal = st.text_area(
        "Mục tiêu nghiên cứu",
        value="Xác định các rủi ro tài chính chính trong tài liệu và đề xuất biện pháp phòng ngừa",
    )
    lang = st.selectbox("Ngôn ngữ", ["auto", "vi", "en"], index=0)
    nrounds = st.slider("Số vòng", 1, 4, 2)
    nhyp = st.slider("Số hypotheses mỗi vòng", 2, 6, 4)

    if st.button("🚀 Run Co-Scientist", disabled=len(st.session_state.store.chunks) == 0):
        # fetch context for the goal
        ctx = st.session_state.retriever.retrieve_context(goal, k=8)
        t0 = time.perf_counter()
        with st.spinner("Co-Scientist đang chạy..."):
            result = run_scientist(
                goal=goal,
                context_text=ctx,
                language=lang,
                retrieved_chunks=st.session_state.store.chunks,
                config={"num_hypotheses": nhyp, "max_rounds": nrounds, "focus": "general"},
            )
        dt = (time.perf_counter() - t0) * 1000
        st.session_state["last_meta"] = result.get("meta")
        st.success(f"Hoàn thành trong {dt/1000:.1f}s")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🥇 Top-ranked")
            for r in result.get("ranked", []):
                st.markdown(f"- **{r.title}** — Elo `{r.elo:.0f}`")
        with col2:
            st.markdown("### 🔁 Reflections")
            for rpt in result.get("reflections", []):
                st.markdown(
                    f"- `{rpt.hyp_id}` • novelty={rpt.novelty:.2f} • "
                    f"correct={rpt.correctness:.2f} • feas={rpt.feasibility:.2f}"
                )

        st.markdown("### 📄 Research Overview")
        if result.get("meta"):
            st.markdown(result["meta"].report_markdown)

# --- tab 3: metrics ---
with tabs[2]:
    s = METRICS.summary()
    st.json(s)
    if METRICS.events:
        st.markdown("### Per-agent breakdown")
        from collections import Counter
        per = Counter()
        for e in METRICS.events:
            per[e["agent"]] += 1
        st.bar_chart(per)
