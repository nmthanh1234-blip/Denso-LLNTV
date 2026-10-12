"""Multi-format document ingest pipeline.

Supported:
  - .pdf  -> text + tables (pdfplumber) + page image OCR (Tesseract via pytesseract)
  - .png/.jpg/.jpeg  -> VLM caption (Gemini) + OCR fallback
  - .csv/.xlsx       -> table-as-text chunks
  - .txt/.md         -> plain text chunks

Each chunk is a `DocumentChunk` with modality + lang tags so the RAG layer
can filter appropriately (bilingual handling).
"""
from __future__ import annotations
import os
import re
import uuid
from pathlib import Path
from typing import List

import pdfplumber
import pymupdf as fitz  # PyMuPDF
from PIL import Image
import pytesseract

from app.utils.schemas import DocumentChunk
from app.utils.logger import get_logger

logger = get_logger(__name__)

VI_DIACRITICS = re.compile(r"[ăâđêôơưĂÂĐÊÔƠƯáàảãạằẳẵặắấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]")


def _detect_lang(text: str) -> str:
    """Crude Vietnamese detector. Returns 'vi' | 'en' | 'mixed'."""
    if not text:
        return "en"
    vi = len(VI_DIACRITICS.findall(text))
    total = max(1, len(text))
    ratio = vi / total
    if ratio > 0.02:
        # some Vietnamese chars present
        ascii_ratio = sum(1 for c in text if c.isascii() and c.isalpha()) / total
        return "vi" if ascii_ratio < 0.7 else "mixed"
    return "en"


def _chunk_text(text: str, max_chars: int = 800) -> List[str]:
    """Naive paragraph + length-based chunker (no prose bloat)."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts, buf, size = [], [], 0
    for sent in re.split(r"(?<=[\.\!\?])\s+", text):
        if size + len(sent) > max_chars and buf:
            parts.append(" ".join(buf).strip())
            buf, size = [sent], len(sent)
        else:
            buf.append(sent)
            size += len(sent)
    if buf:
        parts.append(" ".join(buf).strip())
    return [p for p in parts if len(p) > 20]


# ---------- per-format ingestors ----------

def _ingest_pdf(path: Path, doc_id: str) -> List[DocumentChunk]:
    chunks: List[DocumentChunk] = []
    # 1) text + tables via pdfplumber
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = (page.extract_text() or "").strip()
            if text:
                for piece in _chunk_text(text):
                    chunks.append(
                        DocumentChunk(
                            chunk_id=str(uuid.uuid4()),
                            doc_id=doc_id,
                            source=path.name,
                            page=i + 1,
                            modality="text",
                            lang=_detect_lang(piece),
                            text=piece,
                            meta={"extractor": "pdfplumber"},
                        )
                    )
            # tables
            try:
                tables = page.extract_tables() or []
                for t_idx, t in enumerate(tables):
                    if not t:
                        continue
                    header = " | ".join(str(c) for c in t[0]) if t else ""
                    rows = [" | ".join(str(c) for c in r) for r in t[1:]]
                    table_text = "TABLE " + header + "\n" + "\n".join(rows)
                    chunks.append(
                        DocumentChunk(
                            chunk_id=str(uuid.uuid4()),
                            doc_id=doc_id,
                            source=path.name,
                            page=i + 1,
                            modality="table",
                            lang=_detect_lang(table_text),
                            text=table_text,
                            meta={"extractor": "pdfplumber", "table_idx": t_idx},
                        )
                    )
            except Exception as e:
                logger.warning(f"pdfplumber table extract failed on {path.name} p{i+1}: {e}")

    # 2) OCR for pages with very little text (image-based PDFs)
    try:
        doc = fitz.open(path)
        for i, page in enumerate(doc):
            text = page.get_text("text").strip()
            if len(text) < 30:  # likely scanned page
                pix = page.get_pixmap(dpi=200)
                img_path = path.with_suffix(f".p{i+1}.png")
                pix.save(str(img_path))
                ocr_text = pytesseract.image_to_string(Image.open(img_path), lang="eng+vie")
                if ocr_text.strip():
                    for piece in _chunk_text(ocr_text):
                        chunks.append(
                            DocumentChunk(
                                chunk_id=str(uuid.uuid4()),
                                doc_id=doc_id,
                                source=path.name,
                                page=i + 1,
                                modality="ocr",
                                lang=_detect_lang(piece),
                                text=piece,
                                meta={"extractor": "tesseract"},
                            )
                        )
                try:
                    img_path.unlink()
                except OSError:
                    pass
    except Exception as e:
        logger.warning(f"OCR fallback skipped for {path.name}: {e}")

    return chunks


def _ingest_image(path: Path, doc_id: str) -> List[DocumentChunk]:
    chunks: List[DocumentChunk] = []
    try:
        # OCR first (cheap, local)
        ocr_text = pytesseract.image_to_string(Image.open(path), lang="eng+vie").strip()
        if ocr_text:
            for piece in _chunk_text(ocr_text):
                chunks.append(
                    DocumentChunk(
                        chunk_id=str(uuid.uuid4()),
                        doc_id=doc_id,
                        source=path.name,
                        modality="ocr",
                        lang=_detect_lang(piece),
                        text=piece,
                        meta={"extractor": "tesseract"},
                    )
                )
        # Caption via Gemini (VLM-style) - only if we have a key
        from app.utils.config import settings
        if settings.google_api_key:
            try:
                from app.llm.provider import get_fast_llm
                import base64
                from langchain_core.messages import HumanMessage
                llm = get_fast_llm()
                b64 = base64.b64encode(path.read_bytes()).decode()
                msg = HumanMessage(
                    content=[
                        {"type": "text", "text": "Describe the image in 1-2 sentences. Focus on text, tables, or diagrams."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ]
                )
                cap = llm.invoke([msg]).content.strip()
                if cap:
                    chunks.append(
                        DocumentChunk(
                            chunk_id=str(uuid.uuid4()),
                            doc_id=doc_id,
                            source=path.name,
                            modality="image_caption",
                            lang=_detect_lang(cap),
                            text=cap,
                            meta={"extractor": "gemini-vlm"},
                        )
                    )
            except Exception as e:
                logger.warning(f"VLM caption skipped for {path.name}: {e}")
    except Exception as e:
        logger.warning(f"Image ingest failed for {path.name}: {e}")
    return chunks


def _ingest_table(path: Path, doc_id: str) -> List[DocumentChunk]:
    """CSV / XLSX -> one chunk per ~50 rows."""
    chunks: List[DocumentChunk] = []
    try:
        if path.suffix.lower() == ".csv":
            import csv
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f)
                rows = list(reader)
        else:
            from openpyxl import load_workbook
            wb = load_workbook(path, read_only=True, data_only=True)
            ws = wb.active
            rows = [list(r) for r in ws.iter_rows(values_only=True)]
        if not rows:
            return []
        header = " | ".join(str(c) for c in rows[0])
        for i in range(1, len(rows), 50):
            block = rows[i : i + 50]
            block_text = f"TABLE {header}\n" + "\n".join(
                " | ".join(str(c) for c in r) for r in block
            )
            chunks.append(
                DocumentChunk(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    source=path.name,
                    modality="table",
                    lang="en",
                    text=block_text,
                    meta={"extractor": "tabular", "row_start": i},
                )
            )
    except Exception as e:
        logger.warning(f"Table ingest failed for {path.name}: {e}")
    return chunks


def _ingest_text(path: Path, doc_id: str) -> List[DocumentChunk]:
    chunks: List[DocumentChunk] = []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for piece in _chunk_text(text):
            chunks.append(
                DocumentChunk(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    source=path.name,
                    modality="text",
                    lang=_detect_lang(piece),
                    text=piece,
                    meta={"extractor": "raw"},
                )
            )
    except Exception as e:
        logger.warning(f"Text ingest failed for {path.name}: {e}")
    return chunks


# ---------- public API ----------

def ingest_file(path: str | os.PathLike) -> List[DocumentChunk]:
    p = Path(path)
    doc_id = p.stem
    suffix = p.suffix.lower()
    logger.info(f"Ingesting {p.name} (suffix={suffix})")
    if suffix == ".pdf":
        return _ingest_pdf(p, doc_id)
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}:
        return _ingest_image(p, doc_id)
    if suffix in {".csv", ".xlsx", ".xls"}:
        return _ingest_table(p, doc_id)
    if suffix in {".txt", ".md"}:
        return _ingest_text(p, doc_id)
    logger.warning(f"Unsupported file type: {suffix}")
    return []


def ingest_paths(paths: List[str]) -> List[DocumentChunk]:
    out: List[DocumentChunk] = []
    for p in paths:
        out.extend(ingest_file(p))
    logger.info(f"Total chunks ingested: {len(out)}")
    return out
