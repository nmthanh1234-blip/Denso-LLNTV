"""Ingest pipeline: PDF/Image/Table/Bilingual -> structured chunks."""
from .pipeline import ingest_file, ingest_paths

__all__ = ["ingest_file", "ingest_paths"]
