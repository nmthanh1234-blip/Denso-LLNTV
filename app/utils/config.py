"""Centralised settings — pydantic-based, reads from .env / env vars."""
from __future__ import annotations
import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- LLM ---
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    gemini_model_fast: str = Field(default="gemini-3.8-flash", alias="GEMINI_MODEL_FAST")
    gemini_model_smart: str = Field(default="gemini-3.1-pro-preview", alias="GEMINI_MODEL_SMART")
    embedding_model: str = Field(default="models/gemini-embedding-001", alias="EMBEDDING_MODEL")

    # --- Paths ---
    vector_store_dir: str = Field(default="./data/vector_store", alias="VECTOR_STORE_DIR")
    upload_dir: str = Field(default="./data/uploads", alias="UPLOAD_DIR")

    # --- Reflection ---
    max_reflection_parallel: int = Field(default=4, alias="MAX_REFLECTION_PARALLEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    def ensure_dirs(self) -> None:
        Path(self.vector_store_dir).mkdir(parents=True, exist_ok=True)
        Path(self.upload_dir).mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
