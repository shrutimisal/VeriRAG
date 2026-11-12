"""Central configuration. All settings come from environment variables / .env.

Settings are built by `get_settings()` at call time so edits to the environment
(or .env) are picked up without restarting the interpreter.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Exact sentence the LLM must output when the context has no answer.
NOT_FOUND_MESSAGE = "I could not find the answer to this question in the uploaded documents."

CLAIM_LABELS = ("Supported", "Partially Supported", "Unsupported", "Contradicted")


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else PROJECT_ROOT / p


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    gemini_model: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    chroma_dir: Path
    documents_dir: Path
    eval_dir: Path
    ragas_relevancy_strictness: int


def get_settings() -> Settings:
    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip(),
        embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2").strip(),
        chunk_size=_int("CHUNK_SIZE", 1000),
        chunk_overlap=_int("CHUNK_OVERLAP", 200),
        top_k=_int("TOP_K", 5),
        chroma_dir=_path(os.getenv("CHROMA_DIR", "data/chroma_db")),
        documents_dir=PROJECT_ROOT / "data" / "documents",
        eval_dir=PROJECT_ROOT / "data" / "evaluation",
        ragas_relevancy_strictness=_int("RAGAS_RELEVANCY_STRICTNESS", 1),
    )
