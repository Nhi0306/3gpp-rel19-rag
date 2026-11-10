"""Cau hinh chung cho pipeline 3GPP Rel-19.

Doc bien moi truong tu .env o thu muc goc du an.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _s(key: str, default: str = "") -> str:
    return (os.getenv(key) or default).strip()


def _i(key: str, default: int) -> int:
    try:
        return int(_s(key, str(default)))
    except ValueError:
        return default


def _f(key: str, default: float) -> float:
    try:
        return float(_s(key, str(default)))
    except ValueError:
        return default


REL19_ROOT = Path(_s("REL19_ROOT", str(ROOT / "Rel-19")))

# model tong hop theo profile (chi ap dung khi khong dat SYNTH_MODEL)
_PROFILE_SYNTH = {
    "cheap": "google/gemini-2.5-flash",
    "balanced": "google/gemini-2.5-pro",
    "quality": "anthropic/claude-sonnet-4.5",
}
_PROFILE = _s("RAG_PROFILE", "balanced").lower()

CFG = {
    # qdrant
    "QDRANT_URL": _s("QDRANT_URL"),
    "QDRANT_API_KEY": _s("QDRANT_API_KEY"),
    "QDRANT_COLLECTION": _s("QDRANT_COLLECTION", "3gpp_rel19"),
    "QDRANT_DISTANCE": _s("QDRANT_DISTANCE", "Cosine"),
    "QDRANT_TIMEOUT": _i("QDRANT_TIMEOUT", 60),
    "QDRANT_UPSERT_BATCH": _i("QDRANT_UPSERT_BATCH", 128),
    "QDRANT_PREFER_GRPC": _s("QDRANT_PREFER_GRPC", "true").lower() in ("1", "true", "yes", "on"),
    # float32 (mac dinh) hoac float16 (giam ~50% byte vector khi upload)
    "QDRANT_DATATYPE": _s("QDRANT_DATATYPE", "float32").lower(),
    # tao lai ket noi gRPC sau N group (channel lau ngay bi cham dan)
    "QDRANT_RECONNECT_EVERY": _i("QDRANT_RECONNECT_EVERY", 4),
    # openrouter / openai
    "OPENROUTER_API_KEY": _s("OPENROUTER_API_KEY"),
    "OPENROUTER_BASE_URL": _s("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    "OPENROUTER_EMBED_MODEL": _s("OPENROUTER_EMBED_MODEL", "openai/text-embedding-3-small"),
    "OPENROUTER_SITE_URL": _s("OPENROUTER_SITE_URL"),
    "OPENROUTER_APP_NAME": _s("OPENROUTER_APP_NAME", "3gpp-rel19-rag"),
    "OPENAI_API_KEY": _s("OPENAI_API_KEY"),
    "OPENAI_BASE_URL": _s("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    # embedding
    "EMBED_PROVIDER": _s("EMBED_PROVIDER", "openrouter").lower(),
    "EMBED_MODEL": _s("EMBED_MODEL", "openai/text-embedding-3-small"),
    "EMBED_DIM": _i("EMBED_DIM", 1536),
    "EMBED_BATCH_SIZE": _i("EMBED_BATCH_SIZE", 64),
    "EMBED_CONCURRENCY": _i("EMBED_CONCURRENCY", 2),
    "EMBED_PARTS": _i("EMBED_PARTS", 4),
    # chunking
    "CHUNK_MAX_TOKENS": _i("CHUNK_MAX_TOKENS", 1200),
    "CHUNK_OVERLAP": _i("CHUNK_OVERLAP", 150),
    "CHUNK_MIN_TOKENS": _i("CHUNK_MIN_TOKENS", 150),
    "CHUNK_SKIP_PATTERN": _s("CHUNK_SKIP_PATTERN", "change history"),
    # pipeline
    "DOWNLOAD_WORKERS": _i("DOWNLOAD_WORKERS", 8),
    "HTTP_TIMEOUT": _i("HTTP_TIMEOUT", 60),
    "LOG_LEVEL": _s("LOG_LEVEL", "INFO").upper(),
    # ---- RAG / LLM ----
    "RAG_PROFILE": _PROFILE,
    "PLANNER_MODEL": _s("PLANNER_MODEL", "google/gemini-2.5-flash"),
    "GRADER_MODEL": _s("GRADER_MODEL", "google/gemini-2.5-flash"),
    "VERIFIER_MODEL": _s("VERIFIER_MODEL", "google/gemini-2.5-flash"),
    "SYNTH_MODEL": _s("SYNTH_MODEL") or _PROFILE_SYNTH.get(_PROFILE, _PROFILE_SYNTH["balanced"]),
    "LLM_TEMPERATURE": _f("LLM_TEMPERATURE", 0.2),
    "LLM_MAX_TOKENS": _i("LLM_MAX_TOKENS", 4096),
    "SYNTH_MAX_TOKENS": _i("SYNTH_MAX_TOKENS", 8192),
    "LLM_TIMEOUT": _i("LLM_TIMEOUT", 120),
    # retrieval
    "RAG_DENSE_TOP": _i("RAG_DENSE_TOP", 80),
    "RAG_SPARSE_TOP": _i("RAG_SPARSE_TOP", 80),
    "RAG_DENSE_WEIGHT": _f("RAG_DENSE_WEIGHT", 0.7),
    "RAG_SPARSE_WEIGHT": _f("RAG_SPARSE_WEIGHT", 0.3),
    "RAG_TOP_K": _i("RAG_TOP_K", 12),
    "RAG_MAX_SUBQUERIES": _i("RAG_MAX_SUBQUERIES", 3),
    "RAG_MAX_REFINE": _i("RAG_MAX_REFINE", 2),
    "RAG_NEIGHBOR_WINDOW": _i("RAG_NEIGHBOR_WINDOW", 1),
    "RAG_MIN_SCORE": _f("RAG_MIN_SCORE", 0.35),
    "RAG_EVIDENCE_CHARS": _i("RAG_EVIDENCE_CHARS", 1800),
    # gioi han cho web cong khai
    "RAG_RATE_PER_MIN": _i("RAG_RATE_PER_MIN", 10),
    "RAG_DAILY_LIMIT": _i("RAG_DAILY_LIMIT", 100),
    "RAG_MAX_QUESTION_CHARS": _i("RAG_MAX_QUESTION_CHARS", 2000),
    "RAG_HISTORY_TURNS": _i("RAG_HISTORY_TURNS", 5),
    "RAG_HISTORY_CHARS": _i("RAG_HISTORY_CHARS", 700),
    "RAG_DB": _s("RAG_DB", str(REL19_ROOT / "rag" / "rag.sqlite")),
    # sparse bm25
    "SPARSE_VECTOR_NAME": _s("SPARSE_VECTOR_NAME", "bm25"),
    "SPARSE_COLLECTION": _s("SPARSE_COLLECTION") or f"{_s('QDRANT_COLLECTION', '3gpp_rel19')}_sparse",
    "SPARSE_PARTS": _i("SPARSE_PARTS", 4),
    "SPARSE_UPSERT_BATCH": _i("SPARSE_UPSERT_BATCH", 128),
}


def is_placeholder(value: str) -> bool:
    """True neu gia tri con la placeholder (vd: xxxx-xxxx)."""
    v = (value or "").strip()
    if not v:
        return True
    return "<" in v or "xxxx-xxxx" in v.lower()


def validate(stage: str = "embed") -> None:
    """Kiem tra bien bat buoc theo tung giai doan."""
    required: list[str] = []
    if stage == "embed":
        if CFG["EMBED_PROVIDER"] == "openrouter":
            required.append("OPENROUTER_API_KEY")
        else:
            required.append("OPENAI_API_KEY")
        required += ["QDRANT_URL", "QDRANT_API_KEY"]
    elif stage == "qdrant":
        required += ["QDRANT_URL", "QDRANT_API_KEY"]

    problems = []
    for key in required:
        if is_placeholder(CFG[key]):
            problems.append(f"  - {key} chua duoc dien (hoac con placeholder)")

    if problems:
        print(
            "Thieu cau hinh trong .env:\n" + "\n".join(problems) +
            f"\n\nMo file: {ROOT / '.env'} va dien gia tri that.",
            file=sys.stderr,
        )
        raise SystemExit(2)
