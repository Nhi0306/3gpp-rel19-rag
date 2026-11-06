"""Duong dan + tham so rieng cho RAG."""
from __future__ import annotations

from pathlib import Path

from config import CFG, REL19_ROOT

RAG_DIR = REL19_ROOT / "rag"
RAG_DIR.mkdir(parents=True, exist_ok=True)

BM25_STATS = RAG_DIR / "bm25-stats.json"
SPARSE_PARTS_DIR = RAG_DIR / "sparse-parts"
SPARSE_CURSOR = RAG_DIR / "sparse-cursor.json"
RAG_DB = Path(CFG["RAG_DB"])
RAG_DB.parent.mkdir(parents=True, exist_ok=True)

COLLECTION = CFG["QDRANT_COLLECTION"]
SPARSE_COLLECTION = CFG["SPARSE_COLLECTION"]
VECTOR_NAME = CFG["SPARSE_VECTOR_NAME"]
