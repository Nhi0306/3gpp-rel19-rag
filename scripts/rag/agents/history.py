"""Agent 0: nap 5 luot hoi dap gan nhat tu SQL."""
from __future__ import annotations

import time

from config import CFG
from rag import db
from rag.memory import build_history
from rag.state import RAGState


def load_history(state: RAGState) -> dict:
    sid = state.get("session_id", "")
    if not sid:
        return {"history": [], "history_text": "", "started_at": time.time()}
    turns = db.recent_turns(sid, CFG["RAG_HISTORY_TURNS"])
    return {
        "history": turns,
        "history_text": build_history(turns, CFG["RAG_HISTORY_CHARS"]),
        "started_at": time.time(),
        "iterations": 0,
        "verify_attempts": 0,
        "refine_hint": "",
        "candidates": [],
        "trace": [],
        "usage": [],
    }
