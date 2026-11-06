"""Agent 6: Persist - ghi turn + citation + log LLM vao SQLite."""
from __future__ import annotations

import time

from rag import db
from rag.agents.common import trace
from rag.memory import summarize_answer
from rag.state import RAGState
from rel19.common import log


def persist(state: RAGState) -> dict:
    sid = state.get("session_id", "")
    question = state.get("question", "")
    answer = state.get("answer", "")
    usage = state.get("usage") or []
    citations = state.get("citations") or []
    tin = sum(int(u.get("input_tokens", 0)) for u in usage)
    tout = sum(int(u.get("output_tokens", 0)) for u in usage)
    started = float(state.get("started_at") or time.time())
    latency = int((time.time() - started) * 1000)

    try:
        turn_id = db.add_turn(
            sid, question, answer, summarize_answer(answer),
            state.get("intent", ""),
            {"specs": state.get("specs", []), "section": state.get("section", "")},
            latency, tin, tout, "ok",
        )
        db.add_citations(turn_id, citations)
        for u in usage:
            db.log_llm(
                turn_id, u.get("agent", ""), u.get("model", ""),
                int(u.get("latency_ms", 0)), int(u.get("input_tokens", 0)),
                int(u.get("output_tokens", 0)), u.get("status", "ok"),
            )
        if not db.first_question(sid):
            db.touch_session(sid, question)
    except Exception as exc:  # noqa: BLE001
        log.warning("persist lỗi: %s", exc)
        turn_id = 0

    log.info("persist: turn=%s, %d ms, %d+%d token", turn_id, latency, tin, tout)
    return {
        "turn_id": turn_id,
        "trace": [trace("persist", turn_id=turn_id, latency_ms=latency,
                        tokens_in=tin, tokens_out=tout)],
    }
