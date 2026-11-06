"""FastAPI cho he thong RAG 3GPP Rel-19 (JSON + SSE streaming)."""
from __future__ import annotations

import json
import queue
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import CFG
from rag import db
from rag.graph import graph
from rag.stream import token_sink

WEB_DIR = Path(__file__).resolve().parent / "web"

app = FastAPI(title="3GPP Rel-19 RAG", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

db.init_db()


class ChatIn(BaseModel):
    session_id: str | None = None
    question: str = Field(min_length=1, max_length=4000)


def _session(sid: str | None) -> str:
    if sid:
        return sid
    return db.create_session()


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _result(state: dict, sid: str) -> dict:
    return {
        "session_id": sid,
        "turn_id": state.get("turn_id"),
        "answer": state.get("answer", ""),
        "citations": state.get("citations") or [],
        "iterations": state.get("iterations", 0),
        "verdict": state.get("verdict") or {},
        "trace": state.get("trace") or [],
        "usage": state.get("usage") or [],
    }


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "collection": CFG["QDRANT_COLLECTION"],
        "sparse_collection": CFG["SPARSE_COLLECTION"],
        "profile": CFG["RAG_PROFILE"],
        "synth_model": CFG["SYNTH_MODEL"],
        "db": str(CFG["RAG_DB"]),
    }


@app.post("/api/sessions")
def create_session() -> dict:
    return {"id": db.create_session()}


@app.get("/api/sessions")
def list_sessions() -> list[dict]:
    return db.list_sessions(50)


@app.get("/api/sessions/{sid}")
def get_session(sid: str) -> dict:
    turns = db.session_turns(sid)
    return {"id": sid, "turns": turns}


@app.post("/api/chat")
def chat(body: ChatIn) -> dict:
    sid = _session(body.session_id)
    try:
        state = graph().invoke(
            {"session_id": sid, "question": body.question},
            config={"recursion_limit": 40},
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return _result(state, sid)


@app.post("/api/chat/stream")
def chat_stream(body: ChatIn) -> StreamingResponse:
    sid = _session(body.session_id)
    events: queue.Queue = queue.Queue()

    def run() -> None:
        token_sink.set(lambda text: events.put(("token", text)))
        try:
            state = graph().invoke(
                {"session_id": sid, "question": body.question},
                config={"recursion_limit": 40},
            )
            events.put(("done", state))
        except Exception as exc:  # noqa: BLE001
            events.put(("error", str(exc)))

    threading.Thread(target=run, daemon=True).start()

    def gen():
        yield _sse({"type": "session", "session_id": sid})
        while True:
            kind, data = events.get()
            if kind == "token":
                yield _sse({"type": "token", "text": data})
            elif kind == "error":
                yield _sse({"type": "error", "message": data})
                return
            else:
                yield _sse({"type": "done", **_result(data, sid)})
                return

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
