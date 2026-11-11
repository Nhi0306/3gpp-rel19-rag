"""FastAPI cho he thong RAG 3GPP Rel-19 (JSON + SSE streaming)."""
from __future__ import annotations

import json
import queue
import threading
import time
from collections import defaultdict, deque
from datetime import date
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.callbacks import BaseCallbackHandler
from pydantic import BaseModel, Field

from config import CFG
from rag import db
from rag.graph import graph
from rag.stream import token_sink

WEB_DIR = Path(__file__).resolve().parent / "web"

MAX_QUESTION_CHARS = int(CFG["RAG_MAX_QUESTION_CHARS"])
RATE_PER_MIN = int(CFG["RAG_RATE_PER_MIN"])
DAILY_LIMIT = int(CFG["RAG_DAILY_LIMIT"])

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
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)


# ---- gioi han cho web cong khai (in-memory, du cho 1 instance demo) ----
_lock = threading.Lock()
_hits: dict[str, deque[float]] = defaultdict(deque)
_day = {"date": date.today(), "count": 0}


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def guard(request: Request) -> None:
    """Chan bot/spam: toi da RATE_PER_MIN cau/phut moi IP, DAILY_LIMIT cau/ngay."""
    now = time.time()
    ip = _client_ip(request)
    with _lock:
        today = date.today()
        if _day["date"] != today:
            _day["date"], _day["count"] = today, 0
        if _day["count"] >= DAILY_LIMIT:
            raise HTTPException(
                status_code=429,
                detail="Đã đạt giới hạn câu hỏi trong ngày. Vui lòng quay lại ngày mai.",
            )
        hits = _hits[ip]
        while hits and now - hits[0] > 60:
            hits.popleft()
        if len(hits) >= RATE_PER_MIN:
            raise HTTPException(
                status_code=429,
                detail="Bạn hỏi quá nhanh. Vui lòng thử lại sau ít phút.",
            )
        hits.append(now)
        _day["count"] += 1


def _quota() -> dict:
    with _lock:
        return {
            "used": _day["count"],
            "limit": DAILY_LIMIT,
            "rate_per_min": RATE_PER_MIN,
            "max_chars": MAX_QUESTION_CHARS,
        }


def _session(sid: str | None) -> str:
    if sid:
        return sid
    return db.create_session()


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


NODE_LABELS = {
    "history": "Nạp lịch sử hội thoại",
    "supervisor": "Phân tích câu hỏi",
    "retrieve": "Tìm tài liệu (dense + BM25)",
    "merge": "Gộp kết quả",
    "grader": "Chấm độ liên quan",
    "synthesizer": "Viết câu trả lời",
    "verifier": "Kiểm tra trích dẫn",
    "persist": "Lưu phiên",
}


class StepHandler(BaseCallbackHandler):
    """Ban su kien 'step' cho UI theo tung node cua LangGraph.

    Chi doc metadata cua callback, khong goi them LLM nen khong tang chi phi.
    """

    def __init__(self, events: queue.Queue) -> None:
        super().__init__()
        self.events = events
        self.runs: dict[str, dict] = {}
        self.round = 0

    def _send(self, payload: dict) -> None:
        self.events.put(("step", payload))

    def on_chain_start(self, serialized, inputs, **kwargs):  # noqa: ANN001
        name = kwargs.get("name")
        node = (kwargs.get("metadata") or {}).get("langgraph_node")
        if not name or name != node or node not in NODE_LABELS:
            return
        rid = str(kwargs.get("run_id"))
        total = None
        if isinstance(inputs, dict) and isinstance(inputs.get("evidence"), list):
            total = len(inputs["evidence"])
        self.runs[rid] = {"node": node, "started": time.time(), "total": total}
        self._send({"node": node, "phase": "start", "round": self.round, "run_id": rid})

    def on_chain_error(self, error, **kwargs):  # noqa: ANN001
        rid = str(kwargs.get("run_id"))
        info = self.runs.pop(rid, None)
        if not info:
            return
        self._send({
            "node": info["node"], "phase": "end", "round": self.round, "run_id": rid,
            "ms": int((time.time() - info["started"]) * 1000),
            "meta": {"error": str(error)[:300]},
        })

    def on_chain_end(self, outputs, **kwargs):  # noqa: ANN001
        rid = str(kwargs.get("run_id"))
        info = self.runs.pop(rid, None)
        if not info:
            return
        node = info["node"]
        out = outputs if isinstance(outputs, dict) else {}
        meta: dict = {}
        if node == "history":
            meta = {"turns": len(out.get("history") or [])}
        elif node == "supervisor":
            meta = {"specs": out.get("specs") or [], "sub_queries": out.get("sub_queries") or []}
            self.round = int(out.get("iterations") or self.round)
        elif node == "retrieve":
            tr = (out.get("trace") or [{}])[-1]
            meta = {"hits": len(out.get("candidates") or []), "sub_query": tr.get("query") or ""}
            self.round = int(tr.get("round") or self.round)
        elif node == "merge":
            meta = {"evidence": len(out.get("evidence") or [])}
            tr = (out.get("trace") or [{}])[-1]
            self.round = int(tr.get("round") or self.round)
        elif node == "grader":
            tr = (out.get("trace") or [{}])[-1]
            meta = {
                "kept": tr.get("kept", 0), "total": info.get("total"),
                "sufficient": bool(out.get("sufficient")),
            }
        elif node == "synthesizer":
            meta = {"chars": len(out.get("answer") or ""), "citations": len(out.get("citations") or [])}
        elif node == "verifier":
            verdict = out.get("verdict") or {}
            meta = {"ok": bool(verdict.get("ok")), "issues": verdict.get("issues") or []}
        elif node == "persist":
            meta = {"turn_id": out.get("turn_id")}
        self._send({
            "node": node, "phase": "end", "round": self.round, "run_id": rid,
            "ms": int((time.time() - info["started"]) * 1000), "meta": meta,
        })


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
        "quota": _quota(),
    }


@app.get("/api/diag")
def diag() -> dict:
    """Chan doan nhanh cau hinh tren server (khong lo gia tri secret)."""
    out: dict = {
        "env": {
            k: bool(CFG.get(k))
            for k in ("QDRANT_URL", "QDRANT_API_KEY", "OPENROUTER_API_KEY")
        },
        "profile": CFG["RAG_PROFILE"],
    }
    try:
        from rag.paths import BM25_STATS
        info = {"path": str(BM25_STATS), "exists": BM25_STATS.exists()}
        if BM25_STATS.exists():
            info["mb"] = round(BM25_STATS.stat().st_size / 1e6, 1)
            from rag import retrieval
            info["N"] = retrieval.stats().n_docs
        out["bm25"] = info
    except Exception as exc:  # noqa: BLE001
        out["bm25"] = {"error": str(exc)}
    try:
        from rag import retrieval
        cli = retrieval.client()
        cols = [c.name for c in cli.get_collections().collections]
        out["qdrant"] = {
            "collections": cols,
            "dense_count": cli.count(CFG["QDRANT_COLLECTION"], exact=False).count,
        }
    except Exception as exc:  # noqa: BLE001
        out["qdrant"] = {"error": str(exc)}
    return out


@app.get("/api/diag/search", dependencies=[Depends(guard)])
def diag_search(q: str) -> dict:
    """Chay thu retrieval cho mot cau hoi (ton 1 lan embedding)."""
    from rag import retrieval
    try:
        hits = retrieval.search(q)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}
    return {
        "ok": True,
        "hits": len(hits),
        "top": [
            {
                "spec": (h.get("payload") or {}).get("spec"),
                "section": (h.get("payload") or {}).get("section"),
                "score": round(h.get("score", 0), 3),
                "source": h.get("source"),
            }
            for h in hits[:5]
        ],
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
    return {"id": sid, "turns": turns, "calls": db.session_llm_calls(sid)}


@app.delete("/api/sessions/{sid}")
def delete_session(sid: str) -> dict:
    return {"ok": db.delete_session(sid)}


@app.post("/api/chat", dependencies=[Depends(guard)])
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


@app.post("/api/chat/stream", dependencies=[Depends(guard)])
def chat_stream(body: ChatIn) -> StreamingResponse:
    sid = _session(body.session_id)
    events: queue.Queue = queue.Queue()

    def run() -> None:
        token_sink.set(lambda text: events.put(("token", text)))
        try:
            state = graph().invoke(
                {"session_id": sid, "question": body.question},
                config={"recursion_limit": 40, "callbacks": [StepHandler(events)]},
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
            elif kind == "step":
                yield _sse({"type": "step", **data})
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
