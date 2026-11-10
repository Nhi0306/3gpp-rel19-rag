"""SQLite luu phien hoi dap, turn, citation va log LLM."""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime

from rag.paths import RAG_DB

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    title       TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    meta        TEXT
);

CREATE TABLE IF NOT EXISTS turns (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id     TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    turn_index     INTEGER NOT NULL,
    question       TEXT NOT NULL,
    answer         TEXT,
    answer_summary TEXT,
    intent         TEXT,
    filters        TEXT,
    latency_ms     INTEGER,
    tokens_in      INTEGER,
    tokens_out     INTEGER,
    status         TEXT,
    created_at     TEXT NOT NULL,
    UNIQUE(session_id, turn_index)
);

CREATE TABLE IF NOT EXISTS citations (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    turn_id      INTEGER NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
    rank         INTEGER NOT NULL,
    chunk_id     TEXT,
    spec         TEXT,
    version      TEXT,
    section      TEXT,
    section_path TEXT,
    title        TEXT,
    score        REAL,
    source       TEXT,
    snippet      TEXT
);

CREATE TABLE IF NOT EXISTS llm_calls (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    turn_id     INTEGER,
    agent       TEXT,
    model       TEXT,
    latency_ms  INTEGER,
    tokens_in   INTEGER,
    tokens_out  INTEGER,
    status      TEXT,
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id, turn_index);
CREATE INDEX IF NOT EXISTS idx_citations_turn ON citations(turn_id);
CREATE INDEX IF NOT EXISTS idx_llm_turn ON llm_calls(turn_id);
"""


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def connect() -> sqlite3.Connection:
    con = sqlite3.connect(RAG_DB, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    return con


def init_db() -> None:
    with connect() as con:
        con.executescript(SCHEMA)


def create_session(title: str = "", meta: dict | None = None) -> str:
    sid = str(uuid.uuid4())
    with connect() as con:
        con.execute(
            "INSERT INTO sessions(id, title, created_at, updated_at, meta) VALUES(?,?,?,?,?)",
            (sid, (title or "Phiên mới")[:120], now(), now(),
             json.dumps(meta or {}, ensure_ascii=False)),
        )
    return sid


def touch_session(sid: str, title: str | None = None) -> None:
    with connect() as con:
        if title:
            con.execute("UPDATE sessions SET updated_at=?, title=? WHERE id=?", (now(), title[:120], sid))
        else:
            con.execute("UPDATE sessions SET updated_at=? WHERE id=?", (now(), sid))


def delete_session(sid: str) -> bool:
    """Xoa phien, toan bo turn/citation va log LLM cua phien."""
    with connect() as con:
        ids = [r["id"] for r in con.execute(
            "SELECT id FROM turns WHERE session_id=?", (sid,)
        ).fetchall()]
        if ids:
            marks = ",".join("?" * len(ids))
            con.execute(f"DELETE FROM llm_calls WHERE turn_id IN ({marks})", ids)
        cur = con.execute("DELETE FROM sessions WHERE id=?", (sid,))
        return cur.rowcount > 0


def list_sessions(limit: int = 50) -> list[dict]:
    with connect() as con:
        rows = con.execute(
            """SELECT s.id, s.title, s.created_at, s.updated_at,
                      (SELECT COUNT(*) FROM turns t WHERE t.session_id = s.id) AS turns
               FROM sessions s ORDER BY s.updated_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def session_turns(sid: str) -> list[dict]:
    with connect() as con:
        turns = [dict(r) for r in con.execute(
            "SELECT * FROM turns WHERE session_id=? ORDER BY turn_index", (sid,)
        ).fetchall()]
        for t in turns:
            t["citations"] = [dict(c) for c in con.execute(
                "SELECT * FROM citations WHERE turn_id=? ORDER BY rank", (t["id"],)
            ).fetchall()]
    return turns


def recent_turns(sid: str, limit: int = 5) -> list[dict]:
    with connect() as con:
        rows = con.execute(
            """SELECT question, answer, answer_summary, turn_index
               FROM turns WHERE session_id=? AND status != 'error'
               ORDER BY turn_index DESC LIMIT ?""",
            (sid, limit),
        ).fetchall()
    return [dict(r) for r in reversed(rows)]


def add_turn(
    sid: str, question: str, answer: str, summary: str, intent: str,
    filters: dict, latency_ms: int, tokens_in: int, tokens_out: int, status: str = "ok",
) -> int:
    with connect() as con:
        row = con.execute(
            "SELECT COALESCE(MAX(turn_index), 0) + 1 FROM turns WHERE session_id=?", (sid,)
        ).fetchone()
        idx = int(row[0])
        cur = con.execute(
            """INSERT INTO turns(session_id, turn_index, question, answer, answer_summary,
                                 intent, filters, latency_ms, tokens_in, tokens_out, status, created_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            (sid, idx, question, answer, summary, intent,
             json.dumps(filters, ensure_ascii=False), latency_ms, tokens_in, tokens_out,
             status, now()),
        )
        con.execute("UPDATE sessions SET updated_at=? WHERE id=?", (now(), sid))
        return int(cur.lastrowid)


def add_citations(turn_id: int, citations: list[dict]) -> None:
    if not citations:
        return
    with connect() as con:
        con.executemany(
            """INSERT INTO citations(turn_id, rank, chunk_id, spec, version, section,
                                     section_path, title, score, source, snippet)
               VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            [
                (turn_id, c.get("rank", 0), c.get("chunk_id", ""), c.get("spec", ""),
                 c.get("version", ""), c.get("section", ""), c.get("section_path", ""),
                 c.get("title", ""), float(c.get("score", 0)), c.get("source", ""),
                 c.get("snippet", ""))
                for c in citations
            ],
        )


def log_llm(turn_id: int, agent: str, model: str, latency_ms: int,
            tokens_in: int, tokens_out: int, status: str = "ok") -> None:
    with connect() as con:
        con.execute(
            """INSERT INTO llm_calls(turn_id, agent, model, latency_ms, tokens_in,
                                     tokens_out, status, created_at)
               VALUES(?,?,?,?,?,?,?,?)""",
            (turn_id, agent, model, latency_ms, tokens_in, tokens_out, status, now()),
        )


def first_question(sid: str) -> str:
    with connect() as con:
        row = con.execute(
            "SELECT question FROM turns WHERE session_id=? ORDER BY turn_index LIMIT 1", (sid,)
        ).fetchone()
    return row[0] if row else ""
