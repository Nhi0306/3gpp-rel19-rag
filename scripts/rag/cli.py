"""CLI chat voi he thong RAG da tac tu."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time

from config import CFG
from rag import db
from rag.graph import graph
from rag.stream import token_sink


def _sources(citations: list[dict]) -> str:
    if not citations:
        return "(không có nguồn)"
    lines = []
    for c in citations:
        lines.append(
            f"[{c['rank']}] TS {c['spec']} v{c['version']} §{c['section']} — {c['title']}"
        )
    return "\n".join(lines)


def _print_session(sid: str) -> None:
    title = next((s["title"] for s in db.list_sessions(100) if s["id"] == sid), "")
    print(f"Phiên: {sid}  ({title})")
    print("Lệnh: :new  :sessions  :history  :use <id>  :trace  :quit\n")


def _handle_command(cmd: str, state: dict) -> bool:
    parts = cmd.split(maxsplit=1)
    name = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""
    if name in (":quit", ":q", ":exit"):
        return False
    if name == ":new":
        state["session"] = db.create_session()
        _print_session(state["session"])
    elif name == ":sessions":
        for s in db.list_sessions(20):
            print(f"  {s['id']}  {s['updated_at']}  {s['turns']:>3} lượt  {s['title']}")
    elif name == ":history":
        for t in db.session_turns(state["session"]):
            print(f"\n[{t['turn_index']}] Bạn: {t['question']}\n    Trợ lý: {(t['answer'] or '')[:200]}...")
    elif name == ":use":
        if arg:
            state["session"] = arg
            _print_session(arg)
        else:
            print("dùng: :use <session_id>")
    elif name == ":trace":
        last = state.get("last")
        if last:
            print(json.dumps({"trace": last.get("trace"), "usage": last.get("usage")},
                             ensure_ascii=False, indent=2))
        else:
            print("chưa có lượt nào")
    else:
        print(f"không rõ lệnh {name}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat RAG 3GPP Rel-19")
    parser.add_argument("--session", help="tiếp tục phiên đã có")
    parser.add_argument("--new", action="store_true", help="tạo phiên mới (mặc định)")
    parser.add_argument("--no-stream", action="store_true", help="không stream token")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, CFG["LOG_LEVEL"], logging.INFO),
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    db.init_db()
    g = graph()

    state = {"session": args.session or db.create_session(), "last": None}
    _print_session(state["session"])

    while True:
        try:
            question = input("Bạn: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.startswith(":"):
            if not _handle_command(question, state):
                break
            continue

        print("Trợ lý: ", end="", flush=True)
        printed: list[str] = []

        def sink(text: str) -> None:
            printed.append(text)
            sys.stdout.write(text)
            sys.stdout.flush()

        token = token_sink.set(None if args.no_stream else sink)
        started = time.time()
        try:
            result = g.invoke(
                {"session_id": state["session"], "question": question},
                config={"recursion_limit": 40},
            )
        except Exception as exc:  # noqa: BLE001
            print(f"\n[LỖI] {exc}")
            continue
        finally:
            token_sink.reset(token)

        if not printed:
            print(result.get("answer", ""))
        print()
        if result.get("citations"):
            print("\nNguồn:")
            print(_sources(result["citations"]))
        print(f"\n({time.time() - started:.1f}s, {len(result.get('citations') or [])} nguồn)")
        state["last"] = result
