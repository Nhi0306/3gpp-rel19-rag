"""Dung ngu canh hoi thoai tu cac turn gan nhat."""
from __future__ import annotations

import re

_SENT = re.compile(r"(?<=[.!?])\s+")


def summarize_answer(text: str, max_sentences: int = 2, max_chars: int = 500) -> str:
    """Lay 2 cau dau (bo phan danh sach nguon o cuoi) lam tom tat cho ngu canh."""
    if not text:
        return ""
    body = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            if body:
                break
            continue
        if s.startswith("[") or s.lower().startswith(("nguồn", "nguon", "**nguồn")):
            break
        body.append(s)
    text_body = " ".join(body).strip()
    parts = _SENT.split(text_body)
    out = " ".join(parts[:max_sentences]).strip()
    if len(out) > max_chars:
        out = out[:max_chars].rstrip() + " ..."
    return out


def build_history(turns: list[dict], max_chars: int = 700) -> str:
    blocks: list[str] = []
    for t in turns:
        answer = (t.get("answer_summary") or t.get("answer") or "").strip()
        if not answer:
            continue
        if len(answer) > max_chars:
            answer = answer[:max_chars].rstrip() + " ..."
        blocks.append(f"Hỏi: {t['question'].strip()}\nĐáp: {answer}")
    return "\n\n".join(blocks)
