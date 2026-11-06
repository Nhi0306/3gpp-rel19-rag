"""Agent 5: Verifier - kiem tra moi claim co nguon trong evidence."""
from __future__ import annotations

import re

from rag.agents.common import schema_hint, trace, usage_dict
from rag.llm import Usage, call_json, model_name
from rag.schemas import Verdict
from rag.state import RAGState
from rel19.common import log

SYSTEM = """Bạn là Verifier của hệ thống RAG 3GPP.

Cho CÂU TRẢ LỜI và danh sách NGUỒN (mỗi nguồn có số hiệu [n]).
Hãy kiểm tra:
- Mọi trích dẫn [n] trong câu trả lời có tồn tại trong danh sách nguồn không.
- Các nhận định kỹ thuật quan trọng có được nguồn đó hỗ trợ không.

Trả về DUY NHẤT JSON theo schema:
{schema}

Quy tắc:
- ok: true nếu không có lỗi nghiêm trọng (thiếu nguồn cho ý quan trọng hoặc trích dẫn sai).
- issues: mô tả ngắn các vấn đề (tiếng Việt).
- unsupported: số hiệu [n] bị trích dẫn nhưng không có trong nguồn hoặc không hỗ trợ nội dung.
"""


def verifier(state: RAGState) -> dict:
    answer = state.get("answer", "")
    citations = state.get("citations") or []
    attempts = int(state.get("verify_attempts", 0))
    if not citations:
        return {
            "verdict": {"ok": False, "issues": ["Không có nguồn nào được trích dẫn."],
                        "unsupported": []},
            "trace": [trace("verifier", ok=False, reason="no-citations")],
        }

    claimed = sorted({int(x) for x in re.findall(r"\[(\d+)\]", answer)})
    valid = {c["rank"] for c in citations}
    bogus = [c for c in claimed if c not in valid]

    source_list = "\n".join(
        f"[{c['rank']}] {c['spec']} v{c['version']} §{c['section']} — {c['title']}\n"
        f"    {c['snippet'][:220]}"
        for c in citations
    )
    user = f"CÂU TRẢ LỜI:\n{answer}\n\nNGUỒN:\n{source_list}"

    try:
        verdict, usage = call_json(
            "verifier", SYSTEM.format(schema=schema_hint(Verdict)), user, Verdict, temperature=0.0,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("verifier fallback: %s", exc)
        verdict = Verdict(ok=not bogus, issues=["Không kiểm chứng được (LLM lỗi)."] if bogus else [])
        usage = Usage(agent="verifier", model=model_name("verifier"))

    if bogus:
        verdict.ok = False
        verdict.unsupported = sorted(set(verdict.unsupported) | set(bogus))
        verdict.issues.append(f"Trích dẫn không tồn tại: {bogus}")

    if not verdict.ok and attempts == 0:
        return {
            "verdict": verdict.model_dump(),
            "verify_attempts": 1,
            "trace": [trace("verifier", ok=False, issues=verdict.issues, retry=True)],
            "usage": [usage_dict(usage)],
        }

    if not verdict.ok:
        answer += (
            "\n\n> ⚠️ Một số ý trong câu trả lời chưa được kiểm chứng đầy đủ từ tài liệu nguồn."
        )

    log.info("verifier: ok=%s (%d issue)", verdict.ok, len(verdict.issues))
    return {
        "answer": answer,
        "verdict": verdict.model_dump(),
        "verify_attempts": attempts + 1,
        "trace": [trace("verifier", ok=verdict.ok, issues=verdict.issues)],
        "usage": [usage_dict(usage)],
    }


def after_verify(state: RAGState) -> str:
    verdict = state.get("verdict") or {}
    if not verdict.get("ok") and int(state.get("verify_attempts", 0)) == 1:
        return "synthesizer"
    return "persist"
