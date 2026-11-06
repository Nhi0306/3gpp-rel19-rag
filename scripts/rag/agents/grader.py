"""Agent 3: Grader - cham do lien quan va quyet dinh co can tim lai khong."""
from __future__ import annotations

from config import CFG
from rag.agents.common import schema_hint, trace, usage_dict
from rag.llm import Usage, call_json, model_name
from rag.schemas import Grade
from rag.state import RAGState
from rel19.common import log

SYSTEM = """Bạn là Grader của hệ thống RAG 3GPP.

Cho danh sách các đoạn tài liệu (đánh số [1], [2], ...) và câu hỏi.
Hãy chấm xem đoạn nào thực sự giúp trả lời câu hỏi.

Trả về DUY NHẤT JSON theo schema:
{schema}

Quy tắc:
- relevant: các số hiệu đoạn liên quan trực tiếp (1-based). Có thể rỗng nếu không đoạn nào hữu ích.
- sufficient: true nếu các đoạn này đủ để trả lời hữu ích và có căn cứ, dù chưa bao quát mọi khía cạnh.
  Với câu hỏi rộng (định nghĩa, tổng quan), các đoạn overview/definition là đủ — đừng đòi hỏi mọi chi tiết nhỏ.
  Chỉ đặt false khi thiếu hẳn thông tin cốt lõi (ví dụ: hỏi thủ tục cụ thể mà chỉ có định nghĩa chung).
- missing: nếu chưa đủ, mô tả ngắn thông tin còn thiếu (bằng tiếng Việt).
"""


def grader(state: RAGState) -> dict:
    evidence = state.get("evidence") or []
    question = state.get("question", "")
    if not evidence:
        return {
            "sufficient": False,
            "refine_hint": "Không tìm thấy đoạn tài liệu nào liên quan.",
            "evidence": [],
            "trace": [trace("grader", kept=0, sufficient=False)],
        }

    listing = []
    for i, h in enumerate(evidence, 1):
        p = h.get("payload") or {}
        text = " ".join(str(p.get("text", "")).split())[:400]
        listing.append(f"[{i}] {p.get('spec', '')} §{p.get('section', '')} — {p.get('title', '')}\n{text}")
    user = f"Câu hỏi: {question}\n\nCác đoạn tài liệu:\n" + "\n\n".join(listing)

    try:
        grade, usage = call_json(
            "grader", SYSTEM.format(schema=schema_hint(Grade)), user, Grade, temperature=0.0,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("grader fallback: %s", exc)
        grade = Grade(relevant=list(range(1, min(8, len(evidence)) + 1)), sufficient=True)
        usage = Usage(agent="grader", model=model_name("grader"))

    keep = [evidence[i - 1] for i in grade.relevant if 1 <= i <= len(evidence)]
    if not keep:
        keep = evidence[: min(6, len(evidence))]

    log.info("grader: giữ %d/%d, sufficient=%s", len(keep), len(evidence), grade.sufficient)
    return {
        "evidence": keep,
        "sufficient": grade.sufficient,
        "refine_hint": "" if grade.sufficient else (grade.missing or "Thiếu thông tin chi tiết."),
        "trace": [trace("grader", kept=len(keep), sufficient=grade.sufficient,
                        missing=grade.missing)],
        "usage": [usage_dict(usage)],
    }


def after_grade(state: RAGState) -> str:
    if state.get("sufficient"):
        return "synthesizer"
    if int(state.get("iterations", 0)) >= CFG["RAG_MAX_REFINE"]:
        return "synthesizer"
    # đã có nhiều đoạn điểm cao -> không cần tìm lại (tiết kiệm chi phí/độ trễ)
    evidence = state.get("evidence") or []
    strong = sum(1 for h in evidence if float(h.get("score", 0)) >= CFG["RAG_MIN_SCORE"])
    if len(evidence) >= 8 and strong >= 6:
        return "synthesizer"
    return "supervisor"
