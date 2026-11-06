"""Agent 1: Supervisor - phan tich cau hoi, tach filter, viet lai truy van."""
from __future__ import annotations

from config import CFG
from rag.agents.common import schema_hint, trace, usage_dict
from rag.llm import Usage, call_json, model_name
from rag.schemas import Plan
from rag.state import RAGState
from rel19.common import log

SYSTEM = """Bạn là Supervisor của hệ thống tra cứu tài liệu 3GPP Release 19.

Nhiệm vụ: phân tích câu hỏi và trả về DUY NHẤT một JSON theo schema:
{schema}

Quy tắc:
- specs: số hiệu spec nếu người dùng nhắc tới (vd "23.501", "38.304", "33.501"). Để rỗng nếu không rõ.
- section: số điều khoản nếu được nhắc (vd "5.7.1", "7.4"). Để rỗng nếu không có.
- sub_queries: 1-3 truy vấn tiếng Anh độc lập, giàu thuật ngữ 3GPP, dùng để tìm kiếm.
  Nếu câu hỏi so sánh nhiều khía cạnh, tách thành nhiều truy vấn.
- Nếu câu hỏi nối tiếp ngữ cảnh hội thoại trước (đại từ "nó", "còn ... thì sao"),
  hãy viết lại thành câu đầy đủ dựa trên lịch sử.
- language: "vi" nếu người dùng hỏi tiếng Việt, ngược lại "en".
- intent: lookup | compare | procedure | definition | other.
"""


def supervisor(state: RAGState) -> dict:
    question = state.get("question", "").strip()
    history = state.get("history_text") or "(không có)"
    refine = state.get("refine_hint") or ""
    iterations = int(state.get("iterations", 0))
    if refine:
        iterations += 1

    user = f"Lịch sử hội thoại gần đây:\n{history}\n\nCâu hỏi hiện tại: {question}"
    if refine:
        user += f"\n\nLần tìm trước thiếu thông tin: {refine}\nHãy viết lại truy vấn để bù phần thiếu."

    try:
        plan, usage = call_json(
            "supervisor", SYSTEM.format(schema=schema_hint(Plan)), user, Plan, temperature=0.1,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("supervisor fallback: %s", exc)
        plan = Plan(sub_queries=[question])
        usage = Usage(agent="supervisor", model=model_name("supervisor"))

    subs = [q.strip() for q in plan.sub_queries if q.strip()][: CFG["RAG_MAX_SUBQUERIES"]]
    if not subs:
        subs = [question]
    specs = [s.strip() for s in plan.specs if s.strip()][:3]

    log.info("supervisor: intent=%s specs=%s subs=%s", plan.intent, specs, subs)
    return {
        "language": plan.language or "vi",
        "intent": plan.intent or "lookup",
        "specs": specs,
        "section": plan.section or "",
        "sub_queries": subs,
        "refine_hint": "",
        "iterations": iterations,
        "trace": [trace("supervisor", intent=plan.intent, specs=specs,
                        section=plan.section, sub_queries=subs)],
        "usage": [usage_dict(usage)],
    }
