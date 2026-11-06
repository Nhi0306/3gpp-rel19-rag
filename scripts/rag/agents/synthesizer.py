"""Agent 4: Synthesizer - tra loi tieng Viet kem trich dan bat buoc."""
from __future__ import annotations

from config import CFG
from rag.agents.common import format_evidence, trace, usage_dict
from rag.llm import call_text, call_text_stream
from rag.retrieval import citation_of
from rag.state import RAGState
from rag.stream import token_sink
from rel19.common import log

SYSTEM = """Bạn là kỹ sư tiêu chuẩn 3GPP, trả lời câu hỏi dựa HOÀN TOÀN vào NGỮ CẢNH được cung cấp.

Quy tắc bắt buộc:
1. Chỉ dùng thông tin trong NGỮ CẢNH. Không dùng kiến thức bên ngoài, không bịa.
2. Mỗi nhận định phải kèm trích dẫn dạng [n] ứng với số hiệu đoạn trong NGỮ CẢNH.
   Ví dụ: "SMF chọn UPF cho phiên PDU [3]."
3. Trả lời bằng tiếng Việt, giữ nguyên thuật ngữ tiếng Anh (PDU Session, SMF, UPF, NG-RAN...).
4. Cấu trúc:
   - 1-2 câu tóm tắt trực tiếp câu hỏi.
   - Phần chi tiết (gạch đầu dòng hoặc đoạn ngắn), có thể dùng bảng nếu so sánh.
   - Nếu ngữ cảnh không đủ, nói rõ "Tài liệu hiện có không nêu ..." thay vì suy đoán.
5. Cuối cùng thêm mục "Nguồn" liệt kê các trích dẫn đã dùng, mỗi dòng:
   [n] TS/TR <spec> v<version> §<section> — <title>
6. Không nhắc tới việc bạn là AI, không giải thích quy trình.
"""


def synthesizer(state: RAGState) -> dict:
    question = state.get("question", "")
    evidence = (state.get("evidence") or [])[:10]
    if not evidence:
        answer = (
            "Tài liệu Rel-19 hiện có không chứa thông tin để trả lời câu hỏi này.\n\n"
            "Bạn có thể thử nêu rõ số hiệu spec hoặc điều khoản (ví dụ: TS 23.501 §5.7)."
        )
        return {
            "answer": answer, "citations": [],
            "trace": [trace("synthesizer", evidence=0)],
        }

    context = format_evidence(evidence, max_chars=CFG["RAG_EVIDENCE_CHARS"])
    user = f"Câu hỏi: {question}\n\nNGỮ CẢNH:\n{context}\n"
    verdict = state.get("verdict") or {}
    if verdict.get("issues"):
        user += "\nLần trả lời trước bị đánh giá: " + "; ".join(verdict["issues"][:3])
        user += "\nHãy sửa lại, chỉ dùng thông tin có trong ngữ cảnh."

    if token_sink.get() is not None:
        text, usage = call_text_stream("synthesizer", SYSTEM, user, temperature=0.25)
    else:
        text, usage = call_text("synthesizer", SYSTEM, user, temperature=0.25)

    if not text:
        log.warning("synthesizer trả rỗng, thử lại bằng gemini-2.5-flash")
        text, usage = call_text(
            "synthesizer", SYSTEM, user, temperature=0.25,
            model="google/gemini-2.5-flash", retries=2,
        )
    if not text:
        text = (
            "Không sinh được câu trả lời từ mô hình ngôn ngữ (nội dung rỗng). "
            "Vui lòng thử lại; nếu lặp lại, kiểm tra `SYNTH_MODEL`/`SYNTH_MAX_TOKENS`."
        )
    citations = [citation_of(h, i) for i, h in enumerate(evidence, 1)]
    log.info("synthesizer: %d ký tự, %d nguồn", len(text), len(citations))
    return {
        "answer": text,
        "citations": citations,
        "trace": [trace("synthesizer", evidence=len(evidence), chars=len(text))],
        "usage": [usage_dict(usage)],
    }
