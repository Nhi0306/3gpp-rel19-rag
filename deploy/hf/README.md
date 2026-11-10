---
title: 3GPP Rel-19 RAG
emoji: 📡
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# 3GPP Rel-19 RAG

Hỏi đáp tài liệu 3GPP Release 19 (series 21–38) bằng tiếng Việt, câu trả lời
bắt buộc kèm trích dẫn spec / version / section.

- Truy xuất hybrid: 70% dense vector + 30% BM25.
- Pipeline 5 tác tử: Supervisor → Retriever → Grader → Synthesizer → Verifier.
- Nguồn: Qdrant Cloud (`3gpp_rel19` + `3gpp_rel19_sparse`), LLM qua OpenRouter.

Mã nguồn: https://github.com/Nhi0306/3gpp-rel19-rag
