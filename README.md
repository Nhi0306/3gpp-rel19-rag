# 3GPP Rel-19 — Hệ thống RAG đa tác tử

Hệ thống hỏi đáp trên toàn bộ tài liệu **3GPP Release 19** (series 21–38):
crawl – parse – chunk – embed vào Qdrant, truy xuất **hybrid (vector + BM25)**,
rồi trả lời tiếng Việt bằng một pipeline **5 tác tử (LangGraph)**. Mọi câu trả lời
bắt buộc kèm trích dẫn spec / version / section, và được kiểm chứng lại trước khi trả về.

**Nguyên tắc thiết kế**

- Chỉ dùng nội dung tài liệu đã nạp — không dùng kiến thức ngoài ngữ cảnh.
- Mỗi nhận định kỹ thuật phải có nguồn `[n]`; thiếu căn cứ thì nói rõ là thiếu.
- Trả lời tiếng Việt, giữ nguyên thuật ngữ tiếng Anh (PDU Session, SMF, UPF, NG-RAN…).
- Hội thoại được lưu vào SQL; 5 lượt gần nhất được đưa vào ngữ cảnh cho câu hỏi nối tiếp.

---

## 1. Kiến trúc

```
Người dùng
  ├─ CLI  (stream token ra terminal)
  └─ Web UI  ── FastAPI/SSE ──┐
                              ▼
                     LangGraph (5 agent)
   Supervisor ─► Retriever (fan-out song song) ─► Grader
        ▲                                          │
        └──────── refine (tối đa 2 vòng) ◄─────────┘
                                                   ▼
                              Synthesizer ─► Verifier ─► Persist (SQLite)
                                                   │
                                    retry 1 lần ───┘
```

| Thành phần | Vai trò |
|---|---|
| **Qdrant `3gpp_rel19`** | dense vector 1536 chiều (Cosine, float16), payload đầy đủ |
| **Qdrant `3gpp_rel19_sparse`** | sparse BM25, cùng point ID, payload tối giản |
| **SQLite** | phiên, lượt hỏi–đáp, trích dẫn, log LLM |
| **OpenRouter** | LLM sinh câu trả lời và các bước phân tích/kiểm chứng |
| **LangGraph** | điều phối agent, fan-out theo sub-query |

---

## 2. Dữ liệu và chỉ mục

Toàn bộ Rel-19 được tải từ 3gpp.org: **1.687 spec** (zip → docx → parsed),
sinh ra **214.013 chunk**. Collection dense có đúng 214.013 point.

**Chiến lược chunk**

1. Dựng cây section theo heading `Heading 1..5`, giữ số điều khoản (`4.2.3.1`).
2. Section nhỏ hơn **150 token** được gộp với section kế cận cùng heading cha.
3. Section vượt **1.200 token** cắt tiếp theo paragraph, overlap **150 token**.
4. Mỗi chunk được prepend `TS/TR <spec> v<version> | <section path>` để giữ ngữ cảnh.
5. Bảng chuyển Markdown; bỏ mục lục và bảng `change history` (chiếm ~1/3 dung lượng).

**Hai collection, một point ID**

Mỗi chunk có UUID `uuid5(namespace, chunk_id)` giống nhau ở cả hai collection:
dense giữ toàn bộ payload (`spec`, `version`, `section`, `section_path`, `chunk_index`,
`title`, `text`), sparse chỉ giữ `spec` / `version` / `section_path` / `chunk_index`
để tiết kiệm dung lượng. Khi hợp nhất, payload đầy đủ được lấy từ collection dense.

> Qdrant không cho thêm sparse vector vào collection đã tạo, nên sparse buộc phải
> là collection riêng. Đây là lý do kiến trúc tách đôi thay vì một collection đa vector.

**Thống kê BM25** (`Rel-19/rag/bm25-stats.json`): N = 214.013, 522.615 term,
`avgdl = 229.3` token/chunk. Vocab và `df` được tính một lần trên toàn corpus.

---

## 3. Truy xuất hybrid

Mỗi truy vấn chạy **song song** hai nhánh rồi hợp nhất client-side:

| Nhánh | Cách chạy | Trọng số |
|---|---|---|
| Dense | `openai/text-embedding-3-small`, top 80, lọc `spec` nếu có | **0.7** |
| Sparse | BM25 top 80 trên collection sparse | **0.3** |

**BM25** (tự tính, không dùng fastembed):

```
idf(t)   = ln(1 + (N - df + 0.5) / (df + 0.5))
doc(t,d) = tf · (k1 + 1) / (tf + k1 · (1 - b + b · dl / avgdl))
query(t) = idf(t)          # k1 = 1.2, b = 0.75
```

Điểm hợp nhất: `0.7 · cosine + 0.3 · (bm25 / bm25_max)`, cộng **+0.03** nếu
`section` khớp với câu hỏi, dedupe theo point ID, lấy **top 12**. Với 6 chunk đầu,
hệ thống mở rộng thêm **±1 chunk lân cận** cùng section để giữ mạch văn bản.

Tokenizer giữ cụm kỹ thuật (`ng-ran`, `23.501`, `4.2.3`) và tách thêm các phần
(`ng`, `ran`) để tăng recall; stopword tiếng Anh bị loại.

---

## 4. Luồng tác tử

| # | Agent | Nhiệm vụ | Model (profile `balanced`) |
|---|---|---|---|
| 1 | **Supervisor** | hiểu câu hỏi, trích `spec`/`section`, viết lại 1–3 sub-query tiếng Anh, dùng lịch sử nếu là câu nối tiếp | `gemini-2.5-flash` |
| 2 | **Retriever** | fan-out `Send` theo từng sub-query, chạy dense ∥ sparse, fuse, mở rộng lân cận | — |
| 3 | **Grader** | chấm đoạn nào thực sự liên quan, quyết định đã đủ thông tin chưa | `gemini-2.5-flash` |
| 4 | **Synthesizer** | trả lời tiếng Việt, mỗi ý kèm `[n]`, có mục “Nguồn” | `gemini-2.5-pro` |
| 5 | **Verifier** | kiểm tra trích dẫn có tồn tại và có hỗ trợ nội dung; retry 1 lần | `gemini-2.5-flash` |

**Refine**: nếu Grader báo thiếu thông tin, quay lại Supervisor để viết truy vấn mới,
tối đa **2 vòng**. Khi evidence đã có ≥ 8 đoạn và ≥ 6 đoạn điểm cao (≥ 0.35),
hệ thống bỏ qua refine để tiết kiệm chi phí và độ trễ.

**Persist** ghi lượt hỏi–đáp, trích dẫn và thống kê token vào SQLite sau mỗi câu trả lời.

State dùng reducer `operator.add` cho `candidates` / `trace` / `usage`; mỗi vòng
refine đánh dấu `round`, bước merge chỉ giữ các candidate của vòng mới nhất.

---

## 5. Hội thoại và lưu trữ

SQLite tại `Rel-19/rag/rag.sqlite`:

| Bảng | Nội dung |
|---|---|
| `sessions` | id, tiêu đề, thời gian tạo/cập nhật |
| `turns` | câu hỏi, câu trả lời, tóm tắt, intent, filter, latency, token |
| `citations` | spec, version, section, title, score, snippet cho từng `[n]` |
| `llm_calls` | agent, model, token in/out, thời gian — phục vụ theo dõi chi phí |

Mỗi lượt mới đọc **5 turn gần nhất** từ DB (mỗi câu trả lời tối đa 700 ký tự) để dựng ngữ cảnh.
Tóm tắt 2 câu đầu của câu trả lời được lưu riêng (`answer_summary`) để giảm token
khi đưa vào lịch sử. Không dùng checkpointer của LangGraph — SQL là nguồn duy nhất.

---

## 6. Giao diện

**CLI**

```powershell
python scripts\rag_chat.py
python scripts\rag_chat.py --session <session_id>
```

Lệnh trong phiên: `:new`, `:sessions`, `:history`, `:use <id>`, `:trace`, `:quit`.
Token của Synthesizer được stream trực tiếp ra terminal.

**Web**

```powershell
python scripts\rag_serve.py                 # http://127.0.0.1:8000
python scripts\rag_serve.py --port 8080
```

| Endpoint | Chức năng |
|---|---|
| `GET /api/health` | trạng thái collection, profile, model |
| `POST /api/sessions` | tạo phiên |
| `GET /api/sessions` | danh sách phiên |
| `GET /api/sessions/{id}` | các lượt + trích dẫn của phiên |
| `POST /api/chat` | hỏi–đáp JSON (không stream) |
| `POST /api/chat/stream` | SSE: `session` → `token`… → `done` (kèm citations) |

Giao diện web tĩnh nằm ở `scripts/rag/web/index.html`: chọn phiên, stream câu trả lời,
hiển thị nguồn kèm trích đoạn, cảnh báo khi Verifier không xác nhận.

---

## 7. Cấu hình

Chọn profile qua `RAG_PROFILE` trong `.env`:

| Profile | Synthesizer | Chi phí ước tính / câu |
|---|---|---|
| `cheap` | `google/gemini-2.5-flash` | ~$0.01 |
| `balanced` | `google/gemini-2.5-pro` | ~$0.03–0.05 |
| `quality` | `anthropic/claude-sonnet-4.5` | ~$0.05+ |

Các biến chính:

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `RAG_DENSE_WEIGHT` / `RAG_SPARSE_WEIGHT` | 0.7 / 0.3 | trọng số hybrid |
| `RAG_DENSE_TOP` / `RAG_SPARSE_TOP` | 80 / 80 | số ứng viên mỗi nhánh |
| `RAG_TOP_K` | 12 | số chunk đưa vào ngữ cảnh |
| `RAG_NEIGHBOR_WINDOW` | 1 | mở rộng ±1 chunk |
| `RAG_MAX_REFINE` | 2 | số vòng tìm lại tối đa |
| `RAG_HISTORY_TURNS` | 5 | số lượt gần nhất |
| `SYNTH_MAX_TOKENS` | 8192 | ngân sách token cho câu trả lời |

`LLM_MAX_TOKENS` (mặc định 4096) dùng cho các agent JSON; Synthesizer dùng
`SYNTH_MAX_TOKENS` vì Gemini tiêu tốn nhiều token suy luận trước khi viết.

---

## 8. Vận hành

Phụ thuộc Python: `requests beautifulsoup4 lxml python-docx tiktoken python-dotenv
openpyxl qdrant-client langgraph langchain-openai fastapi uvicorn`.

```powershell
# pipeline dữ liệu
python scripts\3gpp_rel19_pipeline.py all       # manifest → download → extract → parse → chunk → embed
python scripts\3gpp_rel19_pipeline.py update    # kiểm tra spec mới và nạp phần thay đổi
python scripts\3gpp_rel19_pipeline.py stats

# chỉ mục sparse BM25
python scripts\rag_sparse_index.py all          # init + stats + run
python scripts\rag_sparse_index.py status
python scripts\rag_sparse_index.py verify

# chat / web
python scripts\rag_chat.py
python scripts\rag_serve.py
```

Cập nhật hàng tuần: Task Scheduler `3GPP-Rel19-WeeklyUpdate` chạy **Chủ nhật 03:00**
(`install_weekly_task.ps1`), log tại `logs\update-weekly.log`, kết quả tại
`Rel-19\index\last-update.json`.

---

## 9. Deploy (Render)

Web chạy công khai trên Render (Docker, gói Free) để chia sẻ bằng link:

**https://&lt;service-name&gt;.onrender.com**

`render.yaml` khai báo sẵn service, biến môi trường và gói `free`. Cách deploy:

1. Đăng ký Render bằng tài khoản GitHub: https://dashboard.render.com/register
2. **New → Blueprint** → chọn repo `3gpp-rel19-rag` → **Apply**.
3. Điền 3 secret khi Render hỏi: `QDRANT_URL`, `QDRANT_API_KEY`, `OPENROUTER_API_KEY`.
4. Chờ build ~5 phút, mở link `https://<service-name>.onrender.com`.

`Dockerfile` dùng `python:3.12-slim`, giải nén `data/bm25-stats.json.gz` thành
`Rel-19/rag/bm25-stats.json`, lắng nghe `$PORT` do Render cấp.

Giới hạn của gói free: 512 MB RAM, 0.1 CPU; service **ngủ sau 15 phút** không có
request (mở lại chậm ~1–2 phút); 750 giờ chạy/tháng; đĩa tạm nên lịch sử phiên
mất khi restart; quota/rate limit chỉ lưu trong bộ nhớ tiến trình.

Hugging Face Spaces vẫn dùng được (`scripts/deploy_hf.py`) nhưng từ 2026 Docker
Space trên CPU free yêu cầu gói **PRO** trả phí.

---

## 10. Kết quả hiện tại

- **1.687** spec Rel-19 → **214.013** chunk (đã bỏ bảng `change history`).
- Qdrant dense: **214.013** point; sparse BM25: **214.013** point.
- Truy vấn mẫu (TS 23.501 §5.6.1 — PDU Session): 7 bước agent, không refine,
  ~29 giây, ~4.3k token vào / ~2.4k token ra.

---

## 11. Giới hạn đã biết

- Qdrant không hỗ trợ thêm sparse vector vào collection đã tồn tại → phải tách
  `3gpp_rel19_sparse` và hợp nhất phía client.
- Gemini 2.5 tiêu tốn ~1.2–1.5k token suy luận cho mỗi lần sinh; nếu `max_tokens`
  quá thấp, câu trả lời có thể rỗng. Vì vậy Synthesizer có ngân sách riêng và
  fallback sang `gemini-2.5-flash` khi nội dung rỗng.
- Nếu sparse index chưa sẵn sàng, retrieval tự fallback về dense-only.
- Corpus chỉ gồm Rel-19; câu hỏi về release khác sẽ được trả lời là thiếu dữ liệu.
