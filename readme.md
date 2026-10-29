# Vietnam Legal & PropTech RAG

Pipeline tra cứu văn bản đất đai, nhà ở và kinh doanh bất động sản Việt Nam:

`Nguồn luật → crawl → clean → chunk → quality gate → embedding + graph → BM25/vector → graph search → BGE reranking → LLM → câu trả lời có trích dẫn`

Đã có CLI và FastAPI để truy xuất bằng chứng và sinh câu trả lời bằng model OpenAI. Model mặc định `gpt-4.1-mini`, thay được qua `LLM_MODEL` trong `.env` hoặc `config/llm.yaml`. Không tự hợp nhất nội dung sửa đổi hoặc xác định hiệu lực pháp luật theo thời điểm.

## Cài đặt (PowerShell, Python 3.11–3.13)

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-dev.txt -r requirements-rerank.txt
Copy-Item .env.example .env
```

Điền `OPENAI_API_KEY` vào `.env` để thử OpenAI. Không commit khóa API. `requirements.txt` đủ để chạy ứng dụng; `requirements-dev.txt` thêm công cụ kiểm thử. Nếu dùng BGE-M3 local, cài thêm `requirements-bge.txt`; có thể tải model lớn ở lần chạy đầu. Các thư viện LangChain/pandas chưa dùng đã được bỏ khỏi bộ cài bắt buộc.

## Chuẩn bị dữ liệu và thử embedding OpenAI

```powershell
# Chỉ tải, làm sạch, chunk và kiểm tra; không gọi API embedding
.\.venv\Scripts\python main.py --prepare-only

# Sau khi đã có raw data, có thể chạy lại xử lý offline
.\.venv\Scripts\python main.py --skip-crawl --prepare-only

# Embed và đồng bộ corpus đã đạt quality gate
.\.venv\Scripts\python main.py --embedding-only --config src/embedding/config.openai.yaml

# Tra cứu bằng chính cấu hình đã dùng để ingest
.\.venv\Scripts\python search.py "Điều 138 Luật Đất đai quy định gì?" --config src/embedding/config.openai.yaml --top-k 5
```

Có thể chạy toàn bộ bằng `python main.py --config src/embedding/config.openai.yaml`. Lệnh này gọi API và phát sinh chi phí theo lượng dữ liệu. Bản cấu hình OpenAI dùng `text-embedding-3-small`, 1536 chiều, lưu riêng ở `data/vectordb_openai`, collection `vn_legal_openai_small_1536`.

Muốn thử `text-embedding-3-large`, tạo bản sao YAML, đổi `model.name`, đổi `embedding.dimensions` (ví dụ 3072) và **đổi `vector_store.collection_name`**, rồi ingest lại toàn bộ. Không dùng chung collection giữa BGE, các model OpenAI, hoặc các cấu hình chiều khác nhau. Collection có fingerprint để chặn dùng sai model; vector có sẵn trong JSON cũng phải có fingerprint khớp. Collection BGE cũ chưa có fingerprint chỉ được dùng ở chế độ BGE tương thích.

Hybrid search **luôn kết hợp BM25 + dense vector**, cả khi dùng BGE có learned sparse. Hai nhánh lấy ứng viên độc lập rồi gộp bằng weighted RRF. Cấu hình tiếp theo ở [config/retrieval.yaml](config/retrieval.yaml): mở rộng graph và rerank local bằng `BAAI/bge-reranker-v2-m3`. Nhánh BM25 cache tối đa 8 bộ lọc; sau khi ingest từ tiến trình khác, khởi động lại API để làm mới cache.

```powershell
# Dựng lại graph từ corpus hiện có, không gọi OpenAI
.\.venv\Scripts\python build_graph.py

# Tải/cache BGE reranker và thử một cặp câu hỏi–đoạn luật
.\.venv\Scripts\python warmup_reranker.py --smoke-test

# Duyệt graph offline, không cần embedding API key
.\.venv\Scripts\python graph_search.py --doc-id 31_2024_QH15 --article 138
```

Hướng dẫn đầy đủ về graph, reranking và đầu vào cho agent: [retrieval_guide.md](docs/retrieval_guide.md). Không cần embed lại khi chỉ thay cấu hình reranker/graph; cần rebuild graph khi corpus thay đổi.

Với GPU NVIDIA, cần PyTorch bản CUDA để dùng được GPU. Xem lệnh cài đặt trong hướng dẫn trên. Cấu hình mặc định tự chọn FP16 trên CUDA, FP32 trên CPU, batch 1 và cửa sổ 1024 token. Sau khi tải xong model, kiểm tra bằng `python warmup_reranker.py --local-only --smoke-test`; chạy `python chat.py` để chat có rerank.

Không cắt ngầm văn bản quá giới hạn token OpenAI: chương trình báo lỗi để chia lại chunk. Client có timeout/retry; giới hạn số text và tổng token mỗi request. Xem [OpenAI embeddings guide](https://developers.openai.com/api/docs/guides/embeddings) và [API reference](https://developers.openai.com/api/reference/resources/embeddings/methods/create).

## Hỏi–đáp với API thật

Sau khi điền `OPENAI_API_KEY` vào `.env`, có thể kiểm tra quyền truy cập bằng hai request nhỏ:

```powershell
.\.venv\Scripts\python check_api.py
.\.venv\Scripts\python chat.py "Theo văn bản trong corpus, Điều 138 Luật Đất đai 2024 quy định về vấn đề gì?" --top-k 5
```

Lệnh chat dùng collection đã ingest và BGE reranker theo cấu hình. Nếu chưa tải xong reranker, test rõ ràng với `--no-rerank`; BM25, vector, graph và LLM vẫn chạy. Không có việc tự bỏ qua reranker khi lỗi.

```powershell
.\.venv\Scripts\python chat.py "Điều 138 Luật Đất đai 2024 quy định gì?" --no-rerank --include-evidence --output data/live_chat_demo.json
```

Để chat liên tục trong terminal, bỏ đối số câu hỏi; nhập `end` để kết thúc:

```powershell
.\.venv\Scripts\python chat.py --no-rerank
```

Hoặc mở cuộc trò chuyện bằng một câu hỏi có sẵn:

```powershell
.\.venv\Scripts\python chat.py "Gia đình tôi sử dụng đất từ năm 1995, không có giấy tờ." --interactive --no-rerank --output data/live_conversation.json
```

Chế độ này nhớ tối đa 6 lượt gần nhất trong giới hạn 12.000 ký tự, chỉ lưu trong bộ nhớ của phiên terminal. Mỗi lượt tiếp theo có thêm một request LLM để viết lại câu hỏi thành truy vấn độc lập trước khi chạy hybrid search, graph và rerank theo cấu hình. Model vẫn nhận câu hỏi gốc và lịch sử khi sinh câu trả lời; nguồn luật được lấy mới ở mỗi lượt. `end` không gọi API; `Ctrl+C` hoặc EOF cũng thoát. Lỗi API hoặc model không làm mất các lượt đã hoàn tất, có thể nhập lại câu hỏi. Dùng `--output` để lưu toàn bộ các lượt hoàn tất dưới dạng JSON có trường `turns`; file được cập nhật sau mỗi lượt. Lệnh có câu hỏi nhưng không có `--interactive` vẫn trả lời một lần như trước.

`config/llm.yaml` tách model sinh câu trả lời khỏi model embedding và model reranker. API sử dụng Responses API với đầu ra có cấu trúc; mỗi ý trả lời phải có nhãn nguồn trong ngữ cảnh. Máy chủ kiểm tra nhãn và ánh xạ về citation/URL có sẵn; không nhận URL hay nguồn tự tạo từ model. Kiểm tra nhãn không thay thế việc đánh giá nội dung của câu trả lời. Không có bằng chứng thì không gọi model và trả `insufficient_evidence`. Request sinh câu trả lời đặt `store=false`, có timeout và giới hạn output token.

System prompt của chatbot nằm ở [config/prompts/legal_chatbot.md](config/prompts/legal_chatbot.md), chọn qua `system_prompt_file` trong `config/llm.yaml`. Có thể sửa giọng điệu trực tiếp trong file; prompt được đọc lại mỗi lần sinh câu trả lời. CLI mặc định in câu trả lời và nguồn như chatbot; thêm `--json` khi cần xem payload đầy đủ. Với ứng dụng chat, hiển thị trường `answer`; `claims`, `evidence` và `usage` phục vụ máy chủ/debug. CLI tương tác giữ lịch sử trong phiên hiện tại; `/chat` vẫn xử lý từng request độc lập, chưa có bộ nhớ phiên cho frontend.

Nếu câu hỏi có nhiều cách hiểu, chatbot hỏi một câu để xác định ý định trước khi hỏi giấy tờ. Trường `clarification_question` chứa câu hỏi đó hoặc `null`; `missing_information` chỉ chứa tối đa một giới hạn cần đối chiếu. Schema tách câu hỏi khỏi các nhận định có nguồn, tránh trả về nhiều câu hỏi liên tiếp. Câu hỏi làm rõ đã được ghép vào `answer`, frontend không cần ghép lại. Phản hồi chỉ hỏi làm rõ vẫn có `status=insufficient_evidence` vì chưa đưa ra kết luận pháp lý; không cần thêm lời từ chối vào giao diện.

## API

`.env` mặc định chọn cấu hình OpenAI qua `EMBEDDING_CONFIG`. Khi dùng BGE, đặt `EMBEDDING_CONFIG=src/embedding/config.yaml`.

```powershell
.\.venv\Scripts\python -m uvicorn src.api.main:app --reload
Invoke-RestMethod http://127.0.0.1:8000/search -Method Post -ContentType 'application/json; charset=utf-8' -Body '{"query":"Thủ tục cấp giấy chứng nhận quyền sử dụng đất","top_k":5}'
```

Swagger: `http://127.0.0.1:8000/docs`; kiểm tra tiến trình: `GET /health`. `POST /search` trả `plan`, `results`, `related_context`, `agent_context`, `retrieval`; mỗi kết quả giữ citation, URL, ID chunk và điểm từng bước. `POST /graph/search` duyệt dẫn chiếu từ `doc_id`, `article_no`, `max_hops`; thêm `query` để rerank ngữ cảnh graph. `/health` chỉ kiểm tra tiến trình.

`POST /chat` nhận `query`, `top_k`, `include_evidence` và `use_reranker` tùy chọn. Kết quả gồm `answer`, `claims`, `citations`, model thực dùng, usage và bằng chứng truy xuất nếu yêu cầu. Khi chưa có trọng số BGE, gửi `use_reranker: false` để test API sinh câu trả lời ngay:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/chat -Method Post -ContentType 'application/json; charset=utf-8' -Body '{"query":"Điều 138 Luật Đất đai 2024 quy định gì?","top_k":5,"use_reranker":false,"include_evidence":true}'
```

## Bổ sung nguồn luật

Chỉnh [config/legal_sources.json](config/legal_sources.json). Crawler và bộ nhận diện số hiệu văn bản cùng đọc danh mục này. Mỗi mục có `id`, `title`, `doc_number`, `doc_type`, `url`; tùy chọn `expected_article_count`, `fallback_urls`, `aliases`.

Danh mục gồm 5 văn bản ban đầu và 8 nguồn bổ sung đã đối chiếu ngày 29/10/2026:

| Văn bản bổ sung | Nguồn |
|---|---|
| NĐ 71/2024: giá đất | [Cổng Chính phủ](https://vanban.chinhphu.vn/?docid=210523&pageid=27160) |
| NĐ 95/2024: hướng dẫn Luật Nhà ở | [Cổng Chính phủ](https://vanban.chinhphu.vn/?docid=210761&pageid=27160) |
| NĐ 96/2024: hướng dẫn Luật Kinh doanh BĐS | [Cổng Chính phủ](https://vanban.chinhphu.vn/?docid=210798&pageid=27160) |
| NĐ 103/2024: tiền sử dụng đất, thuê đất | [Cổng Chính phủ](https://vanban.chinhphu.vn/?docid=210797&pageid=27160) |
| Luật 43/2024: sửa đổi các luật liên quan | [Cổng Chính phủ](https://vanban.chinhphu.vn/?docid=211201&pageid=27160) |
| NĐ 151/2025: thẩm quyền đất đai hai cấp | [Cổng Chính phủ](https://vanban.chinhphu.vn/?docid=213927&pageid=27160) |
| NĐ 226/2025: sửa đổi nghị định đất đai | [Cổng Chính phủ](https://vanban.chinhphu.vn/?docid=214955&pageid=27160) |
| NĐ 54/2026: sửa đổi nghị định nhà ở/BĐS | [Cổng Chính phủ](https://vanban.chinhphu.vn/?docid=216924&pageid=27160) |

`verified_at` chỉ ghi ngày kiểm tra đường dẫn/số hiệu, không xác nhận văn bản còn nguyên hiệu lực. Danh mục chưa bao quát toàn bộ sửa đổi 2025–2026 hay văn bản địa phương. Cần bổ sung văn bản hợp nhất, quan hệ sửa đổi và hiệu lực theo điều khoản trước khi dùng cho tư vấn áp dụng luật hiện hành.

Crawler ưu tiên file đính kèm trên nguồn chính thức, chỉ nhận nội dung có điều luật; đối chiếu số điều nếu đã cấu hình. Không dùng toàn bộ trang metadata thay cho luật. PDF scan không có text cần OCR hoặc bổ sung nguồn HTML/DOCX qua `fallback_urls`; crawler dừng nếu thiếu văn bản để tránh index lại dữ liệu cũ. Không tự chọn file đính kèm trong phần văn bản liên quan của TVPL.

## Chất lượng dữ liệu và kiểm thử

Chunk giữ ngữ cảnh tên luật/số hiệu/điều/khoản trong `embedding_text`, còn `text` giữ nội dung để hiển thị. Giới hạn thân chunk là 2200 ký tự; ưu tiên đoạn/câu/ranh giới từ, có `parent_unit_id`, `part_index`, `part_count` cho đoạn bị tách. Điều/khoản trích dẫn trong dấu ngoặc kép được giữ trong điều sửa đổi; các dãy khoản không liên tục được giữ nguyên để tránh tạo trích dẫn sai. Nếu text PDF có dấu ngoặc kép lỗi, parser khôi phục phạm vi ở dấu đóng kế tiếp, bắt buộc đối chiếu số điều, và giữ nguyên cả điều với `parse_warning=unbalanced_quotes` thay vì suy ra khoản/điểm. Không dùng overlap xuyên điều/khoản.

Quality gate kiểm tra đầu ra không rỗng, số điều đã biết, ID trùng, độ dài, trích dẫn và độ bao phủ tài liệu/điều/legal unit giữa các bước. Khi ingest, mặc định xóa vector có ID không còn trong **toàn bộ** corpus chunks sau khi upsert thành công; dùng `--no-prune` để giữ chúng. Không dùng prune với tập dữ liệu chỉ là một phần của collection. Dữ liệu vector BGE đã có trong repo được giữ nguyên; `.gitignore` chặn file vector mới nhưng không tự xóa file đã được Git theo dõi.

```powershell
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python -m pip check
```

Tests mặc định không gọi OpenAI thật hoặc tải BGE. Có kiểm thử ChromaDB thật với OpenAI client giả lập. Chỉ đặt `RUN_EMBEDDING_LIVE_TESTS=1` khi muốn tải và kiểm thử BGE thật.
