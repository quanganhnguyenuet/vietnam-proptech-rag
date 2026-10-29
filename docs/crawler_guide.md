# Thu thập và chuẩn bị văn bản luật

Danh mục nguồn duy nhất: [config/legal_sources.json](../config/legal_sources.json). Xem [README](../readme.md) để cài thư viện và chọn embedding.

```powershell
# Tất cả nguồn → clean → legal units → chunks → quality gate
.\.venv\Scripts\python main.py --prepare-only

# Chỉ crawl một văn bản (có thể lặp --doc-id)
.\.venv\Scripts\python -m src.pipeline_prepare --crawl-only --doc-id 95_2024_ND_CP

# Sau khi đã có raw JSON, xử lý lại offline
.\.venv\Scripts\python main.py --skip-crawl --prepare-only
```

Mỗi nguồn cần `id` an toàn cho tên file, `title`, `doc_number`, `doc_type`, `url` HTTPS. `fallback_urls` được thử theo thứ tự nếu nguồn chính lỗi, là PDF scan hoặc thiếu số điều. `expected_article_count` chỉ đếm điều trong phần chính, loại phụ lục và điều được trích dẫn trong dấu ngoặc kép. `aliases` bổ sung tên gọi cho bộ xử lý câu hỏi. Thêm nguồn rồi khởi động lại API để nạp lại danh mục.

Nguồn chính thức ưu tiên DOCX, sau đó PDF có lớp text, rồi HTML. URL tải dạng `?format=pdf` và CDN dạng `?file_name=...docx` đều được nhận diện. Binary `.doc` chưa hỗ trợ trích xuất; dùng PDF có text hoặc HTML tương ứng. TVPL chỉ lấy nội dung văn bản, không tải file có thể thuộc mục văn bản liên quan. Crawler kiểm tra số hiệu trong phần đầu nội dung và số điều nếu đã có cấu hình trước khi ghi raw JSON.

HTTP dùng retry cho lỗi tạm thời và kho CA hệ thống qua `truststore`, giữ xác thực TLS. Không tắt kiểm tra chứng chỉ. Một số nguồn trả HTTP 200 nhưng nội dung không phải toàn văn; crawler từ chối nguồn đó. Nếu không có nguồn hợp lệ, pipeline dừng trước bước clean. HTML dùng chẩn đoán được lưu ở `data/raw/official_attachments/*source.html`; không nhập những file này trực tiếp vào corpus.

Đầu ra:

| Thư mục | Nội dung |
|---|---|
| `data/raw/<id>.json` | Metadata nguồn và `raw_html` hoặc `raw_text` |
| `data/raw/official_attachments/` | File đã tải và HTML chẩn đoán |
| `data/processed/<id>.json` | `clean_text`, `main_text`, `appendices` |
| `data/structured/<id>_legal_units.json` | Đơn vị điều/khoản/phụ lục |
| `data/chunks/<id>_chunks.json` | `chunk_id`, `text`, `embedding_text`, `metadata` |

Quality gate kiểm tra ID, nội dung, citation, độ dài và độ bao phủ giữa các bước. Việc crawl thành công không đồng nghĩa văn bản vẫn còn nguyên hiệu lực. Các quan hệ sửa đổi và thời điểm áp dụng chưa được tự suy ra. PDF scan cần OCR hoặc nguồn dự phòng đã kiểm tra; không lấy trang thuộc tính làm nội dung luật.

Khi chỉ crawl một phần, `--skip-crawl` xử lý tất cả raw JSON đang có. Trước ingest có prune, đảm bảo thư mục chunks chứa toàn bộ corpus dự định lưu trong collection. Chương trình không tự xóa raw/processed/chunk của tài liệu đã bỏ khỏi registry; hãy kiểm tra và di chuyển chúng khi thay đổi phạm vi corpus.
