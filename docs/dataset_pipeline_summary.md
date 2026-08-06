# Tóm Tắt Thay Đổi Dataset Pipeline

## Mục Tiêu

Chuẩn hóa pipeline chuẩn bị dataset cho bot RAG pháp lý bất động sản theo hướng nghiêm túc hơn:

- Corpus chỉ chứa nội dung văn bản và metadata nguồn tối thiểu.
- Không lưu các metadata hiệu lực hoặc quan hệ pháp lý chưa được xác thực.
- Tách rõ nội dung chính và phụ lục/mẫu biểu.
- Sinh ID ổn định cho legal units và chunks.
- Có quality gate bắt buộc trước khi dùng dataset cho RAG.

## Những Gì Đã Làm

### 1. Crawler nguồn dữ liệu

Đã thêm `src/crawler/official_crawler.py`.

Hiện pipeline ưu tiên dùng URL từ `tvpl_crawler.SEED_DOCUMENTS` vì các URL `vbpl.vn` thử trước đó đang trả trang rỗng. Crawler mới vẫn giữ khả năng tải attachment `.docx`, `.pdf`, `.doc` nếu trang nguồn có cung cấp.

Thứ tự ưu tiên nguồn nội dung:

1. Attachment `.docx` nếu có.
2. Attachment `.pdf` nếu có và trích text được.
3. HTML TVPL, ưu tiên `div.content1`.

PDF không được coi là mặc định tốt hơn HTML. PDF chỉ là fallback khi nguồn chính thức có file và text trích xuất được.

### 2. Bỏ metadata chưa xác thực

Đã bỏ khỏi corpus các field:

- `effective_date`
- `status`
- `amended_by`
- `replaces`
- `related_docs`

Lý do: các field này có ý nghĩa pháp lý quan trọng nhưng hiện chưa có nguồn xác thực riêng. Nếu tự suy từ HTML nhiễu hoặc text mojibake thì dễ làm bot trả lời sai về hiệu lực văn bản.

Các field này sẽ được xây dựng sau trong registry/knowledge graph riêng.

### 3. Tách main text và phụ lục

Đã cập nhật `src/crawler/clean_dataset.py` để tạo:

- `clean_text`: toàn văn đã làm sạch.
- `main_text`: phần nội dung chính.
- `appendices`: danh sách phụ lục/mẫu biểu.

Mục tiêu là không để các dòng kiểu `Điều 1`, `Điều 2` trong phụ lục, hợp đồng mẫu, đơn mẫu làm nhiễu parser điều luật chính.

Kết quả sau khi clean lại dataset hiện tại:

- `101_2024_ND_CP`: 68 điều, 49 phụ lục.
- `102_2024_ND_CP`: 113 điều, 63 phụ lục.
- `27_2023_QH15`: 198 điều.
- `29_2023_QH15`: 83 điều.
- `31_2024_QH15`: 260 điều.

### 4. Chuẩn hóa ID

Đã cập nhật `src/chunking.py` để sinh ID theo cấu trúc mới:

```text
doc_id__section_type__article_{article_no}__clause_{clause_no}
```

Ví dụ:

```text
31_2024_QH15__main__article_2__clause_1
```

Với phụ lục:

```text
doc_id__appendix__appendix_1
```

Điểm `đ)` được encode thành `dd` để không bị trùng với `d)`.

### 5. Metadata tối thiểu trong chunks

Legal units và chunks chỉ giữ các metadata nguồn tối thiểu:

- `doc_id`
- `doc_title`
- `doc_number`
- `doc_type`
- `source_url`
- `source_provider`
- `source_policy`
- `source_format`
- `downloaded_files`
- `expected_article_count`
- `section_type`
- citation/article/clause metadata cần cho retrieval.

Không đưa metadata hiệu lực chưa xác thực vào embedding text.

### 6. Quality gate

Đã thêm `src/quality_gate.py`.

Quality gate kiểm tra:

- JSONL parse được.
- Không duplicate `unit_id`.
- Không duplicate `chunk_id`.
- `main_text` không rỗng.
- Số điều phải khớp kỳ vọng với các văn bản seed.
- Chunk không rỗng.
- Chunk có `embedding_text`.
- Chunk có `metadata.unit_id` và `metadata.citation`.
- Không gom nhiều điều vào cùng một unit/chunk.
- Không cho các field chưa xác thực lẻn vào corpus: `effective_date`, `status`, `amended_by`, `replaces`, `related_docs`.

Quality gate hiện đã `PASSED` sau khi clean và rebuild lại dataset.

### 7. Tests

Đã cập nhật `tests/test_chunking.py` và thêm `tests/__init__.py`.

Test hiện kiểm tra:

- Phụ lục không làm nhiễu article chính.
- Chunk ID có `section_type`.
- Metadata nguồn tối thiểu được giữ.
- `d)` và `đ)` không bị trùng ID.
- Quality gate bắt article count mismatch.
- Quality gate từ chối metadata chưa xác thực.

Lệnh đã chạy thành công:

```powershell
D:\Conda\envs\env\python.exe -m unittest tests/test_chunking.py
D:\Conda\envs\env\python.exe -m src.quality_gate
```

## Quyết Định Thiết Kế Quan Trọng

Corpus RAG và knowledge graph phải tách vai trò:

- Corpus RAG: phục vụ tìm kiếm/truy xuất nội dung văn bản.
- Registry/Knowledge graph: quản lý hiệu lực, sửa đổi, thay thế, bãi bỏ, hướng dẫn, viện dẫn.

Không nên để crawler tự suy đoán quan hệ pháp lý rồi ghi thẳng vào corpus.

## Việc Nên Làm Tiếp Theo

1. Tạo `data/registry/documents.jsonl` cho metadata văn bản đã xác thực.
2. Tạo `data/registry/relations.jsonl` cho quan hệ pháp lý đã xác thực.
3. Thêm quality gate riêng cho registry.
4. Bổ sung các văn bản quan trọng: Nghị định 71, 88, 94, 95, 96, 103, 104/2024 và các văn bản sửa đổi mới.
5. Xây retrieval layer biết kết hợp corpus với registry/knowledge graph khi trả lời.
