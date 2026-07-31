# Vietnam PropTech RAG — VBPL Crawler

## 📦 Cài đặt

```bash
pip install requests beautifulsoup4 lxml jsonlines tqdm python-dotenv
```

## 🚀 Chạy crawler

### Option 1: Pipeline đầy đủ (khuyến nghị)
```bash
python run_pipeline.py
```

### Option 2: Chỉ crawl
```bash
python -m src.crawler.vbpl_crawler
```

### Option 3: Chỉ xử lý (sau khi đã crawl)
```bash
python -m src.crawler.data_processor
```

## 📁 Cấu trúc output

```
data/
├── raw/
│   └── html_523642.html          # HTML gốc đã tải về
└── processed/
    ├── land_law_chunks_YYYYMMDD_HHMMSS.jsonl   # Raw chunks (từ crawler)
    ├── land_law_chunks_YYYYMMDD_HHMMSS.json    # Cùng data, dễ đọc hơn
    ├── rag_chunks_YYYYMMDD_HHMMSS.jsonl        # Chunks đã xử lý cho RAG
    ├── metadata_YYYYMMDD_HHMMSS.json           # Thống kê
    └── text/
        └── 523642_31_2024_QH15.txt             # Plain text mỗi văn bản
```

## 📊 Schema của RAG chunk (JSONL)

```json
{
  "id": "abc123def456",
  "text": "Luật Đất đai 2024 | Chương I: Quy định chung | Điều 3. Giải thích từ ngữ\n\nĐiều 3. Giải thích từ ngữ...",
  "metadata": {
    "law_id": "523642",
    "so_hieu": "31/2024/QH15",
    "ten_van_ban": "Luật Đất đai 2024",
    "loai_van_ban": "Luat",
    "ngay_ban_hanh": "2024-01-18",
    "chuong": "1",
    "chuong_ten": "QUY ĐỊNH CHUNG",
    "dieu_so": "3",
    "dieu_ten": "Điều 3. Giải thích từ ngữ",
    "source_url": "https://thuvienphapluat.vn/..."
  }
}
```

## 📄 Văn bản được thu thập

| Số hiệu | Tên văn bản |
|---------|-------------|
| 31/2024/QH15 | Luật Đất đai 2024 |
| 101/2024/NĐ-CP | NĐ về điều tra cơ bản đất đai |
| 102/2024/NĐ-CP | NĐ chi tiết thi hành Luật Đất đai |
| 103/2024/NĐ-CP | NĐ về tiền sử dụng đất, tiền thuê đất |

## ⚠️ Lưu ý

- Crawler nghỉ **2.5 giây** giữa mỗi request để không làm quá tải server
- Một số Nghị định có thể chưa có URL chính xác, cần cập nhật trong `TARGET_DOCUMENTS`
- Nội dung Luật Đất đai 2024 (523642) đã được xác nhận hoạt động tốt
