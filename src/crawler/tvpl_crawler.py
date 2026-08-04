import json
import os
import time
import requests
from bs4 import BeautifulSoup
from pathlib import Path


_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]

SEED_DOCUMENTS = [
    {
        "id": "31_2024_QH15",
        "title": "Luật Đất đai 2024",
        "doc_number": "31/2024/QH15",
        "doc_type": "luat",
        "url": "https://thuvienphapluat.vn/van-ban/Bat-dong-san/Luat-Dat-dai-2024-31-2024-QH15-523642.aspx"
    },
    {
        "id": "27_2023_QH15",
        "title": "Luật Nhà ở 2023",
        "doc_number": "27/2023/QH15",
        "doc_type": "luat",
        "url": "https://thuvienphapluat.vn/van-ban/Bat-dong-san/Luat-Nha-o-27-2023-QH15-528669.aspx"
    },
    {
        "id": "29_2023_QH15",
        "title": "Luật Kinh doanh Bất động sản 2023",
        "doc_number": "29/2023/QH15",
        "doc_type": "luat",
        "url": "https://thuvienphapluat.vn/van-ban/Bat-dong-san/Luat-Kinh-doanh-bat-dong-san-29-2023-QH15-530116.aspx"
    },
    {
        "id": "101_2024_ND_CP",
        "title": "Nghị định 101/2024/NĐ-CP Điều tra cơ bản đất đai",
        "doc_number": "101/2024/NĐ-CP",
        "doc_type": "nghi_dinh",
        "url": "https://thuvienphapluat.vn/van-ban/Bat-dong-san/Nghi-dinh-101-2024-ND-CP-dang-ky-cap-giay-chung-nhan-quyen-su-dung-dat-tai-san-gan-lien-dat-613131.aspx"
    },
    {
        "id": "102_2024_ND_CP",
        "title": "Nghị định 102/2024/NĐ-CP Hướng dẫn Luật Đất đai",
        "doc_number": "102/2024/NĐ-CP",
        "doc_type": "nghi_dinh",
        "url": "https://thuvienphapluat.vn/van-ban/Bat-dong-san/Nghi-dinh-102-2024-ND-CP-huong-dan-Luat-Dat-dai-603982.aspx"
    }
]

RAW_DATA_DIR = _ROOT / "data" / "raw"

# Headers giả lập browser thật — thuvienphapluat trả 403 nếu thiếu Accept, Referer
# Không set Accept-Encoding: để requests tự quản lý (gzip/deflate) và tự decompress
# Nếu set thủ công 'br', server trả brotli nhưng requests không decode được → binary garbage
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Referer": "https://thuvienphapluat.vn/",
}


def crawl_thuvienphapluat(doc_info: dict, session: requests.Session):
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    print(f"[*] Fetching: {doc_info['title']} ({doc_info['doc_number']})...")

    try:
        response = session.get(doc_info["url"], timeout=20)
        response.encoding = "utf-8"

        print(f"    Status: {response.status_code}")

        # Nếu lỗi HTTP thì chỉ skip văn bản này, không ngắt chương trình
        if response.status_code != 200:
            print(f"[✘] Lỗi HTTP {response.status_code} khi tải {doc_info['id']}")
            return False

        soup = BeautifulSoup(response.text, "html.parser")

        # Lấy phần nội dung văn bản (div.content1)
        content_div = soup.find("div", class_="content1")
        if not content_div:
            print(f"[!] Warning: Không tìm thấy div.content1 trong {doc_info['id']}, lấy toàn bộ body.")
            content_div = soup.find("body")

        raw_html = str(content_div) if content_div else response.text

        data = {
            "doc_id": doc_info["id"],
            "title": doc_info["title"],
            "doc_number": doc_info["doc_number"],
            "doc_type": doc_info["doc_type"],
            "source_url": doc_info["url"],
            "crawled_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "raw_html": raw_html,
        }

        output_file = RAW_DATA_DIR / f"{doc_info['id']}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"[✔] Đã lưu: {output_file}")
        return True

    except Exception as e:
        print(f"[✘] Error khi cào {doc_info['id']}: {e}")
        return False

