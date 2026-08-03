import re
import json
from pathlib import Path
from bs4 import BeautifulSoup


# Đường dẫn tuyệt đối tính từ vị trí file này (src/crawler/)
_HERE    = Path(__file__).resolve().parent          # src/crawler/
_ROOT    = _HERE.parents[1]                         # project root

RAW_DIR      = _ROOT / "data" / "raw"
PROCESSED_DIR = _ROOT / "data" / "processed"


def clean_html(raw_html: str) -> str:
    """Chuyển HTML → text thuần sạch theo cấu trúc Markdown."""
    soup = BeautifulSoup(raw_html, "lxml")

    # Lấy text từng đoạn <p> — giữ nguyên cấu trúc điều/khoản
    paragraphs = []
    for tag in soup.find_all("p"):
        text = tag.get_text(separator=" ")
        text = _normalize(text)
        if text:
            paragraphs.append(text)

    text = "\n".join(paragraphs)

    # Chương → # Chương I ...
    text = re.sub(r"^(Chương\s+[0-9IVXLCDM]+.*)", r"\n# \1\n", text, flags=re.MULTILINE | re.IGNORECASE)
    # Mục → ## Mục 1 ...
    text = re.sub(r"^(Mục\s+\d+.*)", r"\n## \1\n", text, flags=re.MULTILINE | re.IGNORECASE)
    # Điều → ## Điều X. ...
    text = re.sub(r"^(Điều\s+\d+\.\s.*)", r"\n## \1\n", text, flags=re.MULTILINE | re.IGNORECASE)
    # Điểm a), b) → bullet
    text = re.sub(r"^([a-z]\)\s+.*)$", r"- \1", text, flags=re.MULTILINE)
    # Dọn dòng trống thừa
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _normalize(text: str) -> str:
    """Làm sạch một đoạn text."""
    text = re.sub(r"\r\n\s*", " ", text)   # soft line-break từ Word
    text = re.sub(r"\n\s*",   " ", text)
    text = text.replace("\xa0", " ")        # &nbsp;
    text = text.replace("\u200b", "")       # zero-width space
    text = re.sub(r" {2,}", " ", text)      # nhiều dấu cách → 1
    text = re.sub(r"^(\d+)\s+\.", r"\1.", text)
    text = re.sub(r"^([a-zđ])\s+\)", r"\1)", text, flags=re.IGNORECASE)
    text = re.sub(r"\bĐi\s+ều\b", "Điều", text)
    text = re.sub(r"\bCh\s+ương\b", "Chương", text)
    text = re.sub(r"\bM\s+ục\b", "Mục", text)
    return text.strip()


def clean_file(input_path: Path, output_path: Path):
    """Đọc file raw JSON, clean HTML, ghi ra output_path (JSON + .txt cùng thư mục)."""
    print(f"[*] Đang xử lý: {input_path.name}")

    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    raw_html = data.get("raw_html", "")
    if not raw_html:
        print("  [!] Không tìm thấy raw_html")
        return

    clean_text = clean_html(raw_html)
    num_dieu   = len(re.findall(r"Điều \d+\.", clean_text))
    print(f"  → {len(clean_text):,} ký tự | ~{num_dieu} điều")

    # Ghi JSON (bỏ raw_html, thêm clean_text)
    output_data = {k: v for k, v in data.items() if k != "raw_html"}
    output_data["clean_text"] = clean_text

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    # Ghi thêm .txt cùng thư mục để dễ đọc
    txt_path = output_path.with_suffix(".txt")
    txt_path.write_text(clean_text, encoding="utf-8")

    print(f"  [✔] {output_path}")
    print(f"  [✔] {txt_path}")
