import sys
import time
from pathlib import Path

import requests


_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.crawler import clean_dataset, tvpl_crawler


RAW_DIR = _ROOT / "data" / "raw"
PROCESSED_DIR = _ROOT / "data" / "processed"


def crawl_documents():
    session = requests.Session()
    session.headers.update(tvpl_crawler.HEADERS)

    print("[*] Warm-up: lay cookies tu trang chu...")
    try:
        session.get("https://thuvienphapluat.vn/", timeout=10)
        print("[OK] Cookies OK")
    except Exception as e:
        print(f"[!] Warm-up that bai: {e}")

    time.sleep(2)

    documents = tvpl_crawler.SEED_DOCUMENTS
    print(f"[*] Bat dau crawl {len(documents)} van ban...\n" + "=" * 50)

    for idx, doc in enumerate(documents, 1):
        print(f"\n[Progress {idx}/{len(documents)}]")
        tvpl_crawler.crawl_thuvienphapluat(doc, session)
        time.sleep(3)


def clean_documents():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw_files = list(RAW_DIR.glob("*.json"))
    if not raw_files:
        print(f"[!] Khong co file nao trong {RAW_DIR}")
        return

    for raw_file in raw_files:
        out_file = PROCESSED_DIR / raw_file.name
        clean_dataset.clean_file(raw_file, out_file)

    print("\n[OK] Xong!")


def main():
    crawl_documents()
    clean_documents()


if __name__ == "__main__":
    main()
