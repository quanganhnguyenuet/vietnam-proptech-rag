import sys
import time
from pathlib import Path

import requests


_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent if _HERE.name == "src" else _HERE.parents[1]

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.crawler import clean_dataset, official_crawler
from src import chunking
from src.quality_gate import run_quality_gate


RAW_DIR = _ROOT / "data" / "raw"
PROCESSED_DIR = _ROOT / "data" / "processed"


def crawl_documents():
    session = requests.Session()
    session.headers.update(official_crawler.HEADERS)

    print("[*] Warm-up TVPL source...")
    try:
        session.get("https://thuvienphapluat.vn/", timeout=10)
        print("[OK] Cookies OK")
    except Exception as e:
        print(f"[!] Warm-up that bai: {e}")

    time.sleep(2)

    documents = official_crawler.OFFICIAL_DOCUMENTS
    print(f"[*] Bat dau crawl {len(documents)} van ban tu TVPL...\n" + "=" * 50)

    failed_documents = []
    try:
        for idx, doc in enumerate(documents, 1):
            print(f"\n[Progress {idx}/{len(documents)}]")
            try:
                success = official_crawler.crawl_official_document(doc, session)
            except Exception as exc:
                print(f"[!] Crawl failed for {doc['id']}: {exc}")
                success = False

            if not success:
                failed_documents.append(doc["id"])
            time.sleep(3)
    finally:
        session.close()

    if failed_documents:
        failed = ", ".join(failed_documents)
        raise RuntimeError(
            f"Crawl failed for {len(failed_documents)} document(s): {failed}. "
            "Stopped before cleaning to avoid indexing stale raw files."
        )


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


def build_chunking_outputs():
    chunking.build_all(PROCESSED_DIR, _ROOT / "data" / "structured", _ROOT / "data" / "chunks")
    run_quality_gate(PROCESSED_DIR, _ROOT / "data" / "structured", _ROOT / "data" / "chunks")


def main():
    crawl_documents()
    clean_documents()
    build_chunking_outputs()


if __name__ == "__main__":
    main()
