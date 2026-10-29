import sys
import argparse
import time
from pathlib import Path

import requests


_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent if _HERE.name == "src" else _HERE.parents[1]

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.crawler import clean_dataset, official_crawler
from src.crawler.http import create_session
from src import chunking
from src.quality_gate import run_quality_gate
from src.retrieval.legal_graph import build_graph_artifact


RAW_DIR = _ROOT / "data" / "raw"
PROCESSED_DIR = _ROOT / "data" / "processed"


def crawl_documents(document_ids: list[str] | None = None):
    session = create_session(official_crawler.HEADERS)

    print("[*] Warm-up TVPL source...")
    try:
        session.get("https://thuvienphapluat.vn/", timeout=10)
        print("[OK] Cookies OK")
    except Exception as e:
        print(f"[!] Warm-up that bai: {e}")

    time.sleep(2)

    documents = official_crawler.OFFICIAL_DOCUMENTS
    if document_ids:
        unknown = set(document_ids) - {doc["id"] for doc in documents}
        if unknown:
            raise ValueError(f"Unknown document IDs: {sorted(unknown)}")
        documents = [doc for doc in documents if doc["id"] in document_ids]
    print(f"[*] Crawling {len(documents)} legal documents from the source registry...\n" + "=" * 50)

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
        raise FileNotFoundError(f"No raw JSON files in {RAW_DIR}; run crawling first")

    for raw_file in raw_files:
        out_file = PROCESSED_DIR / raw_file.name
        clean_dataset.clean_file(raw_file, out_file)

    print("\n[OK] Xong!")


def build_chunking_outputs():
    chunking.build_all(PROCESSED_DIR, _ROOT / "data" / "structured", _ROOT / "data" / "chunks")
    run_quality_gate(PROCESSED_DIR, _ROOT / "data" / "structured", _ROOT / "data" / "chunks")
    report = build_graph_artifact(_ROOT / "data/structured", _ROOT / "data/chunks", _ROOT / "data/graph/legal_graph.json")
    print(f"[GRAPH] {report['nodes']} nodes, {report['edges']} edges")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Crawl and prepare legal sources without embedding")
    parser.add_argument("--doc-id", action="append", help="Crawl only this registry ID; repeat for multiple sources")
    parser.add_argument("--crawl-only", action="store_true", help="Download raw sources without cleaning/chunking")
    args = parser.parse_args()
    crawl_documents(args.doc_id)
    if args.crawl_only:
        return
    clean_documents()
    build_chunking_outputs()


if __name__ == "__main__":
    main()
