"""Search legal reference paths without an embedding API key."""
import argparse
import json
from pathlib import Path
import sys

from src.retrieval.config import RetrievalConfig, DEFAULT_RETRIEVAL_CONFIG
from src.retrieval.retriever import LegalRetriever

ROOT = Path(__file__).resolve().parent


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc-id", required=True)
    parser.add_argument("--article", required=True)
    parser.add_argument("--max-hops", type=int, default=2)
    parser.add_argument("--query", help="Optionally rerank graph evidence with the configured local model")
    parser.add_argument("--retrieval-config", type=Path, default=DEFAULT_RETRIEVAL_CONFIG)
    args = parser.parse_args()
    retriever = LegalRetriever(None, RetrievalConfig.from_yaml(args.retrieval_config), project_root=ROOT)
    response = retriever.graph_search(args.doc_id, args.article, max_hops=args.max_hops, query=args.query)
    print(json.dumps(response, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
