"""Search an ingested corpus using the same configuration as ingestion."""
import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv

from src.embedding.vector_store import ChromaVectorStore
from src.retrieval.retriever import LegalRetriever
from src.retrieval.config import RetrievalConfig, DEFAULT_RETRIEVAL_CONFIG

ROOT = Path(__file__).resolve().parent


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--config", type=Path, default=Path(os.getenv("EMBEDDING_CONFIG", "src/embedding/config.yaml")))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retrieval-config", type=Path, default=Path(os.getenv("RETRIEVAL_CONFIG", str(DEFAULT_RETRIEVAL_CONFIG))))
    parser.add_argument("--no-rerank", action="store_true", help="Explicitly disable cross-encoder reranking for comparison")
    parser.add_argument("--no-graph", action="store_true", help="Explicitly disable graph expansion for comparison")
    args = parser.parse_args()
    with ChromaVectorStore.from_config(args.config, ROOT) as store:
        config = RetrievalConfig.from_yaml(args.retrieval_config)
        if args.no_rerank:
            config = replace(config, reranker=replace(config.reranker, enabled=False))
        if args.no_graph:
            config = replace(config, graph=replace(config.graph, enabled=False))
        print(json.dumps(LegalRetriever(store, config, project_root=ROOT).retrieve(args.query, args.top_k), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
