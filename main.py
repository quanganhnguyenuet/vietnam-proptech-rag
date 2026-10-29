"""Run the local legal RAG data pipeline end to end."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.embedding.vector_store import ingest_chunks_to_store
from src.pipeline_prepare import (
    build_chunking_outputs,
    clean_documents,
    crawl_documents,
)
from src.quality_gate import run_quality_gate
from src.retrieval.legal_graph import build_graph_artifact


ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT / "data" / "processed"
STRUCTURED_DIR = ROOT / "data" / "structured"
CHUNKS_DIR = ROOT / "data" / "chunks"
DEFAULT_CONFIG = ROOT / "src" / "embedding" / "config.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare legal documents, embed chunks, and sync ChromaDB."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--skip-crawl",
        action="store_true",
        help="Reuse local raw files, then clean, chunk, validate, and embed.",
    )
    mode.add_argument(
        "--embedding-only",
        action="store_true",
        help="Validate existing outputs and only sync chunks to ChromaDB.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Crawl, clean, chunk and validate without calling an embedding model.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Embedding and vector store YAML configuration.",
    )
    parser.add_argument(
        "--no-prune",
        action="store_true",
        help="Keep vector DB records whose IDs are absent from current chunk files.",
    )
    return parser


def prepare_data(*, skip_crawl: bool, embedding_only: bool) -> None:
    if embedding_only:
        run_quality_gate(PROCESSED_DIR, STRUCTURED_DIR, CHUNKS_DIR)
        build_graph_artifact(STRUCTURED_DIR, CHUNKS_DIR, ROOT / "data/graph/legal_graph.json")
        return

    if not skip_crawl:
        crawl_documents()
    clean_documents()
    build_chunking_outputs()


def run_pipeline(args: argparse.Namespace) -> int:
    prepare_data(
        skip_crawl=args.skip_crawl,
        embedding_only=args.embedding_only,
    )
    if getattr(args, "prepare_only", False):
        return 0
    return ingest_chunks_to_store(
        CHUNKS_DIR,
        config_path=args.config,
        project_root=ROOT,
        prune=not args.no_prune,
    )


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args()
    total = run_pipeline(args)
    if args.prepare_only:
        print("\n[OK] Data prepared and validated; no embeddings generated.")
    else:
        print(f"\n[OK] Pipeline completed. Indexed {total} chunks into ChromaDB.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
