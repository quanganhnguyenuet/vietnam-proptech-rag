"""Build the legal reference graph without embedding or LLM calls."""
import argparse
import json
from pathlib import Path

from src.retrieval.legal_graph import build_graph_artifact

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structured-dir", type=Path, default=ROOT / "data/structured")
    parser.add_argument("--chunks-dir", type=Path, default=ROOT / "data/chunks")
    parser.add_argument("--output", type=Path, default=ROOT / "data/graph/legal_graph.json")
    args = parser.parse_args()
    print(json.dumps(build_graph_artifact(args.structured_dir, args.chunks_dir, args.output), indent=2))


if __name__ == "__main__":
    main()
