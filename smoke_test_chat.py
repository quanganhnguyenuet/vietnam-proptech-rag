"""Exercise the actual /chat endpoint in-process using the configured API key."""
import argparse
import json
from pathlib import Path

from fastapi.testclient import TestClient

from src.api.main import app

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="Theo văn bản trong corpus, khoản 9 Điều 138 Luật Đất đai 2024 quy định trách nhiệm gì?")
    parser.add_argument("--rerank", action="store_true", help="Also use the local BGE reranker; weights must already be cached")
    parser.add_argument("--output", type=Path, default=ROOT / "data/live_chat_api_demo.json")
    args = parser.parse_args()
    with TestClient(app) as client:
        response = client.post("/chat", json={"query": args.query, "top_k": 3,
                                               "include_evidence": True, "use_reranker": args.rerank})
    result = response.json()
    if response.status_code != 200:
        print(json.dumps({"http_status": response.status_code, "error": result.get("detail")}, ensure_ascii=True))
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    citations = result.get("citations", [])
    allowed = {source["label"] for source in result["evidence"]["agent_context"]["sources"]}
    valid = bool(citations) and all(item["label"] in allowed for item in citations)
    report = {"http_status": response.status_code, "status": result["status"], "model": result["model"],
              "citation_count": len(citations), "citation_labels_valid": valid,
              "retrieval": result["evidence"]["retrieval"]["method"],
              "reranker_enabled": result["evidence"]["retrieval"]["reranker_enabled"],
              "graph_enabled": result["evidence"]["retrieval"]["graph"]["enabled"],
              "usage": result["usage"], "output": str(args.output)}
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0 if result["status"] == "answered" and valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
