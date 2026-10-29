"""Ask a question through retrieval, graph, reranking and answer generation."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

from dotenv import load_dotenv
from openai import APIError

from src.embedding.vector_store import ChromaVectorStore
from src.retrieval.config import RetrievalConfig, DEFAULT_RETRIEVAL_CONFIG
from src.retrieval.retriever import LegalRetriever
from src.rag.generator import AnswerGenerator, LLMConfig, DEFAULT_LLM_CONFIG
from src.rag.service import LegalChatService
from src.rag.session import ChatSession
from src.rag.generator import GenerationError
from src.retrieval.legal_graph import GraphUnavailable
from src.retrieval.reranker import RerankerUnavailable

ROOT = Path(__file__).resolve().parent


def print_answer(answer: dict, *, as_json: bool = False):
    if as_json:
        print(json.dumps(answer, ensure_ascii=False, indent=2))
        return
    print(answer["answer"])
    if answer.get("citations"):
        print("\nNguồn tham khảo:")
        for source in answer["citations"]:
            print(f"[{source['label']}] {source.get('citation', '')} — {source.get('source_url', '')}")


def api_error_message(exc: APIError) -> str:
    return json.dumps({"error": "OpenAI request failed", "error_type": type(exc).__name__,
                       "status_code": getattr(exc, "status_code", None), "code": getattr(exc, "code", None)})


def run_chat_loop(service, *, initial_query=None, top_k=5, include_evidence=False,
                  as_json=False, output: Path | None = None):
    session = ChatSession(service)
    transcript = []
    print("Chat pháp luật — nhập end để kết thúc.")
    pending = initial_query
    while True:
        try:
            if pending is None:
                query = input("\nBạn: ")
            else:
                query, pending = pending, None
                print(f"\nBạn: {query}")
            query = query.strip()
            if query.casefold() == "end":
                break
            if not query:
                continue
            answer = session.answer(query, top_k, include_evidence=include_evidence)
            print("\nAgent:")
            print_answer(answer, as_json=as_json)
            if output:
                transcript.append({"query": query, "response": answer})
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps({"turns": transcript}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except (EOFError, KeyboardInterrupt):
            break
        except APIError as exc:
            print(api_error_message(exc), file=sys.stderr)
            print("Lượt này chưa hoàn tất. Bạn có thể thử lại hoặc nhập end.")
        except (GenerationError, GraphUnavailable, RerankerUnavailable, ValueError) as exc:
            print(f"Không thể hoàn tất lượt này: {exc}", file=sys.stderr)
            print("Bạn có thể thử lại hoặc nhập end.")
    print("Đã kết thúc cuộc trò chuyện.")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", help="Omit to start an interactive conversation")
    parser.add_argument("--interactive", "-i", action="store_true", help="Continue chatting after the initial question; type end to exit")
    parser.add_argument("--config", type=Path, default=Path(os.getenv("EMBEDDING_CONFIG", "src/embedding/config.openai.yaml")))
    parser.add_argument("--retrieval-config", type=Path, default=Path(os.getenv("RETRIEVAL_CONFIG", str(DEFAULT_RETRIEVAL_CONFIG))))
    parser.add_argument("--llm-config", type=Path, default=Path(os.getenv("LLM_CONFIG", str(DEFAULT_LLM_CONFIG))))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-rerank", action="store_true", help="Explicit test without the local reranker")
    parser.add_argument("--no-graph", action="store_true")
    parser.add_argument("--include-evidence", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print the full API payload instead of the chatbot answer")
    parser.add_argument("--output", type=Path, help="Save one answer, or the conversation transcript in interactive mode, as JSON")
    args = parser.parse_args()
    config = RetrievalConfig.from_yaml(args.retrieval_config)
    if args.no_rerank:
        config = replace(config, reranker=replace(config.reranker, enabled=False))
    if args.no_graph:
        config = replace(config, graph=replace(config.graph, enabled=False))
    with ChromaVectorStore.from_config(args.config, ROOT) as store:
        service = LegalChatService(LegalRetriever(store, config, project_root=ROOT),
                                   AnswerGenerator(LLMConfig.from_yaml(args.llm_config)))
        if args.query is None or args.interactive:
            run_chat_loop(service, initial_query=args.query, top_k=args.top_k,
                          include_evidence=args.include_evidence, as_json=args.json, output=args.output)
            return
        answer = service.answer(args.query, args.top_k, include_evidence=args.include_evidence)
    payload = json.dumps(answer, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print_answer(answer, as_json=args.json)


if __name__ == "__main__":
    try:
        main()
    except APIError as exc:
        print(api_error_message(exc), file=sys.stderr)
        raise SystemExit(1) from None
