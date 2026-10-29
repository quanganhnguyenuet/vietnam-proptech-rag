"""Small live probes. Never print credentials or raw provider error messages."""
import argparse
import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, APIError

from src.rag.generator import LLMConfig, DEFAULT_LLM_CONFIG

ROOT = Path(__file__).resolve().parent


def main():
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm-config", type=Path, default=DEFAULT_LLM_CONFIG)
    args = parser.parse_args()
    config = LLMConfig.from_yaml(args.llm_config)
    client = OpenAI(timeout=15, max_retries=0)
    report = {}
    for stage in ("embedding", "generation"):
        try:
            if stage == "embedding":
                response = client.embeddings.create(model="text-embedding-3-small", input=["Kiểm tra kết nối embedding pháp luật Việt Nam."])
                report[stage] = {"ok": True, "model": response.model, "dimensions": len(response.data[0].embedding)}
            else:
                response = client.responses.create(model=config.model, input="Reply with exactly OK.", max_output_tokens=20, store=False)
                report[stage] = {"ok": response.status == "completed", "model": response.model}
        except APIError as exc:
            report[stage] = {"ok": False, "error_type": type(exc).__name__,
                             "status_code": getattr(exc, "status_code", None), "code": getattr(exc, "code", None)}
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0 if all(result["ok"] for result in report.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
