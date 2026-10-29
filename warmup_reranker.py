"""Download/load the configured local reranker and optionally run a Vietnamese smoke test."""
import argparse
import json
import os
import faulthandler
import time
from dataclasses import replace
from pathlib import Path

from src.retrieval.config import DEFAULT_RETRIEVAL_CONFIG, RetrievalConfig
from src.retrieval.reranker import CrossEncoderReranker

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_RETRIEVAL_CONFIG)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--local-only", action="store_true", help="Validate cached model without network access")
    parser.add_argument("--output", type=Path, help="Save the validation report as JSON")
    parser.add_argument("--http-download", action="store_true", help="Use standard HTTPS instead of Xet transfer")
    parser.add_argument("--force-download", action="store_true", help="Replace incomplete/corrupt cached model files")
    parser.add_argument("--timeout-seconds", type=int, default=900, help="Stop a stalled download/load after this total time")
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    if args.local_only and (args.force_download or args.http_download):
        parser.error("--local-only cannot be combined with download options")
    faulthandler.dump_traceback_later(args.timeout_seconds, exit=True)
    # This explicit preparation command may download files. Runtime search
    # defaults to cached local files so HTTP requests cannot hang on a download.
    config = replace(RetrievalConfig.from_yaml(args.config).reranker, local_files_only=args.local_only)
    if args.http_download:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    if args.force_download:
        from huggingface_hub import snapshot_download
        snapshot_download(config.model_name, cache_dir=str(ROOT / config.cache_dir),
                          allow_patterns=["*.json", "*.safetensors", "*.model"],
                          force_download=True, max_workers=2)
    reranker = CrossEncoderReranker(config, ROOT)
    started = time.monotonic()
    reranker.preload()
    report = {"model": config.model_name, "device": str(reranker.model.device), "loaded": True,
              "dtype": str(next(reranker.model.parameters()).dtype), "local_files_only": config.local_files_only,
              "batch_size": config.batch_size, "max_length": config.max_length,
              "load_seconds": round(time.monotonic() - started, 2)}
    if args.smoke_test:
        started = time.monotonic()
        ranked = reranker.rerank("Thủ tục cấp giấy chứng nhận quyền sử dụng đất", [
            {"chunk_id": "land", "text": "Hồ sơ đăng ký, cấp Giấy chứng nhận quyền sử dụng đất.", "metadata": {}},
            {"chunk_id": "other", "text": "Quy định về vận tải hành khách bằng đường hàng không.", "metadata": {}},
        ])
        report["smoke_test"] = [{"chunk_id": hit["chunk_id"], "score": hit["rerank_score"]} for hit in ranked]
        report["smoke_test_passed"] = ranked[0]["chunk_id"] == "land"
        report["inference_seconds"] = round(time.monotonic() - started, 2)
        if not report["smoke_test_passed"]:
            raise RuntimeError("Reranker did not rank the relevant Vietnamese passage first")
    if str(reranker.model.device).startswith("cuda"):
        import torch
        report["peak_cuda_memory_mib"] = round(torch.cuda.max_memory_allocated(reranker.model.device) / 1024**2, 1)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
