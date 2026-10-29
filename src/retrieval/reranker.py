"""Local cross-encoder reranking with token windows for long legal passages."""
import math
from pathlib import Path
from threading import Lock

from src.retrieval.config import RerankerConfig


class RerankerUnavailable(RuntimeError):
    pass


class CrossEncoderReranker:
    def __init__(self, config: RerankerConfig, project_root: Path, model=None):
        self.config = config
        self.project_root = project_root
        self._model = model
        self._lock = Lock()

    @property
    def model(self):
        if self._model is None:
            try:
                import torch
                from sentence_transformers import CrossEncoder
                device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
                dtype = self.config.dtype
                if dtype == "auto":
                    dtype = "float16" if device.startswith("cuda") else "float32"
                cache_dir = str(self.project_root / self.config.cache_dir)
                model_path = self.config.model_name
                if self.config.local_files_only:
                    # Some AutoProcessor versions probe missing processor files
                    # online even when the Hub model was requested local-only.
                    # A resolved snapshot directory prevents all Hub probes.
                    candidate = self.project_root / self.config.model_name
                    if candidate.is_dir():
                        model_path = str(candidate.resolve())
                    else:
                        from huggingface_hub import snapshot_download
                        model_path = snapshot_download(self.config.model_name, cache_dir=cache_dir,
                                                       local_files_only=True)
                self._model = CrossEncoder(
                    model_path, device=device,
                    max_length=self.config.max_length,
                    cache_folder=None if self.config.local_files_only else cache_dir,
                    model_kwargs={"dtype": getattr(torch, dtype)},
                    local_files_only=self.config.local_files_only,
                    trust_remote_code=False,
                )
            except Exception as exc:
                raise RerankerUnavailable(
                    "Cannot load the local reranker. Install requirements-rerank.txt and download "
                    f"{self.config.model_name} with warmup_reranker.py. Check CUDA/device/dtype "
                    "settings, or explicitly disable reranker in retrieval config."
                ) from exc
        return self._model

    def rerank(self, query: str, hits: list[dict]) -> list[dict]:
        if not hits:
            return []
        with self._lock:
            tokenizer = self.model.tokenizer
            query_length = len(tokenizer.encode(query, add_special_tokens=False))
            # Never silently truncate the question or tail of a legal provision.
            budget = self.config.max_length - query_length - tokenizer.num_special_tokens_to_add(pair=True) - 8
            if budget <= self.config.window_overlap:
                raise ValueError("Question leaves no reranking passage budget; increase reranker.max_length")
            pairs, owners = [], []
            window_counts = [0] * len(hits)
            for owner, hit in enumerate(hits):
                meta = hit.get("metadata") or {}
                text = f"{meta.get('citation', '')}\n{hit.get('text', '')}".strip()
                token_ids = tokenizer.encode(text, add_special_tokens=False)
                if not token_ids:
                    raise ValueError(f"Empty reranking passage: {hit['chunk_id']}")
                for start in range(0, len(token_ids), budget - self.config.window_overlap):
                    passage = tokenizer.decode(token_ids[start:start + budget], skip_special_tokens=True)
                    actual_length = len(tokenizer.encode(passage, add_special_tokens=False))
                    if actual_length + query_length + tokenizer.num_special_tokens_to_add(pair=True) > self.config.max_length:
                        raise ValueError("Tokenizer round trip exceeds reranker token budget")
                    pairs.append((query, passage))
                    owners.append(owner)
                    window_counts[owner] += 1
                    if start + budget >= len(token_ids):
                        break
            scores = self.model.predict(pairs, batch_size=self.config.batch_size,
                                        show_progress_bar=False, convert_to_numpy=True)
            scores = scores.reshape(-1).tolist() if hasattr(scores, "reshape") else list(scores)
            if len(scores) != len(pairs) or any(not math.isfinite(float(score)) for score in scores):
                raise ValueError("Reranker returned missing, invalid or non-finite scores")
            best = [-math.inf] * len(hits)
            for owner, score in zip(owners, scores):
                best[owner] = max(best[owner], float(score))
        ranked = [{**hit, "retrieval_score": hit.get("score"), "rerank_score": best[i],
                   "rerank_windows": window_counts[i], "reranker_model": self.config.model_name,
                   "score": best[i]} for i, hit in enumerate(hits)]
        return sorted(ranked, key=lambda hit: (-hit["score"], hit["chunk_id"]))

    def preload(self) -> None:
        with self._lock:
            _ = self.model
