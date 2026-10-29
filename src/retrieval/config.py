"""Configuration for hybrid retrieval, local reranking and bounded graph search."""
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_RETRIEVAL_CONFIG = Path(__file__).resolve().parents[2] / "config/retrieval.yaml"


@dataclass(frozen=True)
class RerankerConfig:
    enabled: bool = True
    model_name: str = "BAAI/bge-reranker-v2-m3"
    device: str | None = None
    dtype: str = "auto"
    cache_dir: str = "data/models"
    batch_size: int = 1
    max_length: int = 1024
    window_overlap: int = 64
    local_files_only: bool = True

    def __post_init__(self):
        if self.dtype not in {"auto", "float32", "float16", "bfloat16"}:
            raise ValueError("Reranker dtype must be auto, float32, float16 or bfloat16")
        if self.batch_size <= 0 or self.max_length < 128:
            raise ValueError("Reranker batch_size must be positive and max_length >= 128")
        if self.model_name == "BAAI/bge-reranker-v2-m3" and self.max_length > 8192:
            raise ValueError("BGE reranker v2-m3 max_length must not exceed 8192")
        if not 0 <= self.window_overlap < self.max_length // 2:
            raise ValueError("Reranker overlap must be nonnegative and below half max_length")


@dataclass(frozen=True)
class GraphConfig:
    enabled: bool = True
    path: str = "data/graph/legal_graph.json"
    seed_k: int = 8
    max_hops: int = 2
    max_articles: int = 20
    max_chunks: int = 40
    chunks_per_article: int = 2
    context_k: int = 8

    def __post_init__(self):
        if not 1 <= self.max_hops <= 3:
            raise ValueError("graph.max_hops must be between 1 and 3")
        if any(value <= 0 for value in (self.seed_k, self.max_articles, self.max_chunks, self.chunks_per_article, self.context_k)):
            raise ValueError("Graph budgets must be positive")


@dataclass(frozen=True)
class RetrievalConfig:
    dense_candidates: int = 50
    bm25_candidates: int = 50
    candidate_pool: int = 80
    dense_weight: float = 1.0
    bm25_weight: float = 1.0
    rrf_k: int = 60
    max_context_chars: int = 24000
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)

    def __post_init__(self):
        if min(self.dense_candidates, self.bm25_candidates, self.candidate_pool, self.rrf_k, self.max_context_chars) <= 0:
            raise ValueError("Retrieval candidate and context budgets must be positive")
        if self.dense_weight <= 0 or self.bm25_weight <= 0:
            raise ValueError("Both vector and BM25 weights must be positive")

    @classmethod
    def from_yaml(cls, path: str | Path = DEFAULT_RETRIEVAL_CONFIG):
        with Path(path).open(encoding="utf-8-sig") as stream:
            raw = yaml.safe_load(stream) or {}
        return cls(**{key: value for key, value in raw.items() if key not in {"reranker", "graph"}},
                   reranker=RerankerConfig(**raw.get("reranker", {})),
                   graph=GraphConfig(**raw.get("graph", {})))
