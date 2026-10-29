"""
Embedding service sử dụng BGE-M3 (BAAI/bge-m3).

Hỗ trợ:
  - Dense embedding  (1024-dim vectors, cosine similarity)
  - Sparse embedding (learned sparse, thay thế BM25)

Sử dụng FlagEmbedding cho full hybrid support.
Fallback sang sentence-transformers nếu chỉ cần dense.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import yaml

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingConfig:
    provider: str = "bge_m3"
    model_name: str = "BAAI/bge-m3"
    cache_dir: str | None = None
    device: str | None = None

    use_dense: bool = True
    use_sparse: bool = True
    use_colbert: bool = False

    max_length: int = 8192
    batch_size: int = 32
    normalize: bool = True
    dimensions: int | None = None
    timeout: float = 60.0
    max_retries: int = 3

    def __post_init__(self):
        if self.provider not in {"bge_m3", "openai"}:
            raise ValueError(f"Unknown embedding provider: {self.provider}")
        if self.batch_size <= 0 or self.max_length <= 0:
            raise ValueError("batch_size and max_length must be positive")
        if self.timeout <= 0 or self.max_retries < 0:
            raise ValueError("timeout must be positive and max_retries nonnegative")
        if self.provider == "openai":
            limits = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}
            if self.model_name not in limits:
                raise ValueError("OpenAI provider supports text-embedding-3-small/large")
            if self.dimensions is not None and not 1 <= self.dimensions <= limits[self.model_name]:
                raise ValueError("Invalid dimensions for the configured OpenAI model")
            if not self.use_dense or self.use_sparse or self.use_colbert:
                raise ValueError("OpenAI embeddings require dense=true, sparse=false, colbert=false")
            if self.max_length > 8192:
                raise ValueError("OpenAI embedding max_length must not exceed 8192")

    @property
    def fingerprint(self) -> str:
        dimension = self.dimensions or (1536 if self.model_name == "text-embedding-3-small" else 3072 if self.provider == "openai" else 1024)
        return f"{self.provider}:{self.model_name}:{dimension}:normalize={self.normalize}"

    @classmethod
    def from_yaml(cls, path: str | Path = _CONFIG_PATH) -> EmbeddingConfig:
        path = Path(path)
        if not path.exists():
            logger.warning("Config file %s not found, using defaults.", path)
            return cls()

        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        model_cfg = raw.get("model", {})
        embed_cfg = raw.get("embedding", {})
        return cls(
            provider=model_cfg.get("provider", "bge_m3"),
            model_name=model_cfg.get("name", cls.model_name),
            cache_dir=model_cfg.get("cache_dir"),
            device=model_cfg.get("device"),
            use_dense=embed_cfg.get("use_dense", cls.use_dense),
            use_sparse=embed_cfg.get("use_sparse", cls.use_sparse),
            use_colbert=embed_cfg.get("use_colbert", cls.use_colbert),
            max_length=embed_cfg.get("max_length", cls.max_length),
            batch_size=embed_cfg.get("batch_size", cls.batch_size),
            normalize=embed_cfg.get("normalize", cls.normalize),
            dimensions=embed_cfg.get("dimensions"),
            timeout=embed_cfg.get("timeout", cls.timeout),
            max_retries=embed_cfg.get("max_retries", cls.max_retries),
        )


# ---------------------------------------------------------------------------
# Embedding result
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingResult:
    """Kết quả embedding cho một hoặc nhiều text."""

    dense: list[list[float]] = field(default_factory=list)
    sparse: list[dict[str, Any]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.dense) if self.dense else len(self.sparse)


# ---------------------------------------------------------------------------
# BGE-M3 Embedder
# ---------------------------------------------------------------------------

class BGEM3Embedder:
    """
    Wrapper quanh BAAI/bge-m3 sử dụng thư viện FlagEmbedding.

    Cách dùng:
        embedder = BGEM3Embedder.from_config()
        result = embedder.encode(["Điều 1. Phạm vi điều chỉnh..."])
        print(result.dense[0][:5])   # [0.012, -0.034, ...]
        print(result.sparse[0])      # {"indices": [...], "values": [...]}
    """

    def __init__(self, config: EmbeddingConfig | None = None):
        self.config = config or EmbeddingConfig()
        if self.config.provider != "bge_m3":
            raise ValueError("BGEM3Embedder requires the bge_m3 provider; use get_embedder for provider selection")
        self._model = None

    @classmethod
    def from_config(cls, config_path: str | Path = _CONFIG_PATH) -> BGEM3Embedder:
        config = EmbeddingConfig.from_yaml(config_path)
        return cls(config)

    @property
    def model(self):
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self):
        """Load BGE-M3 model. Lazy-load để không tốn RAM khi import module."""
        model = self._try_load_flag_embedding()
        if model is not None:
            self._backend = "flag_embedding"
            return model

        # Fallback: sentence-transformers (chỉ hỗ trợ dense)
        model = self._try_load_sentence_transformers()
        if model is not None:
            self._backend = "sentence_transformers"
            if self.config.use_sparse:
                logger.warning(
                    "Sparse embedding không khả dụng với sentence-transformers. "
                    "Chỉ dùng dense embedding. "
                    "Để dùng sparse, cần fix FlagEmbedding (xem log ở trên)."
                )
                self.config.use_sparse = False
            return model

        raise ImportError(
            "Không thể load model embedding. Cần cài một trong:\n"
            "  pip install FlagEmbedding\n"
            "  pip install sentence-transformers"
        )

    def _try_load_flag_embedding(self):
        """Thử load bằng FlagEmbedding (hỗ trợ dense + sparse)."""
        try:
            from FlagEmbedding import BGEM3FlagModel
        except (ImportError, Exception) as exc:
            logger.warning(
                "FlagEmbedding không load được: %s\n"
                "  Thử fallback sang sentence-transformers...",
                exc,
            )
            return None

        logger.info(
            "Loading BGE-M3 (FlagEmbedding): %s (device=%s)",
            self.config.model_name,
            self.config.device or "auto",
        )

        kwargs: dict[str, Any] = {
            "use_fp16": self.config.device == "cuda",
        }
        if self.config.cache_dir:
            kwargs["cache_dir"] = self.config.cache_dir
        if self.config.device:
            kwargs["device"] = self.config.device

        try:
            model = BGEM3FlagModel(self.config.model_name, **kwargs)
            logger.info("BGE-M3 model loaded (FlagEmbedding).")
            return model
        except Exception as exc:
            logger.warning("FlagEmbedding model load thất bại: %s", exc)
            return None

    def _try_load_sentence_transformers(self):
        """Fallback: load bằng sentence-transformers (chỉ dense)."""
        try:
            from sentence_transformers import SentenceTransformer
        except (ImportError, Exception) as exc:
            logger.warning("sentence-transformers không load được: %s", exc)
            return None

        logger.info(
            "Loading BGE-M3 (sentence-transformers): %s (device=%s)",
            self.config.model_name,
            self.config.device or "auto",
        )

        kwargs: dict[str, Any] = {}
        if self.config.cache_dir:
            kwargs["cache_folder"] = self.config.cache_dir
        if self.config.device:
            kwargs["device"] = self.config.device

        try:
            model = SentenceTransformer(self.config.model_name, **kwargs)
            model.max_seq_length = self.config.max_length
            logger.info("BGE-M3 model loaded (sentence-transformers, dense only).")
            return model
        except Exception as exc:
            logger.warning("sentence-transformers model load thất bại: %s", exc)
            return None

    def encode(self, texts: list[str]) -> EmbeddingResult:
        """
        Encode danh sách text thành embedding vectors.

        Args:
            texts: Danh sách các đoạn text cần embed.

        Returns:
            EmbeddingResult chứa dense và/hoặc sparse vectors.
        """
        if not texts:
            return EmbeddingResult()

        # Resolve the lazy model first because loading it also selects the backend.
        # Reading _backend before this point routes the first fallback call through
        # the FlagEmbedding API even when sentence-transformers was loaded.
        _ = self.model
        backend = getattr(self, "_backend", "flag_embedding")

        if backend == "flag_embedding":
            return self._encode_flag_embedding(texts)
        return self._encode_sentence_transformers(texts)

    def _encode_flag_embedding(self, texts: list[str]) -> EmbeddingResult:
        """Encode bằng FlagEmbedding (dense + sparse)."""
        output = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            max_length=self.config.max_length,
            return_dense=self.config.use_dense,
            return_sparse=self.config.use_sparse,
            return_colbert_vecs=self.config.use_colbert,
        )

        result = EmbeddingResult()

        if self.config.use_dense and "dense_vecs" in output:
            vecs = output["dense_vecs"]
            result.dense = vecs.tolist() if hasattr(vecs, "tolist") else vecs

        if self.config.use_sparse and "lexical_weights" in output:
            result.sparse = self._convert_sparse(output["lexical_weights"])

        return result

    def _encode_sentence_transformers(self, texts: list[str]) -> EmbeddingResult:
        """Encode bằng sentence-transformers (dense only)."""
        vecs = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize,
            show_progress_bar=len(texts) > 10,
        )

        result = EmbeddingResult()
        result.dense = vecs.tolist() if hasattr(vecs, "tolist") else vecs
        return result

    def encode_single(self, text: str) -> EmbeddingResult:
        """Encode một đoạn text duy nhất."""
        return self.encode([text])

    @staticmethod
    def _convert_sparse(lexical_weights: list[dict]) -> list[dict[str, Any]]:
        """
        Chuyển đổi lexical_weights từ FlagEmbedding sang format
        {"indices": [...], "values": [...]} để lưu vào vector store.
        """
        sparse_list = []
        for token_weights in lexical_weights:
            indices = []
            values = []
            for token_id, weight in token_weights.items():
                indices.append(int(token_id))
                values.append(float(weight))
            sparse_list.append({"indices": indices, "values": values})
        return sparse_list

    @property
    def dense_dim(self) -> int:
        """Chiều của dense vector (1024 cho BGE-M3)."""
        return 1024


class Embedder(Protocol):
    config: EmbeddingConfig

    def encode(self, texts: list[str]) -> EmbeddingResult: ...
    def encode_single(self, text: str) -> EmbeddingResult: ...


class OpenAIEmbedder:
    """Dense embeddings with bounded requests; never silently truncate legal text."""

    def __init__(self, config: EmbeddingConfig, client=None, tokenizer=None):
        if config.provider != "openai":
            raise ValueError("OpenAIEmbedder requires the openai provider")
        self.config = config
        self._client = client
        self._tokenizer = tokenizer

    @property
    def client(self):
        if self._client is None:
            from dotenv import load_dotenv
            from openai import OpenAI

            load_dotenv(Path(__file__).resolve().parents[2] / ".env")
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("Set OPENAI_API_KEY in .env or the environment before embedding")
            self._client = OpenAI(timeout=self.config.timeout, max_retries=self.config.max_retries)
        return self._client

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            import tiktoken
            self._tokenizer = tiktoken.get_encoding("cl100k_base")
        return self._tokenizer

    @property
    def dense_dim(self) -> int:
        return self.config.dimensions or (1536 if self.config.model_name == "text-embedding-3-small" else 3072)

    def encode_single(self, text: str) -> EmbeddingResult:
        return self.encode([text])

    def encode(self, texts: list[str]) -> EmbeddingResult:
        # Validate the entire input before making any billable request.
        lengths = []
        for index, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Empty embedding text at index {index}")
            length = len(self.tokenizer.encode(text, disallowed_special=()))
            if length > self.config.max_length:
                raise ValueError(f"Text {index} has {length} tokens; re-chunk below {self.config.max_length}")
            lengths.append(length)
        result = EmbeddingResult()
        start = 0
        while start < len(texts):
            end, token_count = start, 0
            while end < len(texts) and end - start < min(self.config.batch_size, 2048):
                if token_count + lengths[end] > 300000:
                    break
                token_count += lengths[end]
                end += 1
            kwargs = {"model": self.config.model_name, "input": texts[start:end], "encoding_format": "float"}
            if self.config.dimensions is not None:
                kwargs["dimensions"] = self.config.dimensions
            response = self.client.embeddings.create(**kwargs)
            rows = sorted(response.data, key=lambda row: row.index)
            if [row.index for row in rows] != list(range(end - start)):
                raise ValueError("OpenAI returned missing or duplicate embedding indices")
            if any(len(row.embedding) != self.dense_dim for row in rows):
                raise ValueError("OpenAI returned unexpected vector dimensions")
            result.dense.extend([list(row.embedding) for row in rows])
            start = end
        return result


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

_default_embedder: Embedder | None = None
_embedders: dict[str, Embedder] = {}


def get_embedder(config_path: str | Path = _CONFIG_PATH) -> Embedder:
    """Cache by complete configuration, so switching models cannot reuse BGE."""
    if _default_embedder is not None and Path(config_path).resolve() == _CONFIG_PATH:
        return _default_embedder
    config = EmbeddingConfig.from_yaml(config_path)
    key = json.dumps(vars(config), sort_keys=True)
    if key not in _embedders:
        _embedders[key] = OpenAIEmbedder(config) if config.provider == "openai" else BGEM3Embedder(config)
    return _embedders[key]


def embed_texts(texts: list[str], config_path: str | Path = _CONFIG_PATH) -> EmbeddingResult:
    """Shortcut: embed danh sách text."""
    return get_embedder(config_path).encode(texts)


def embed_chunks_file(
    chunks_path: str | Path,
    output_path: str | Path | None = None,
    config_path: str | Path = _CONFIG_PATH,
    text_field: str = "embedding_text",
) -> Path:
    """
    Đọc file chunks JSON, embed từng chunk, ghi kết quả ra file mới.

    Mỗi chunk trong output sẽ có thêm:
      - "dense_vector": list[float] (nếu use_dense=true)
      - "sparse_vector": {"indices": [...], "values": [...]} (nếu use_sparse=true)

    Args:
        chunks_path:  Đường dẫn đến file *_chunks.json
        output_path:  Đường dẫn output. Mặc định: thêm suffix _embedded.json
        config_path:  Đường dẫn config YAML
        text_field:   Tên field chứa text cần embed (mặc định "embedding_text")

    Returns:
        Path đến file output đã ghi.
    """
    chunks_path = Path(chunks_path)
    if output_path is None:
        output_path = chunks_path.with_name(
            chunks_path.stem + "_embedded" + chunks_path.suffix
        )
    output_path = Path(output_path)

    with chunks_path.open("r", encoding="utf-8-sig") as f:
        chunks = json.load(f)

    logger.info("Embedding %d chunks from %s ...", len(chunks), chunks_path.name)

    embedder = get_embedder(config_path)
    texts = [chunk.get(text_field, chunk.get("text", "")) for chunk in chunks]

    # Encode theo batch
    all_dense: list[list[float]] = []
    all_sparse: list[dict[str, Any]] = []
    batch_size = embedder.config.batch_size

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        result = embedder.encode(batch)
        all_dense.extend(result.dense)
        all_sparse.extend(result.sparse)

        done = min(start + batch_size, len(texts))
        logger.info("  Embedded %d / %d chunks", done, len(texts))

    # Gắn vector vào chunks
    for idx, chunk in enumerate(chunks):
        chunk.pop("dense_vector", None)
        chunk.pop("sparse_vector", None)
        chunk["embedding_fingerprint"] = embedder.config.fingerprint
        if all_dense:
            chunk["dense_vector"] = all_dense[idx]
        if all_sparse:
            chunk["sparse_vector"] = all_sparse[idx]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    logger.info("Saved embedded chunks to %s", output_path)
    return output_path
