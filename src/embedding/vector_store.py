"""
Vector Store service sử dụng ChromaDB (local persistent).

Hỗ trợ:
  - Upsert chunks với dense vectors + metadata
  - Semantic search (cosine similarity)
  - Quản lý collection

Lưu ý: ChromaDB chưa hỗ trợ sparse vector natively.
  Sparse vectors được lưu trong metadata để dùng cho hybrid search
  ở retriever layer sau này.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
import yaml

from src.embedding.embedder import Embedder, EmbeddingConfig, get_embedder

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class VectorStoreConfig:
    persist_directory: str = "data/vectordb"
    collection_name: str = "vn_legal_chunks"
    distance_metric: str = "cosine"

    @classmethod
    def from_yaml(cls, path: str | Path = _CONFIG_PATH) -> VectorStoreConfig:
        path = Path(path)
        if not path.exists():
            logger.warning("Config file %s not found, using defaults.", path)
            return cls()

        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        vs_cfg = raw.get("vector_store", {})
        return cls(
            persist_directory=vs_cfg.get("persist_directory", cls.persist_directory),
            collection_name=vs_cfg.get("collection_name", cls.collection_name),
            distance_metric=vs_cfg.get("distance_metric", cls.distance_metric),
        )


# ---------------------------------------------------------------------------
# ChromaDB Vector Store
# ---------------------------------------------------------------------------

class ChromaVectorStore:
    """
    Wrapper quanh ChromaDB để lưu và truy vấn embedded chunks.

    Cách dùng:
        store = ChromaVectorStore.from_config()
        store.upsert_chunks(chunks_with_vectors)
        results = store.search("quyền sử dụng đất", top_k=5)
    """

    def __init__(
        self,
        config: VectorStoreConfig | None = None,
        embedder: Embedder | None = None,
        project_root: Path | None = None,
        embedding_config: EmbeddingConfig | None = None,
    ):
        self.config = config or VectorStoreConfig()
        self._embedder = embedder
        supplied_config = getattr(embedder, "config", None)
        self.embedding_config = embedding_config or (supplied_config if isinstance(supplied_config, EmbeddingConfig) else EmbeddingConfig())
        self._project_root = project_root or Path(__file__).resolve().parents[2]
        self._client: chromadb.ClientAPI | None = None
        self._collection: chromadb.Collection | None = None

    @classmethod
    def from_config(
        cls,
        config_path: str | Path = _CONFIG_PATH,
        project_root: Path | None = None,
    ) -> ChromaVectorStore:
        if not Path(config_path).is_file():
            raise FileNotFoundError(f"Embedding config not found: {config_path}")
        config = VectorStoreConfig.from_yaml(config_path)
        return cls(config=config, embedder=get_embedder(config_path), project_root=project_root,
                   embedding_config=EmbeddingConfig.from_yaml(config_path))

    @property
    def client(self) -> chromadb.ClientAPI:
        if self._client is None:
            persist_dir = self._project_root / self.config.persist_directory
            persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(persist_dir))
            logger.info("ChromaDB client initialized at %s", persist_dir)
        return self._client

    @property
    def collection(self) -> chromadb.Collection:
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": self.config.distance_metric,
                          "embedding_fingerprint": self.embedding_config.fingerprint},
            )
            metadata = self._collection.metadata or {}
            actual_fingerprint = metadata.get("embedding_fingerprint")
            if actual_fingerprint and actual_fingerprint != self.embedding_config.fingerprint:
                self._collection = None
                raise ValueError("Collection embedding model/dimensions differ; choose a new collection and re-ingest")
            if not actual_fingerprint:
                if self._collection.count() and self.embedding_config.provider != "bge_m3":
                    self._collection = None
                    raise ValueError("Legacy collection has unknown embeddings; use a separate OpenAI collection")
                if not self._collection.count():
                    self._collection.modify(metadata={**metadata, "embedding_fingerprint": self.embedding_config.fingerprint})
                else:
                    logger.warning("Using legacy BGE collection without embedding fingerprint; re-ingest to verify provenance")
            actual_metric = (self._collection.metadata or {}).get("hnsw:space")
            if actual_metric and actual_metric != self.config.distance_metric:
                self._collection = None
                raise ValueError(
                    f"Collection '{self.config.collection_name}' uses "
                    f"'{actual_metric}', not configured metric "
                    f"'{self.config.distance_metric}'."
                )
            logger.info(
                "Collection '%s' ready (%d documents)",
                self.config.collection_name,
                self._collection.count(),
            )
        return self._collection

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = get_embedder()
        return self._embedder

    # ----- Upsert -----

    def upsert_chunks(self, chunks: list[dict], batch_size: int = 100) -> int:
        """
        Upsert chunks vào ChromaDB.

        Mỗi chunk cần có:
          - chunk_id: str
          - embedding_text: str (text đã chuẩn bị cho embedding)
          - text: str (text gốc để hiển thị)
          - metadata: dict

        Nếu chunk đã có "dense_vector", dùng luôn.
        Nếu không, tự embed embedding_text.

        Returns:
            Số chunks đã upsert.
        """
        total = 0
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        # Check all externally supplied vectors before writing any batch.
        for chunk in chunks:
            if chunk.get("dense_vector") is not None and chunk.get("embedding_fingerprint") != self.embedding_config.fingerprint:
                raise ValueError("Precomputed vectors must have a matching embedding_fingerprint; re-embed this file")
        if chunks:
            _ = self.collection  # reject incompatible collections before billable calls

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            ids = []
            embeddings = []
            documents = []
            metadatas = []
            texts_to_embed = []
            embed_indices = []

            for idx, chunk in enumerate(batch):
                chunk_id = chunk["chunk_id"]
                ids.append(chunk_id)
                documents.append(chunk.get("text", ""))

                # Chuẩn bị metadata cho ChromaDB (chỉ chấp nhận str, int, float, bool)
                meta = self._flatten_metadata(chunk.get("metadata", {}))

                # Lưu sparse vector vào metadata (dạng JSON string)
                if chunk.get("sparse_vector"):
                    meta["_sparse_vector"] = json.dumps(
                        chunk["sparse_vector"], ensure_ascii=False
                    )

                metadatas.append(meta)

                # Kiểm tra có sẵn dense vector chưa
                dense_vector = chunk.get("dense_vector")
                if dense_vector is not None:
                    embeddings.append(dense_vector)
                else:
                    embeddings.append(None)
                    texts_to_embed.append(
                        chunk.get("embedding_text", chunk.get("text", ""))
                    )
                    embed_indices.append(idx)

            # Embed những chunk chưa có vector
            if texts_to_embed:
                result = self.embedder.encode(texts_to_embed)
                if len(result.dense) != len(texts_to_embed):
                    raise ValueError(
                        "Dense embedding count does not match the number of input texts: "
                        f"{len(result.dense)} != {len(texts_to_embed)}"
                    )
                if result.sparse and len(result.sparse) != len(texts_to_embed):
                    raise ValueError(
                        "Sparse embedding count does not match the number of input texts: "
                        f"{len(result.sparse)} != {len(texts_to_embed)}"
                    )
                for i, embed_idx in enumerate(embed_indices):
                    embeddings[embed_idx] = result.dense[i]
                    if result.sparse:
                        metadatas[embed_idx]["_sparse_vector"] = json.dumps(
                            result.sparse[i], ensure_ascii=False
                        )

            # Upsert vào ChromaDB
            expected_dim = self.embedding_config.dimensions or (1536 if self.embedding_config.model_name == "text-embedding-3-small" else 3072 if self.embedding_config.provider == "openai" else 1024)
            if any(vector is None or len(vector) != expected_dim for vector in embeddings):
                raise ValueError(f"Expected dense embeddings of dimension {expected_dim}")
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )

            self.refresh_lexical_index()
            total += len(batch)
            logger.info("  Upserted %d / %d chunks", total, len(chunks))

        return total

    # ----- Search -----

    def search(
        self,
        query: str,
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        """
        Semantic search trong collection.

        Args:
            query:  Câu truy vấn
            top_k:  Số kết quả trả về
            where:  Bộ lọc metadata (ChromaDB where clause)

        Returns:
            List of dicts, mỗi dict chứa:
              - chunk_id, text, metadata, distance, score
        """
        if top_k <= 0 or self.count() == 0:
            return []

        query_result = self.embedder.encode_single(query)
        if not query_result.dense:
            raise ValueError("Dense embedding is required for vector search.")
        query_vector = query_result.dense[0]

        return self.search_by_vector(query_vector, top_k=top_k, where=where)

    def search_by_vector(
        self,
        query_vector: list[float],
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        """Search with a precomputed dense vector."""
        collection_count = self.count()
        if top_k <= 0 or collection_count == 0:
            return []

        kwargs: dict[str, Any] = {
            "query_embeddings": [query_vector],
            "n_results": min(top_k, collection_count),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)

        # Parse kết quả
        hits = []
        for i in range(len(results["ids"][0])):
            raw_meta = results["metadatas"][0][i] if results["metadatas"] else {}
            meta = self._public_metadata(raw_meta)
            distance = results["distances"][0][i] if results["distances"] else None

            hits.append(
                {
                    "chunk_id": results["ids"][0][i],
                    "text": results["documents"][0][i] if results["documents"] else "",
                    "metadata": meta,
                    "distance": distance,
                    "score": self._distance_to_score(distance),
                }
            )

        return hits

    def hybrid_search(
        self, query: str, top_k: int = 5, where: dict | None = None,
        dense_candidates: int = 50, sparse_candidates: int = 50,
        rrf_k: int = 60, dense_weight: float = 1.0, sparse_weight: float = 1.0,
        *, bm25_candidates: int | None = None, bm25_weight: float | None = None,
    ) -> list[dict]:
        """Always fuse BM25 and vector search, including when BGE returns sparse vectors.

        sparse_candidates/sparse_weight remain aliases for older callers.
        """
        from src.retrieval.hybrid import hybrid_search
        return hybrid_search(
            self, query, top_k=top_k, where=where,
            dense_candidates=dense_candidates,
            bm25_candidates=bm25_candidates if bm25_candidates is not None else sparse_candidates,
            rrf_k=rrf_k, dense_weight=dense_weight,
            bm25_weight=bm25_weight if bm25_weight is not None else sparse_weight,
        )

    def refresh_lexical_index(self) -> None:
        """Invalidate cached BM25 indexes after a corpus update."""
        self._bm25_cache = None

    def get_chunks(self, ids: list[str]) -> list[dict]:
        if not ids:
            return []
        rows = self.collection.get(ids=ids, include=["documents", "metadatas"])
        return [{"chunk_id": chunk_id, "text": rows["documents"][i] or "",
                 "metadata": self._public_metadata(rows["metadatas"][i]) if rows["metadatas"] else {}}
                for i, chunk_id in enumerate(rows["ids"])]

    # ----- Utils -----

    def count(self) -> int:
        """Số documents trong collection."""
        return self.collection.count()

    def delete_collection(self) -> None:
        """Xoá toàn bộ collection."""
        self.client.delete_collection(self.config.collection_name)
        self._collection = None
        self.refresh_lexical_index()
        logger.info("Deleted collection '%s'", self.config.collection_name)

    def close(self) -> None:
        """Release Chroma resources and persistent file handles."""
        if self._client is not None:
            close = getattr(self._client, "close", None)
            if callable(close):
                close()
        self._collection = None
        self._client = None

    def __enter__(self) -> ChromaVectorStore:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _distance_to_score(self, distance: float | None) -> float | None:
        if distance is None:
            return None
        if self.config.distance_metric == "l2":
            return 1.0 / (1.0 + max(distance, 0.0))
        return 1.0 - distance

    @staticmethod
    def _lexical_tokens(text: str) -> list[str]:
        return re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold())

    @staticmethod
    def _public_metadata(metadata: dict | None) -> dict:
        public = dict(metadata or {})
        public.pop("_sparse_vector", None)
        return public

    @staticmethod
    def _sparse_dot(left: dict, right: dict) -> float:
        left_values = dict(zip(left["indices"], left["values"]))
        right_values = dict(zip(right["indices"], right["values"]))
        if len(left_values) > len(right_values):
            left_values, right_values = right_values, left_values
        return sum(
            float(value) * float(right_values.get(index, 0.0))
            for index, value in left_values.items()
        )

    @staticmethod
    def _flatten_metadata(metadata: dict) -> dict:
        """
        Flatten metadata cho ChromaDB.
        ChromaDB chỉ chấp nhận value kiểu str, int, float, bool.
        """
        flat = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                flat[key] = value
            elif isinstance(value, list):
                # Convert list → JSON string
                flat[key] = json.dumps(value, ensure_ascii=False)
            else:
                flat[key] = str(value)
        return flat


# ---------------------------------------------------------------------------
# Ingestion: Chunks JSON → Embed → ChromaDB
# ---------------------------------------------------------------------------

def ingest_chunks_to_store(
    chunks_dir: str | Path,
    config_path: str | Path = _CONFIG_PATH,
    project_root: Path | None = None,
    prune: bool = False,
) -> int:
    """
    Pipeline đầy đủ: đọc tất cả file *_chunks.json → embed → upsert vào ChromaDB.

    Args:
        chunks_dir:    Thư mục chứa file *_chunks.json
        config_path:   Đường dẫn config YAML
        project_root:  Thư mục gốc project
        prune:         Xóa ID trong collection nếu không còn trong các file chunk

    Returns:
        Tổng số chunks đã ingest.
    """
    chunks_dir = Path(chunks_dir)
    total = 0
    indexed_ids: set[str] = set()
    chunk_paths = sorted(chunks_dir.glob("*_chunks.json"))
    if not chunk_paths:
        raise FileNotFoundError(f"No *_chunks.json files found in {chunks_dir}")

    chunk_batches: list[tuple[Path, list[dict]]] = []
    for chunks_path in chunk_paths:
        with chunks_path.open("r", encoding="utf-8-sig") as f:
            chunks = json.load(f)

        for chunk in chunks:
            chunk_id = chunk.get("chunk_id")
            if not chunk_id:
                raise ValueError(f"{chunks_path.name} contains a chunk without chunk_id")
            if chunk_id in indexed_ids:
                raise ValueError(f"Duplicate chunk_id across input files: {chunk_id}")
            indexed_ids.add(chunk_id)
        chunk_batches.append((chunks_path, chunks))

    if not indexed_ids:
        raise ValueError(f"No chunks found in {chunks_dir}")

    with ChromaVectorStore.from_config(config_path, project_root) as store:
        existing_ids: set[str] = set()
        if prune:
            existing = store.collection.get(include=[])
            existing_ids = set(existing.get("ids") or [])

        for chunks_path, chunks in chunk_batches:
            logger.info("[Ingest] %s — %d chunks", chunks_path.name, len(chunks))
            count = store.upsert_chunks(chunks)
            total += count

        if prune:
            stale_ids = sorted(existing_ids - indexed_ids)
            for start in range(0, len(stale_ids), 1000):
                store.collection.delete(ids=stale_ids[start : start + 1000])
            if stale_ids:
                logger.info("[Ingest] Removed %d stale chunks.", len(stale_ids))

            actual_ids = set(store.collection.get(include=[]).get("ids") or [])
            if actual_ids != indexed_ids:
                missing = len(indexed_ids - actual_ids)
                extra = len(actual_ids - indexed_ids)
                raise RuntimeError(
                    "Vector store synchronization failed: "
                    f"missing={missing}, extra={extra}"
                )

        logger.info("[Ingest] Done. Total: %d chunks in collection.", store.count())
    return total


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    root = Path(__file__).resolve().parents[2]
    chunks_dir = root / "data" / "chunks"

    if not chunks_dir.exists():
        print(f"[!] Không tìm thấy thư mục chunks: {chunks_dir}")
        sys.exit(1)

    total = ingest_chunks_to_store(chunks_dir, project_root=root)
    print(f"\n[OK] Ingested {total} chunks into ChromaDB.")
