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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
import yaml

from src.embedding.embedder import BGEM3Embedder, EmbeddingConfig, get_embedder

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
        embedder: BGEM3Embedder | None = None,
        project_root: Path | None = None,
    ):
        self.config = config or VectorStoreConfig()
        self._embedder = embedder
        self._project_root = project_root or Path(__file__).resolve().parents[2]
        self._client: chromadb.ClientAPI | None = None
        self._collection: chromadb.Collection | None = None

    @classmethod
    def from_config(
        cls,
        config_path: str | Path = _CONFIG_PATH,
        project_root: Path | None = None,
    ) -> ChromaVectorStore:
        config = VectorStoreConfig.from_yaml(config_path)
        return cls(config=config, project_root=project_root)

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
                metadata={"hnsw:space": self.config.distance_metric},
            )
            actual_metric = (self._collection.metadata or {}).get("hnsw:space")
            if actual_metric and actual_metric != self.config.distance_metric:
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
    def embedder(self) -> BGEM3Embedder:
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
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )

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
        self,
        query: str,
        top_k: int = 5,
        where: dict | None = None,
        dense_candidates: int = 50,
        sparse_candidates: int = 50,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ) -> list[dict]:
        """Combine dense and learned-sparse rankings with weighted RRF."""
        if top_k <= 0 or self.count() == 0:
            return []
        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero.")

        query_result = self.embedder.encode_single(query)
        if not query_result.dense:
            raise ValueError("Dense embedding is required for hybrid search.")

        dense_hits = self.search_by_vector(
            query_result.dense[0],
            top_k=max(top_k, dense_candidates),
            where=where,
        )
        if not query_result.sparse:
            logger.warning("Query has no sparse embedding; returning dense results.")
            return dense_hits[:top_k]

        get_kwargs: dict[str, Any] = {"include": ["documents", "metadatas"]}
        if where:
            get_kwargs["where"] = where
        rows = self.collection.get(**get_kwargs)

        sparse_hits = []
        for idx, chunk_id in enumerate(rows["ids"]):
            metadata = rows["metadatas"][idx] if rows["metadatas"] else {}
            raw_sparse = (metadata or {}).get("_sparse_vector")
            if not raw_sparse:
                continue
            try:
                sparse_vector = json.loads(raw_sparse)
                sparse_score = self._sparse_dot(query_result.sparse[0], sparse_vector)
            except (TypeError, ValueError, KeyError, json.JSONDecodeError):
                logger.warning("Ignoring invalid sparse vector for chunk '%s'.", chunk_id)
                continue
            if sparse_score <= 0:
                continue
            sparse_hits.append(
                {
                    "chunk_id": chunk_id,
                    "text": rows["documents"][idx] if rows["documents"] else "",
                    "metadata": self._public_metadata(metadata),
                    "distance": None,
                    "sparse_score": sparse_score,
                }
            )

        if not sparse_hits:
            logger.warning(
                "Collection has no usable sparse vectors; returning dense results. "
                "Re-run ingestion to populate sparse metadata."
            )
            return dense_hits[:top_k]

        sparse_hits.sort(key=lambda hit: (-hit["sparse_score"], hit["chunk_id"]))
        sparse_hits = sparse_hits[: max(top_k, sparse_candidates)]

        fused: dict[str, dict] = {}
        for rank, hit in enumerate(dense_hits, 1):
            fused[hit["chunk_id"]] = {
                **hit,
                "dense_rank": rank,
                "sparse_rank": None,
                "sparse_score": None,
                "score": dense_weight / (rrf_k + rank),
            }

        for rank, hit in enumerate(sparse_hits, 1):
            existing = fused.get(hit["chunk_id"])
            contribution = sparse_weight / (rrf_k + rank)
            if existing is None:
                fused[hit["chunk_id"]] = {
                    **hit,
                    "dense_rank": None,
                    "sparse_rank": rank,
                    "score": contribution,
                }
            else:
                existing["sparse_rank"] = rank
                existing["sparse_score"] = hit["sparse_score"]
                existing["score"] += contribution

        return sorted(
            fused.values(),
            key=lambda hit: (-hit["score"], hit["chunk_id"]),
        )[:top_k]

    # ----- Utils -----

    def count(self) -> int:
        """Số documents trong collection."""
        return self.collection.count()

    def delete_collection(self) -> None:
        """Xoá toàn bộ collection."""
        self.client.delete_collection(self.config.collection_name)
        self._collection = None
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
