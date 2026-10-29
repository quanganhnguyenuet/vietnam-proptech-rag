"""
Test embedding pipeline: config, embedder, vector store.

Chạy: python -m pytest tests/test_embedding.py -v
       hoặc: python -m unittest tests/test_embedding.py -v

Lưu ý:
  - TestEmbedderLive và TestVectorStoreLive cần GPU/CPU đủ mạnh
    và sẽ download model BGE-M3 (~2GB) lần đầu.
  - Mặc định các test live bị skip, bỏ comment @unittest.skip để chạy.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

from src.embedding.embedder import BGEM3Embedder, EmbeddingConfig, EmbeddingResult

RUN_LIVE_TESTS = os.getenv("RUN_EMBEDDING_LIVE_TESTS") == "1"


# ===========================================================================
# Test Config
# ===========================================================================

class TestEmbeddingConfig(unittest.TestCase):
    """Test đọc config từ YAML."""

    def test_default_config(self):
        config = EmbeddingConfig()
        self.assertEqual(config.model_name, "BAAI/bge-m3")
        self.assertTrue(config.use_dense)
        self.assertTrue(config.use_sparse)
        self.assertFalse(config.use_colbert)
        self.assertEqual(config.max_length, 8192)
        self.assertEqual(config.batch_size, 32)
        self.assertTrue(config.normalize)

    def test_from_yaml(self):
        yaml_content = {
            "model": {
                "name": "BAAI/bge-m3",
                "cache_dir": "/tmp/models",
                "device": "cpu",
            },
            "embedding": {
                "use_dense": True,
                "use_sparse": False,
                "use_colbert": False,
                "max_length": 4096,
                "batch_size": 16,
                "normalize": True,
            },
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            yaml.dump(yaml_content, f)
            tmp_path = f.name

        config = EmbeddingConfig.from_yaml(tmp_path)
        self.assertEqual(config.model_name, "BAAI/bge-m3")
        self.assertEqual(config.cache_dir, "/tmp/models")
        self.assertEqual(config.device, "cpu")
        self.assertFalse(config.use_sparse)
        self.assertEqual(config.max_length, 4096)
        self.assertEqual(config.batch_size, 16)

        Path(tmp_path).unlink()

    def test_from_yaml_missing_file(self):
        """Nếu file không tồn tại, trả về config mặc định."""
        config = EmbeddingConfig.from_yaml("/nonexistent/path.yaml")
        self.assertEqual(config.model_name, "BAAI/bge-m3")

    def test_from_yaml_empty_file(self):
        """Nếu file rỗng, trả về config mặc định."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write("")
            tmp_path = f.name

        config = EmbeddingConfig.from_yaml(tmp_path)
        self.assertEqual(config.model_name, "BAAI/bge-m3")

        Path(tmp_path).unlink()


# ===========================================================================
# Test EmbeddingResult
# ===========================================================================

class TestEmbeddingResult(unittest.TestCase):

    def test_empty_result(self):
        result = EmbeddingResult()
        self.assertEqual(len(result), 0)

    def test_result_with_dense(self):
        result = EmbeddingResult(dense=[[0.1, 0.2], [0.3, 0.4]])
        self.assertEqual(len(result), 2)

    def test_result_with_sparse(self):
        result = EmbeddingResult(
            sparse=[{"indices": [1, 2], "values": [0.5, 0.6]}]
        )
        self.assertEqual(len(result), 1)


# ===========================================================================
# Test Embedder (mocked model)
# ===========================================================================

class TestBGEM3EmbedderMocked(unittest.TestCase):
    """Test embedder logic với mock model (không cần download model thật)."""

    def _make_embedder_with_mock(self):
        """Tạo embedder với model giả."""
        import numpy as np

        embedder = BGEM3Embedder(EmbeddingConfig(use_dense=True, use_sparse=True))

        mock_model = MagicMock()
        mock_model.encode.return_value = {
            "dense_vecs": np.random.rand(2, 1024).astype(np.float32),
            "lexical_weights": [
                {101: 0.5, 202: 0.3},
                {303: 0.8, 404: 0.1, 505: 0.6},
            ],
        }
        embedder._model = mock_model
        return embedder

    def test_encode_returns_dense_and_sparse(self):
        embedder = self._make_embedder_with_mock()
        result = embedder.encode(["text 1", "text 2"])

        self.assertEqual(len(result.dense), 2)
        self.assertEqual(len(result.dense[0]), 1024)
        self.assertEqual(len(result.sparse), 2)

        # Sparse phải có indices và values
        for sp in result.sparse:
            self.assertIn("indices", sp)
            self.assertIn("values", sp)
            self.assertEqual(len(sp["indices"]), len(sp["values"]))

    def test_encode_empty_list(self):
        embedder = self._make_embedder_with_mock()
        result = embedder.encode([])
        self.assertEqual(len(result), 0)

    def test_encode_single(self):
        import numpy as np

        embedder = BGEM3Embedder(EmbeddingConfig(use_dense=True, use_sparse=False))
        mock_model = MagicMock()
        mock_model.encode.return_value = {
            "dense_vecs": np.random.rand(1, 1024).astype(np.float32),
        }
        embedder._model = mock_model

        result = embedder.encode_single("Điều 1. Phạm vi điều chỉnh")
        self.assertEqual(len(result.dense), 1)
        self.assertEqual(len(result.dense[0]), 1024)

    def test_dense_dim(self):
        embedder = BGEM3Embedder()
        self.assertEqual(embedder.dense_dim, 1024)

    def test_convert_sparse(self):
        lexical_weights = [
            {101: 0.5, 202: 0.3},
            {303: 0.8},
        ]
        sparse = BGEM3Embedder._convert_sparse(lexical_weights)

        self.assertEqual(len(sparse), 2)
        self.assertEqual(sparse[0]["indices"], [101, 202])
        self.assertAlmostEqual(sparse[0]["values"][0], 0.5)
        self.assertEqual(sparse[1]["indices"], [303])

    def test_first_encode_uses_sentence_transformers_fallback(self):
        import numpy as np

        class FakeSentenceTransformer:
            def encode(
                self,
                texts,
                batch_size,
                normalize_embeddings,
                show_progress_bar,
            ):
                return np.zeros((len(texts), 1024), dtype=np.float32)

        embedder = BGEM3Embedder(EmbeddingConfig(use_sparse=True))
        embedder._try_load_flag_embedding = MagicMock(return_value=None)
        embedder._try_load_sentence_transformers = MagicMock(
            return_value=FakeSentenceTransformer()
        )

        result = embedder.encode(["quyền sử dụng đất"])

        self.assertEqual(embedder._backend, "sentence_transformers")
        self.assertFalse(embedder.config.use_sparse)
        self.assertEqual(len(result.dense[0]), 1024)


# ===========================================================================
# Test embed_chunks_file (mocked)
# ===========================================================================

class TestEmbedChunksFile(unittest.TestCase):

    def test_embed_chunks_file_creates_output(self):
        import numpy as np
        from src.embedding.embedder import embed_chunks_file, _default_embedder
        import src.embedding.embedder as embedder_module

        # Tạo file chunks tạm
        chunks = [
            {
                "chunk_id": "test__article_1",
                "text": "Điều 1. Phạm vi điều chỉnh",
                "embedding_text": "Luật Đất đai 2024 Điều 1 Phạm vi điều chỉnh",
                "metadata": {"unit_id": "test__article_1", "doc_id": "test"},
            },
            {
                "chunk_id": "test__article_2",
                "text": "Điều 2. Đối tượng áp dụng",
                "embedding_text": "Luật Đất đai 2024 Điều 2 Đối tượng áp dụng",
                "metadata": {"unit_id": "test__article_2", "doc_id": "test"},
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            chunks_path = Path(tmpdir) / "test_chunks.json"
            with chunks_path.open("w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False)

            # Mock embedder
            mock_embedder = MagicMock(spec=BGEM3Embedder)
            mock_embedder.config = EmbeddingConfig(batch_size=2)
            mock_embedder.encode.return_value = EmbeddingResult(
                dense=np.random.rand(2, 1024).tolist(),
                sparse=[
                    {"indices": [1, 2], "values": [0.5, 0.3]},
                    {"indices": [3, 4], "values": [0.8, 0.1]},
                ],
            )

            # Patch singleton
            old_default = embedder_module._default_embedder
            embedder_module._default_embedder = mock_embedder

            try:
                output_path = embed_chunks_file(chunks_path)

                self.assertTrue(output_path.exists())
                self.assertIn("_embedded", output_path.name)

                with output_path.open("r", encoding="utf-8") as f:
                    result = json.load(f)

                self.assertEqual(len(result), 2)
                self.assertIn("dense_vector", result[0])
                self.assertIn("sparse_vector", result[0])
                self.assertEqual(len(result[0]["dense_vector"]), 1024)
            finally:
                embedder_module._default_embedder = old_default


# ===========================================================================
# Test Vector Store Config
# ===========================================================================

class TestVectorStoreConfig(unittest.TestCase):

    def test_default_config(self):
        from src.embedding.vector_store import VectorStoreConfig

        config = VectorStoreConfig()
        self.assertEqual(config.persist_directory, "data/vectordb")
        self.assertEqual(config.collection_name, "vn_legal_chunks")
        self.assertEqual(config.distance_metric, "cosine")

    def test_from_yaml(self):
        from src.embedding.vector_store import VectorStoreConfig

        yaml_content = {
            "vector_store": {
                "persist_directory": "data/test_vectordb",
                "collection_name": "test_collection",
                "distance_metric": "l2",
            }
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            yaml.dump(yaml_content, f)
            tmp_path = f.name

        config = VectorStoreConfig.from_yaml(tmp_path)
        self.assertEqual(config.persist_directory, "data/test_vectordb")
        self.assertEqual(config.collection_name, "test_collection")
        self.assertEqual(config.distance_metric, "l2")

        Path(tmp_path).unlink()


# ===========================================================================
# Test Vector Store (mocked ChromaDB)
# ===========================================================================

class TestChromaVectorStoreMocked(unittest.TestCase):
    """Test vector store logic với mock ChromaDB."""

    def test_flatten_metadata(self):
        from src.embedding.vector_store import ChromaVectorStore

        meta = {
            "doc_id": "31_2024_QH15",
            "article_no": 1,
            "score": 0.95,
            "is_main": True,
            "tags": ["đất đai", "nhà ở"],
            "extra": None,
        }

        flat = ChromaVectorStore._flatten_metadata(meta)

        self.assertEqual(flat["doc_id"], "31_2024_QH15")
        self.assertEqual(flat["article_no"], 1)
        self.assertEqual(flat["score"], 0.95)
        self.assertTrue(flat["is_main"])
        self.assertEqual(json.loads(flat["tags"]), ["đất đai", "nhà ở"])
        self.assertNotIn("extra", flat)  # None bị bỏ qua

    def test_upsert_persists_generated_sparse_vector(self):
        from src.embedding.vector_store import ChromaVectorStore

        mock_embedder = MagicMock(spec=BGEM3Embedder)
        mock_embedder.encode.return_value = EmbeddingResult(
            dense=[[1.0] + [0.0] * 1023],
            sparse=[{"indices": [11, 22], "values": [0.7, 0.2]}],
        )
        mock_collection = MagicMock()
        store = ChromaVectorStore(embedder=mock_embedder)
        store._collection = mock_collection

        count = store.upsert_chunks(
            [
                {
                    "chunk_id": "test_1",
                    "text": "Phạm vi điều chỉnh",
                    "embedding_text": "Luật Đất đai Phạm vi điều chỉnh",
                    "metadata": {"doc_id": "31_2024_QH15"},
                }
            ]
        )

        self.assertEqual(count, 1)
        upsert_kwargs = mock_collection.upsert.call_args.kwargs
        stored_sparse = json.loads(upsert_kwargs["metadatas"][0]["_sparse_vector"])
        self.assertEqual(stored_sparse["indices"], [11, 22])
        self.assertEqual(stored_sparse["values"], [0.7, 0.2])

    def test_l2_distance_has_bounded_similarity_score(self):
        from src.embedding.vector_store import ChromaVectorStore, VectorStoreConfig

        store = ChromaVectorStore(config=VectorStoreConfig(distance_metric="l2"))

        self.assertEqual(store._distance_to_score(0.0), 1.0)
        self.assertAlmostEqual(store._distance_to_score(3.0), 0.25)

    def test_hybrid_search_fuses_dense_and_bm25_even_with_bge_sparse(self):
        from src.embedding.vector_store import ChromaVectorStore

        mock_embedder = MagicMock(spec=BGEM3Embedder)
        mock_embedder.encode_single.return_value = EmbeddingResult(
            dense=[[1.0, 0.0]],
            sparse=[{"indices": [7], "values": [1.0]}],
        )
        mock_collection = MagicMock()
        mock_collection.count.return_value = 3
        mock_collection.get.return_value = {
            "ids": ["dense_only", "both", "sparse_only"],
            "documents": ["nhà ở", "đất đai", "đất đai đất đai"],
            "metadatas": [
                {"doc_id": "d", "_sparse_vector": json.dumps({"indices": [8], "values": [1.0]})},
                {"doc_id": "d", "_sparse_vector": json.dumps({"indices": [7], "values": [0.8]})},
                {"doc_id": "d", "_sparse_vector": json.dumps({"indices": [7], "values": [1.0]})},
            ],
        }
        store = ChromaVectorStore(embedder=mock_embedder)
        store._collection = mock_collection
        store.search_by_vector = MagicMock(
            return_value=[
                {"chunk_id": "both", "text": "Both", "metadata": {"doc_id": "d"}, "distance": 0.1, "score": 0.9},
                {"chunk_id": "dense_only", "text": "Dense", "metadata": {"doc_id": "d"}, "distance": 0.2, "score": 0.8},
            ]
        )

        results = store.hybrid_search("đất đai", top_k=3)

        self.assertEqual(results[0]["chunk_id"], "both")
        self.assertEqual(results[0]["dense_rank"], 1)
        self.assertEqual(results[0]["bm25_rank"], 2)
        self.assertNotIn("_sparse_vector", results[0]["metadata"])


# ===========================================================================
# LIVE TEST (skip mặc định — bỏ skip để chạy với model thật)
# ===========================================================================

@unittest.skipUnless(
    RUN_LIVE_TESTS,
    "Set RUN_EMBEDDING_LIVE_TESTS=1 to run the BGE-M3 live tests.",
)
class TestEmbedderLive(unittest.TestCase):
    """Test với model BGE-M3 thật. Chạy lần đầu sẽ download model."""

    @classmethod
    def setUpClass(cls):
        cls.embedder = BGEM3Embedder(
            EmbeddingConfig(
                use_dense=True,
                use_sparse=True,
                batch_size=2,
            )
        )

    def test_encode_vietnamese_text(self):
        texts = [
            "Điều 1. Phạm vi điều chỉnh của Luật Đất đai 2024",
            "Quyền sử dụng đất là gì?",
        ]
        result = self.embedder.encode(texts)

        self.assertEqual(len(result.dense), 2)
        self.assertEqual(len(result.dense[0]), 1024)
        self.assertEqual(len(result.sparse), 2)

        # Dense vectors phải khác nhau
        self.assertNotEqual(result.dense[0][:10], result.dense[1][:10])

    def test_similar_texts_have_high_cosine(self):
        """Text tương tự phải có cosine similarity cao."""
        import numpy as np

        texts = [
            "Quyền sử dụng đất ở",
            "Quyền sử dụng đất đai",
            "Thuế thu nhập doanh nghiệp",
        ]
        result = self.embedder.encode(texts)

        vec_a = np.array(result.dense[0])
        vec_b = np.array(result.dense[1])
        vec_c = np.array(result.dense[2])

        sim_ab = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
        sim_ac = np.dot(vec_a, vec_c) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_c))

        # "quyền sử dụng đất" phải gần "quyền sử dụng đất đai" hơn "thuế TNDN"
        self.assertGreater(sim_ab, sim_ac)


@unittest.skipUnless(
    RUN_LIVE_TESTS,
    "Set RUN_EMBEDDING_LIVE_TESTS=1 to run the ChromaDB live test.",
)
class TestVectorStoreLive(unittest.TestCase):
    """Test end-to-end với ChromaDB thật."""

    def test_upsert_and_search(self):
        from src.embedding.vector_store import ChromaVectorStore, VectorStoreConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = VectorStoreConfig(
                persist_directory=tmpdir,
                collection_name="test_live",
                distance_metric="cosine",
            )
            store = ChromaVectorStore(config=config)
            try:
                chunks = [
                    {
                        "chunk_id": "test_1",
                        "text": "Điều 1. Phạm vi điều chỉnh của Luật Đất đai",
                        "embedding_text": "Luật Đất đai Điều 1 Phạm vi điều chỉnh",
                        "metadata": {"doc_id": "31_2024_QH15", "article_no": 1},
                    },
                    {
                        "chunk_id": "test_2",
                        "text": "Điều 2. Đối tượng áp dụng",
                        "embedding_text": "Luật Đất đai Điều 2 Đối tượng áp dụng",
                        "metadata": {"doc_id": "31_2024_QH15", "article_no": 2},
                    },
                ]

                count = store.upsert_chunks(chunks)
                self.assertEqual(count, 2)
                self.assertEqual(store.count(), 2)

                stored = store.collection.get(
                    ids=["test_1"],
                    include=["metadatas"],
                )
                self.assertIn("_sparse_vector", stored["metadatas"][0])

                results = store.search("phạm vi điều chỉnh", top_k=2)
                self.assertEqual(len(results), 2)
                self.assertEqual(results[0]["chunk_id"], "test_1")
                self.assertGreater(results[0]["score"], 0.5)

                hybrid_results = store.hybrid_search(
                    "phạm vi điều chỉnh",
                    top_k=2,
                )
                self.assertEqual(len(hybrid_results), 2)
                self.assertEqual(hybrid_results[0]["chunk_id"], "test_1")
                self.assertIsNotNone(hybrid_results[0]["bm25_rank"])
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
