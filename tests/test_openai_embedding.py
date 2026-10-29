import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.embedding.embedder import EmbeddingConfig, OpenAIEmbedder, get_embedder
from src.embedding.vector_store import ChromaVectorStore, VectorStoreConfig


def config(**kwargs):
    return EmbeddingConfig(provider="openai", model_name="text-embedding-3-small",
                           use_sparse=False, dimensions=2, **kwargs)


def fake_embedder(**kwargs):
    client, tokenizer = MagicMock(), MagicMock()
    tokenizer.encode.side_effect = lambda text, **_: list(text)
    client.embeddings.create.side_effect = lambda **kw: SimpleNamespace(data=[
        SimpleNamespace(index=i, embedding=[float(i), 1.0]) for i in reversed(range(len(kw["input"])))
    ])
    return OpenAIEmbedder(config(**kwargs), client, tokenizer)


def test_openai_batches_and_restores_response_order():
    embedder = fake_embedder(batch_size=2)
    result = embedder.encode(["đất", "nhà", "sổ"])
    assert result.dense == [[0.0, 1.0], [1.0, 1.0], [0.0, 1.0]]
    assert result.sparse == []
    assert embedder.client.embeddings.create.call_count == 2
    assert embedder.client.embeddings.create.call_args.kwargs["dimensions"] == 2


@pytest.mark.parametrize("texts", [[""], ["   "], ["x" * 10]])
def test_openai_invalid_text_fails_before_billable_calls(texts):
    embedder = fake_embedder(max_length=5)
    with pytest.raises(ValueError):
        embedder.encode(texts)
    embedder.client.embeddings.create.assert_not_called()


def test_openai_empty_list_never_loads_client_or_tokenizer():
    assert OpenAIEmbedder(config()).encode([]).dense == []


def test_openai_missing_key_is_actionable():
    with patch.dict("os.environ", {}, clear=True), patch("dotenv.load_dotenv"):
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            _ = OpenAIEmbedder(config()).client


@pytest.mark.parametrize("override", [{"dimensions": 2000}, {"batch_size": 0}, {"use_sparse": True}])
def test_invalid_openai_settings(override):
    values = dict(provider="openai", model_name="text-embedding-3-small", use_sparse=False)
    with pytest.raises(ValueError):
        EmbeddingConfig(**{**values, **override})


def test_factory_does_not_reuse_bge_for_openai():
    root = Path(__file__).resolve().parents[1]
    assert isinstance(get_embedder(root / "src/embedding/config.openai.yaml"), OpenAIEmbedder)
    assert get_embedder(root / "src/embedding/config.yaml").config.provider == "bge_m3"


def test_store_uses_explicit_config_for_embedder(tmp_path):
    source = Path(__file__).resolve().parents[1] / "src/embedding/config.openai.yaml"
    store = ChromaVectorStore.from_config(source, tmp_path)
    assert isinstance(store.embedder, OpenAIEmbedder)
    assert store.embedding_config.provider == "openai"


def test_collection_rejects_model_mismatch_before_embedding():
    store = ChromaVectorStore(embedding_config=config())
    collection = MagicMock()
    collection.metadata = {"hnsw:space": "cosine", "embedding_fingerprint": "different:model:2"}
    store._client = MagicMock()
    store._client.get_or_create_collection.return_value = collection
    with pytest.raises(ValueError, match="model/dimensions"):
        _ = store.collection


def test_precomputed_vectors_require_matching_provenance():
    store = ChromaVectorStore(embedding_config=config())
    with pytest.raises(ValueError, match="fingerprint"):
        store.upsert_chunks([{"chunk_id": "x", "dense_vector": [1.0, 0.0]}])


def test_chroma_round_trip_with_openai_adapter(tmp_path):
    # Real ChromaDB + injected fake OpenAI transport; no API cost or model download.
    embedder = fake_embedder()
    store = ChromaVectorStore(VectorStoreConfig(persist_directory="db", collection_name="test_openai"),
                              embedder=embedder, project_root=tmp_path)
    with store:
        store.upsert_chunks([{"chunk_id": "land", "text": "quyền sử dụng đất",
                              "embedding_text": "đất", "metadata": {"doc_id": "test"}}])
        assert store.search("đất")[0]["chunk_id"] == "land"
        assert store.hybrid_search("đất")[0]["chunk_id"] == "land"
