from dataclasses import replace
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.embedding.embedder import EmbeddingResult, EmbeddingConfig
from src.embedding.vector_store import ChromaVectorStore, VectorStoreConfig
from src.retrieval.config import GraphConfig, RerankerConfig, RetrievalConfig
from src.retrieval.legal_graph import LegalGraph, ReferenceResolver, article_id
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.retriever import LegalRetriever


SOURCES = [
    {"id": "a", "doc_number": "1/2024/QH15", "doc_type": "luat", "title": "Luật Đất đai 2024"},
    {"id": "b", "doc_number": "2/2024/NĐ-CP", "doc_type": "nghi_dinh", "title": "Nghị định B"},
]


def corpus():
    units = [
        {"unit_id": "ua1", "doc_id": "a", "article_no": 1, "section_type": "main", "article_title": "Quy định", "text": "Theo Điều 2 Nghị định số 2/2024/NĐ-CP."},
        {"unit_id": "ub2", "doc_id": "b", "article_no": 2, "section_type": "main", "article_title": "Điều kiện", "text": "Theo Điều 3 Nghị định này."},
        {"unit_id": "ub3", "doc_id": "b", "article_no": 3, "section_type": "main", "article_title": "Hồ sơ", "text": "Theo Điều 1 Luật Đất đai."},
    ]
    chunks = [{"chunk_id": unit["unit_id"], "text": unit["text"], "metadata": {
        **{k: v for k, v in unit.items() if k != "text"}, "citation": f"{unit['doc_id']}, Điều {unit['article_no']}",
        "source_url": "https://vbpl.vn/demo"}} for unit in units]
    return units, chunks


def test_graph_keeps_cross_law_citation_and_evidence():
    units, chunks = corpus()
    graph = LegalGraph.build(units, chunks, SOURCES)
    edges = list(graph.graph.get_edge_data(article_id("a", 1), article_id("b", 2)).values())
    assert edges[0]["relation"] == "cites"
    assert "2/2024/NĐ-CP" in edges[0]["evidence"]
    assert edges[0]["legal_effect_verified"] is False


def test_graph_unknown_document_never_becomes_a_local_reference():
    refs = list(ReferenceResolver(SOURCES).references("Điều 5 Nghị định 999/2024/NĐ-CP", "a"))
    assert refs[0]["target_doc"] is None
    assert refs[0]["scope"] == "unknown_document"


def test_reference_qualifiers_are_bound_to_each_article_group():
    refs = list(ReferenceResolver(SOURCES).references("Điều 1 Luật Đất đai và Điều 2 Nghị định 2/2024/NĐ-CP", "b"))
    assert [(ref["article_no"], ref["target_doc"]) for ref in refs] == [("1", "a"), ("2", "b")]


def test_multiple_articles_in_one_reference():
    refs = list(ReferenceResolver(SOURCES).references("Các Điều 1, Điều 2 và 3 Luật Đất đai", "b"))
    assert [(ref["article_no"], ref["target_doc"]) for ref in refs] == [("1", "a"), ("2", "a"), ("3", "a")]


def test_explicit_amendment_context_points_to_base_law():
    units, chunks = corpus()
    units[0]["article_title"] = "Sửa đổi Điều 2 Nghị định số 2/2024/NĐ-CP"
    graph = LegalGraph.build(units, chunks, SOURCES)
    edges = list(graph.graph.get_edge_data(article_id("a", 1), article_id("b", 2)).values())
    assert any(edge["relation"] == "amends" for edge in edges)


def test_graph_retains_target_clause_and_selects_referenced_clause():
    units, chunks = corpus()
    units[0]["text"] = "Theo khoản 7 Điều 2 Nghị định 2/2024/NĐ-CP."
    chunks[0]["text"] = units[0]["text"]
    chunks[1]["metadata"]["clause_no"] = 1
    chunks.append({"chunk_id": "ub2_clause7", "text": "Nội dung khoản 7", "metadata": {**chunks[1]["metadata"], "clause_no": 7}})
    graph = LegalGraph.build(units, chunks, SOURCES)
    hits, _ = graph.expand([chunks[0]], GraphConfig(max_hops=1, max_articles=1, chunks_per_article=1))
    assert [hit["chunk_id"] for hit in hits] == ["ub2_clause7"]
    assert hits[0]["graph_path"][0]["target_clause_no"] == 7


def test_graph_search_follows_cycles_with_budgets_and_round_trip(tmp_path):
    units, chunks = corpus()
    graph = LegalGraph.build(units, chunks, SOURCES)
    path = tmp_path / "graph.json"
    graph.save(path)
    loaded = LegalGraph.load(path)
    seed = {**chunks[0], "score": 1}
    hits, report = loaded.expand([seed], GraphConfig(max_hops=2, max_articles=2, max_chunks=2))
    assert {hit["chunk_id"] for hit in hits} == {"ub2", "ub3"}
    assert all(hit["graph_hops"] <= 2 for hit in hits)
    assert report["visited_articles"] == 3
    assert all(hit["graph_path"] for hit in hits)


def test_stale_graph_seed_does_not_expand():
    units, chunks = corpus()
    graph = LegalGraph.build(units, chunks, SOURCES)
    hits, report = graph.expand([{**chunks[0], "text": "changed"}], GraphConfig())
    assert hits == []
    assert report["stale_seed_chunks"] == ["ua1"]


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return [ord(char) for char in text]

    def decode(self, ids, **kwargs):
        return "".join(map(chr, ids))

    def num_special_tokens_to_add(self, **kwargs):
        return 4


def test_reranker_scores_tail_windows_and_reorders_without_losing_provenance(tmp_path):
    model = MagicMock()
    model.tokenizer = CharacterTokenizer()
    model.predict.side_effect = lambda pairs, **kwargs: np.array([9.0 if "đúng" in passage else -2.0 for _, passage in pairs])
    reranker = CrossEncoderReranker(RerankerConfig(max_length=128, window_overlap=16), tmp_path, model)
    hits = [{"chunk_id": "first", "text": "khác", "score": 0.9, "metadata": {}},
            {"chunk_id": "tail", "text": "x " * 200 + "đúng", "score": 0.1, "metadata": {}, "graph_path": [{"relation": "cites"}]}]
    ranked = reranker.rerank("câu hỏi", hits)
    assert ranked[0]["chunk_id"] == "tail"
    assert ranked[0]["rerank_windows"] > 1
    assert ranked[0]["retrieval_score"] == 0.1
    assert ranked[0]["graph_path"] == hits[1]["graph_path"]


def test_reranker_rejects_bad_provider_scores(tmp_path):
    model = MagicMock(tokenizer=CharacterTokenizer())
    model.predict.return_value = np.array([float("nan")])
    with pytest.raises(ValueError, match="non-finite"):
        CrossEncoderReranker(RerankerConfig(), tmp_path, model).rerank("q", [{"chunk_id": "x", "text": "text"}])


@pytest.mark.parametrize("cuda,device,dtype,expected_device,expected_dtype", [
    (True, None, "auto", "cuda", "float16"),
    (False, None, "auto", "cpu", "float32"),
    (True, "cpu", "float32", "cpu", "float32"),
])
def test_reranker_loads_cached_model_with_device_appropriate_precision(
        monkeypatch, tmp_path, cuda, device, dtype, expected_device, expected_dtype):
    torch = MagicMock()
    torch.cuda.is_available.return_value = cuda
    factory = MagicMock()
    snapshot = MagicMock(return_value=str(tmp_path / "snapshot"))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=factory))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot))
    reranker = CrossEncoderReranker(RerankerConfig(device=device, dtype=dtype), tmp_path)
    reranker.preload()
    reranker.preload()
    assert factory.call_count == 1
    assert factory.call_args.args == (str(tmp_path / "snapshot"),)
    assert snapshot.call_args.kwargs["local_files_only"] is True
    kwargs = factory.call_args.kwargs
    assert kwargs["device"] == expected_device
    assert kwargs["model_kwargs"]["dtype"] is getattr(torch, expected_dtype)
    assert kwargs["local_files_only"] is True
    assert kwargs["trust_remote_code"] is False


def test_hybrid_bm25_has_independent_recall_and_cache_invalidation():
    embedder = MagicMock()
    embedder.encode_single.return_value = EmbeddingResult(dense=[[1, 0]], sparse=[{"indices": [1], "values": [1]}])
    store = ChromaVectorStore(embedder=embedder)
    store._collection = MagicMock()
    store._collection.count.return_value = 2
    store._collection.get.return_value = {"ids": ["vector", "bm25"], "documents": ["khác", "sổ đỏ"], "metadatas": [{}, {}]}
    store.search_by_vector = MagicMock(return_value=[{"chunk_id": "vector", "text": "khác", "metadata": {}, "score": 1}])
    hits = store.hybrid_search("sổ đỏ", top_k=2)
    assert {hit["chunk_id"] for hit in hits} == {"vector", "bm25"}
    assert next(hit for hit in hits if hit["chunk_id"] == "bm25")["dense_rank"] is None
    store.hybrid_search("sổ đỏ", top_k=2)
    assert store._collection.get.call_count == 1
    store.refresh_lexical_index()
    store.hybrid_search("sổ đỏ", top_k=2)
    assert store._collection.get.call_count == 2


def test_retrieval_reranks_graph_candidates_but_keeps_exact_results_separate():
    units, chunks = corpus()
    graph = LegalGraph.build(units, chunks, SOURCES)
    store = MagicMock()
    store.hybrid_search.return_value = [{**chunks[0], "score": 1}]
    store.get_chunks.side_effect = lambda ids: [chunk for chunk in chunks if chunk["chunk_id"] in ids]
    reranker = MagicMock()
    reranker.rerank.side_effect = lambda query, hits: [{**hit, "rerank_score": 1.0} for hit in reversed(hits)]
    response = LegalRetriever(store, RetrievalConfig(), reranker=reranker, graph=graph).retrieve("Điều 1", 1)
    assert [hit["chunk_id"] for hit in response["results"]] == ["ua1"]
    assert {hit["chunk_id"] for hit in response["related_context"]} == {"ub2", "ub3"}
    assert reranker.rerank.call_count == 1
    assert len(reranker.rerank.call_args.args[1]) == 3
    assert "[S1]" in response["agent_context"]["text"]
    assert "[G1]" in response["agent_context"]["text"]


def test_graph_candidates_missing_from_vector_collection_are_excluded():
    units, chunks = corpus()
    store = MagicMock()
    store.hybrid_search.return_value = [{**chunks[0], "score": 1}]
    store.get_chunks.return_value = []
    config = RetrievalConfig(reranker=RerankerConfig(enabled=False))
    response = LegalRetriever(store, config, graph=LegalGraph.build(units, chunks, SOURCES)).retrieve("Điều 1", 1)
    assert not response["related_context"]
    assert set(response["retrieval"]["graph"]["missing_or_stale_chunks"]) == {"ub2", "ub3"}


def test_graph_only_agent_tool_never_calls_embedding_or_reranker_without_query():
    units, chunks = corpus()
    store, reranker = MagicMock(), MagicMock()
    retriever = LegalRetriever(store, reranker=reranker, graph=LegalGraph.build(units, chunks, SOURCES))
    result = retriever.graph_search("a", "1")
    assert result["graph"]["article_found"]
    assert result["collection_verified"] is False
    store.hybrid_search.assert_not_called()
    reranker.rerank.assert_not_called()


def test_graph_only_reranks_when_agent_supplies_question():
    units, chunks = corpus()
    reranker = MagicMock()
    reranker.rerank.side_effect = lambda query, hits: hits
    retriever = LegalRetriever(None, reranker=reranker, graph=LegalGraph.build(units, chunks, SOURCES))
    result = retriever.graph_search("a", "1", query="Điều kiện")
    assert result["reranked"] is True
    reranker.rerank.assert_called_once()


def test_exact_lookup_context_budget_never_slices_legal_passages():
    units, chunks = corpus()
    retriever = LegalRetriever(None, RetrievalConfig(max_context_chars=1), graph=LegalGraph.build(units, chunks, SOURCES))
    result = retriever.graph_search("a", "1")
    assert result["agent_context"]["text"] == ""
    assert result["agent_context"]["truncated"] is True


def test_real_chroma_hybrid_graph_and_cross_encoder_adapter_end_to_end(tmp_path):
    units, chunks = corpus()
    chunks[0]["text"] += " Hồ sơ giấy chứng nhận."
    units[0]["text"] = chunks[0]["text"]
    embedder = MagicMock()
    embedder.config = EmbeddingConfig(provider="openai", model_name="text-embedding-3-small", use_sparse=False, dimensions=2)
    embedder.encode.return_value = EmbeddingResult(dense=[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    embedder.encode_single.return_value = EmbeddingResult(dense=[[1.0, 0.0]])
    model = MagicMock(tokenizer=CharacterTokenizer())
    model.predict.side_effect = lambda pairs, **kwargs: np.array([2.0 if "giấy" in passage else 1.0 for _, passage in pairs])
    graph = LegalGraph.build(units, chunks, SOURCES)
    config = RetrievalConfig(dense_candidates=1, bm25_candidates=1)
    reranker = CrossEncoderReranker(config.reranker, tmp_path, model)
    with ChromaVectorStore(VectorStoreConfig(persist_directory="db", collection_name="graph_integration"),
                           embedder=embedder, project_root=tmp_path) as store:
        store.upsert_chunks(chunks)
        response = LegalRetriever(store, config, graph=graph, reranker=reranker).retrieve("giấy chứng nhận", 1)
    assert response["results"][0]["chunk_id"] == "ua1"
    assert response["results"][0]["bm25_rank"] == 1
    assert response["results"][0]["dense_rank"] == 1
    assert {hit["chunk_id"] for hit in response["related_context"]} == {"ub2", "ub3"}
    assert response["retrieval"]["rerank_candidate_count"] == 3
