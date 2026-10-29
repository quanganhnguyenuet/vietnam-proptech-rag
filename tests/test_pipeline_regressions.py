import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src import chunking
from src.crawler.clean_dataset import clean_html
from src.crawler.sources import load_sources
from src.embedding.embedder import EmbeddingResult
from src.embedding.vector_store import ChromaVectorStore
from src.quality_gate import QualityGateError, check_units, run_quality_gate
from src.retrieval.query_processor import process_query
from src.retrieval.retriever import LegalRetriever
from src.retrieval.config import RetrievalConfig, RerankerConfig, GraphConfig
from tests.test_chunking import sample_doc


def test_quoted_amendment_is_not_a_new_article_or_clause():
    doc = sample_doc(clean_text='Điều 1. Sửa đổi\n1. Sửa như sau:\n“Điều 12. Quy định mới\n1. Nội dung mới.\n2. Nội dung khác.”\n2. Thi hành.\nĐiều 2. Hiệu lực\nCó hiệu lực.')
    units = chunking.build_legal_units(doc)
    assert [(u["article_no"], u.get("clause_no")) for u in units] == [(1, 1), (1, 2), (2, None)]
    assert "Điều 12" in units[0]["text"]
    assert check_units(Path("demo.json"), units) == []


def test_unbalanced_pdf_quotes_do_not_swallow_following_articles():
    doc = sample_doc(clean_text='Điều 1. Sửa đổi\n1. “Nội dung có “dấu mở thừa.\n2. Nội dung tiếp theo.”\nĐiều 2. Hiệu lực\nÁp dụng.\nDẫn chiếu khoản 1\nĐiều 65.\nNội dung tiếp tục.')
    units = chunking.build_legal_units(doc)
    assert [unit["article_no"] for unit in units] == [1, 2]
    assert units[0]["level"] == "article"
    assert units[0]["parse_warning"] == "unbalanced_quotes"
    assert "Điều 65." in units[-1]["text"]


def test_indented_clauses_and_unsplittable_heading_body():
    doc = sample_doc(clean_text="Điều 1. Quy định\n  1. A.\n  2. B.\nĐiều 2. Có hiệu lực ngay.")
    units = chunking.build_legal_units(doc)
    assert len(units) == 3
    assert units[-1]["text"] == "Có hiệu lực ngay."


def test_nonsequential_clauses_stay_together():
    doc = sample_doc(clean_text="Điều 1. Sửa đổi\n1. A\n3. B")
    units = chunking.build_legal_units(doc)
    assert len(units) == 1
    assert units[0]["parse_warning"] == "non_sequential_clauses"


def test_long_text_keeps_words_and_part_metadata():
    text = "quyền sử dụng đất " * 400
    parts = chunking.split_long_paragraph(text)
    assert all(len(part) <= chunking.MAX_CHARS for part in parts)
    assert " ".join(parts).split() == text.split()
    unit = chunking.build_legal_units(sample_doc())[0]
    chunks = chunking.chunk_unit({**unit, "text": text})
    assert [c["metadata"]["part_index"] for c in chunks] == list(range(1, len(chunks) + 1))


def test_html_cleaning_preserves_breaks_without_scripts_or_duplicate_lists():
    result = clean_html('<p>Điều 1. Quy định<br/>1. Nội dung.</p><li><p>2. Tiếp theo.</p></li><script>bad</script>')
    assert "\n1. Nội dung." in result
    assert result.count("2. Tiếp theo.") == 1
    assert "bad" not in result


def test_quality_gate_rejects_empty_pipeline(tmp_path):
    with pytest.raises(QualityGateError):
        run_quality_gate(tmp_path / "p", tmp_path / "u", tmp_path / "c")


def test_quality_gate_rejects_missing_unit_chunks(tmp_path):
    for directory in ("p", "u", "c"):
        (tmp_path / directory).mkdir()
    doc = sample_doc()
    units = chunking.build_legal_units(doc)
    chunks = chunking.build_chunks(units)
    (tmp_path / "p/doc.json").write_text(json.dumps(doc), encoding="utf-8")
    (tmp_path / "u/doc_legal_units.json").write_text(json.dumps(units), encoding="utf-8")
    (tmp_path / "c/doc_chunks.json").write_text(json.dumps(chunks[1:]), encoding="utf-8")
    with pytest.raises(QualityGateError):
        run_quality_gate(tmp_path / "p", tmp_path / "u", tmp_path / "c")


def test_registry_documents_are_recognized_with_spaced_numbers():
    assert len(load_sources()) == 13
    plan = process_query("Điều 5 Nghị định 95 / 2024 / NĐ-CP")
    assert plan.entities.document_ids == ("95_2024_ND_CP",)
    assert {"doc_id": "95_2024_ND_CP"} in plan.metadata_filter["$and"]


def test_unknown_number_is_filtered_instead_of_searching_unrelated_laws():
    assert process_query("999/2026/NĐ-CP").metadata_filter == {"doc_number": "999/2026/NĐ-CP"}


def test_openai_hybrid_combines_lexical_and_dense_rankings():
    embedder = MagicMock()
    embedder.encode_single.return_value = EmbeddingResult(dense=[[1.0, 0.0]])
    store = ChromaVectorStore(embedder=embedder)
    store._collection = MagicMock()
    store._collection.count.return_value = 2
    store._collection.get.return_value = {"ids": ["wrong", "right"], "documents": ["quyền sở hữu nhà", "sổ đỏ đất"], "metadatas": [{}, {}]}
    store.search_by_vector = MagicMock(return_value=[{"chunk_id": "wrong", "metadata": {}, "text": "nhà"}])
    results = store.hybrid_search("sổ đỏ", top_k=2)
    assert {hit["chunk_id"] for hit in results} == {"wrong", "right"}


def test_retriever_preserves_hard_filters_and_deduplicates_expansions():
    store = MagicMock()
    store.hybrid_search.return_value = [{"chunk_id": "x", "metadata": {}, "text": "đất", "score": 1}]
    config = RetrievalConfig(reranker=RerankerConfig(enabled=False), graph=GraphConfig(enabled=False))
    result = LegalRetriever(store, config).retrieve("Thủ tục sổ đỏ theo Điều 5 Nghị định 101/2024/NĐ-CP", 3)
    assert len(result["results"]) == 1
    assert store.hybrid_search.call_count > 1
    assert store.hybrid_search.call_args.kwargs["where"] == result["plan"]["metadata_filter"]
