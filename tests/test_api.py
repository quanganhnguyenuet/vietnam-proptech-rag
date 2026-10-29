from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from src.api.main import app
from src.retrieval.config import RetrievalConfig, RerankerConfig, GraphConfig
from src.retrieval.retriever import LegalRetriever


def test_search_api_validates_and_returns_citations():
    store = MagicMock()
    store.hybrid_search.return_value = [{"chunk_id": "x", "text": "Quy định", "metadata": {"citation": "Luật Đất đai, Điều 1", "source_url": "https://vbpl.vn/demo"}}]
    retriever = LegalRetriever(store, RetrievalConfig(reranker=RerankerConfig(enabled=False), graph=GraphConfig(enabled=False)))
    with patch("src.api.main.ChromaVectorStore.from_config", return_value=store), patch("src.api.main.LegalRetriever.from_config", return_value=retriever):
        with TestClient(app) as client:
            assert client.get("/health").status_code == 200
            assert client.post("/search", json={"query": "   "}).status_code == 422
            assert client.post("/search", json={"query": "đất", "top_k": 0}).status_code == 422
            response = client.post("/search", json={"query": "quyền sử dụng đất"})
            assert response.status_code == 200
            assert response.json()["results"][0]["metadata"]["citation"] == "Luật Đất đai, Điều 1"
    store.close.assert_called_once()


def test_graph_api_returns_offline_paths_without_embedding():
    from src.retrieval.legal_graph import LegalGraph
    from tests.test_graph_retrieval import corpus, SOURCES
    units, chunks = corpus()
    store = MagicMock()
    retriever = LegalRetriever(store, graph=LegalGraph.build(units, chunks, SOURCES))
    with patch("src.api.main.ChromaVectorStore.from_config", return_value=store), patch("src.api.main.LegalRetriever.from_config", return_value=retriever):
        with TestClient(app) as client:
            response = client.post("/graph/search", json={"doc_id": "a", "article_no": "1"})
            assert response.status_code == 200
            assert response.json()["graph"]["article_found"]
            assert response.json()["related_context"]
            assert client.post("/graph/search", json={"doc_id": "a", "article_no": "1", "max_hops": 3}).status_code == 422
    store.hybrid_search.assert_not_called()
