"""Retrieval → evidence → grounded answer, shared by API and CLI."""
from src.rag.generator import AnswerGenerator
from src.retrieval.retriever import LegalRetriever
from src.retrieval.query_normalizer import normalize_query


class LegalChatService:
    def __init__(self, retriever: LegalRetriever, generator: AnswerGenerator):
        self.retriever = retriever
        self.generator = generator

    def answer(self, query: str, top_k: int = 5, *, include_evidence: bool = True,
               use_reranker: bool | None = None, history: list[dict] | None = None):
        question = normalize_query(query).normalized
        retrieval_query = self.generator.rewrite_query(question, history) if history else question
        evidence = self.retriever.retrieve(retrieval_query, top_k) if use_reranker is None else self.retriever.retrieve(retrieval_query, top_k, use_reranker=use_reranker)
        result = self.generator.generate(question, evidence, history=history, retrieval_query=retrieval_query) if history else self.generator.generate(question, evidence)
        if history:
            result["retrieval_query"] = retrieval_query
        if include_evidence:
            result["evidence"] = evidence
        return result
