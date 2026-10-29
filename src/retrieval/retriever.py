"""Execute the existing query plan against the selected embedding collection."""
import json
from dataclasses import asdict, replace
from pathlib import Path

from src.embedding.vector_store import ChromaVectorStore
from src.retrieval.query_processor import process_query
from src.retrieval.config import RetrievalConfig, DEFAULT_RETRIEVAL_CONFIG
from src.retrieval.legal_graph import LegalGraph, text_hash
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.query_normalizer import normalize_query
from src.retrieval.legal_graph import article_id


class LegalRetriever:
    def __init__(self, store: ChromaVectorStore, config: RetrievalConfig | None = None,
                 *, project_root: Path | None = None, reranker=None, graph=None):
        self.store = store
        self.config = config or RetrievalConfig()
        self.project_root = project_root or Path(__file__).resolve().parents[2]
        self._reranker = reranker
        self._graph = graph

    @classmethod
    def from_config(cls, store, path=DEFAULT_RETRIEVAL_CONFIG, project_root=None):
        return cls(store, RetrievalConfig.from_yaml(path), project_root=project_root)

    @property
    def graph(self):
        if self._graph is None:
            self._graph = LegalGraph.load(self.project_root / self.config.graph.path)
        return self._graph

    @property
    def reranker(self):
        if self._reranker is None:
            self._reranker = CrossEncoderReranker(self.config.reranker, self.project_root)
        return self._reranker

    def retrieve(self, query: str, top_k: int | None = None, *, use_reranker: bool | None = None) -> dict:
        plan = process_query(query)
        limit = top_k if top_k is not None else plan.top_k
        if not 1 <= limit <= 100:
            raise ValueError("top_k must be between 1 and 100")
        candidates = max(self.config.candidate_pool, plan.candidate_k, limit)
        rankings = []
        for expanded in plan.expanded_queries:
            rankings.append(self.store.hybrid_search(
                expanded, top_k=candidates, where=plan.metadata_filter or None,
                dense_candidates=max(limit, self.config.dense_candidates),
                bm25_candidates=max(limit, self.config.bm25_candidates), rrf_k=self.config.rrf_k,
                dense_weight=self.config.dense_weight, bm25_weight=self.config.bm25_weight,
            ))
        merged = {}
        for ranking in rankings:
            for rank, hit in enumerate(ranking, 1):
                metadata = hit.get("metadata") or {}
                if plan.entities.point_labels:
                    labels = metadata.get("point_labels", [])
                    if isinstance(labels, str):
                        labels = json.loads(labels)
                    if labels and not set(labels).intersection(plan.entities.point_labels):
                        continue
                row = merged.setdefault(hit["chunk_id"], {**hit, "origin": "hybrid", "score": 0.0})
                route_weight = 1.1 if metadata.get("doc_id") in plan.preferred_doc_ids else 1.0
                row["score"] += route_weight / (60 + rank)
        primary = sorted(merged.values(), key=lambda hit: (-hit["score"], hit["chunk_id"]))[:candidates]
        related, graph_diagnostics = [], {"enabled": self.config.graph.enabled}
        if self.config.graph.enabled and primary:
            expanded, diagnostics = self.graph.expand(primary, self.config.graph, query=plan.normalized_query)
            graph_diagnostics.update(diagnostics)
            # Graph artifacts may predate the selected vector collection. Only
            # return nodes whose content still matches the stored corpus.
            stored = {hit["chunk_id"]: hit for hit in self.store.get_chunks([hit["chunk_id"] for hit in expanded])}
            skipped = []
            primary_ids = {hit["chunk_id"] for hit in primary}
            for hit in expanded:
                current = stored.get(hit["chunk_id"])
                if current is None or text_hash(current["text"]) != hit["text_hash"]:
                    skipped.append(hit["chunk_id"])
                elif hit["chunk_id"] not in primary_ids:
                    related.append({**{k: v for k, v in hit.items() if k != "text_hash"}, **current})
            graph_diagnostics["missing_or_stale_chunks"] = skipped
            graph_diagnostics["candidate_count"] = len(related)
        pool = [*primary, *related]
        reranker_enabled = self.config.reranker.enabled if use_reranker is None else use_reranker
        ranked = self.reranker.rerank(plan.normalized_query, pool) if reranker_enabled and pool else pool
        results = [hit for hit in ranked if hit["origin"] == "hybrid"][:limit]
        related_context = [hit for hit in ranked if hit["origin"] == "graph"][:self.config.graph.context_k]
        context = self._agent_context(results, related_context)
        return {"plan": asdict(plan), "results": results, "related_context": related_context,
                "agent_context": context,
                "retrieval": {"method": "bm25_vector_rrf", "hybrid_candidate_count": len(primary),
                              "reranker_enabled": reranker_enabled,
                              "reranker_model": self.config.reranker.model_name if reranker_enabled else None,
                              "rerank_candidate_count": len(pool), "graph": graph_diagnostics}}

    def _agent_context(self, primary, related):
        blocks, sources, used = [], [], 0
        truncated = False
        for prefix, hits in (("S", primary), ("G", related)):
            for index, hit in enumerate(hits, 1):
                label = f"{prefix}{index}"
                meta = hit.get("metadata") or {}
                block = f"[{label}] {meta.get('citation', hit['chunk_id'])}\nNguồn: {meta.get('source_url', '')}\n{hit['text']}"
                if used + len(block) + 2 > self.config.max_context_chars:
                    truncated = True
                    continue
                blocks.append(block)
                sources.append({"label": label, "chunk_id": hit["chunk_id"], "origin": hit["origin"],
                                "citation": meta.get("citation"), "source_url": meta.get("source_url"),
                                "graph_path": hit.get("graph_path", [])})
                used += len(block) + 2
        return {"text": "\n\n".join(blocks), "sources": sources, "truncated": truncated,
                "legal_effect_verified": False}

    def graph_search(self, doc_id: str, article_no: str | int, *, max_hops: int = 2,
                     query: str | None = None):
        """Agent tool for graph-only navigation; no embedding API is called.

        This browses the graph artifact corpus, independently of the selected
        vector collection. /search additionally checks expanded chunks in Chroma.
        """
        if not 1 <= max_hops <= self.config.graph.max_hops:
            raise ValueError(f"max_hops must be between 1 and {self.config.graph.max_hops}")
        node = article_id(doc_id, article_no)
        keys = self.graph.by_article.get(node, [])
        if not keys:
            return {"results": [], "related_context": [], "graph": {"article_found": False},
                    "collection_verified": False}
        seeds = [{"chunk_id": key, "text": self.graph.chunks[key]["text"],
                  "metadata": self.graph.chunks[key]["metadata"], "origin": "graph_seed", "score": 1.0}
                 for key in keys[:self.config.graph.chunks_per_article]]
        cleaned_query = normalize_query(query).normalized if query is not None else ""
        related, diagnostics = self.graph.expand(seeds, replace(self.config.graph, max_hops=max_hops), query=cleaned_query)
        related = [{k: v for k, v in hit.items() if k != "text_hash"} for hit in related]
        ranked = [*seeds, *related]
        if cleaned_query and self.config.reranker.enabled:
            ranked = self.reranker.rerank(cleaned_query, ranked)
        primary = [hit for hit in ranked if hit["origin"] == "graph_seed"]
        context = [hit for hit in ranked if hit["origin"] == "graph"][:self.config.graph.context_k]
        return {"results": primary, "related_context": context,
                "agent_context": self._agent_context(primary, context),
                "graph": {**diagnostics, "article_found": True, "fingerprint": self.graph.fingerprint},
                "collection_verified": False, "reranked": bool(cleaned_query and self.config.reranker.enabled)}
