"""BM25 + dense retrieval with rank fusion; independent of embedding provider."""
import json
from collections import OrderedDict

from rank_bm25 import BM25Plus


def hybrid_search(store, query, *, top_k, where, dense_candidates, bm25_candidates,
                  rrf_k, dense_weight, bm25_weight):
    if top_k <= 0 or store.count() == 0:
        return []
    if min(rrf_k, dense_candidates, bm25_candidates) <= 0 or min(dense_weight, bm25_weight) <= 0:
        raise ValueError("Both retrieval branches require positive budgets and weights")
    result = store.embedder.encode_single(query)
    if not result.dense:
        raise ValueError("Dense embedding is required for hybrid search")
    dense = store.search_by_vector(result.dense[0], top_k=dense_candidates, where=where)
    cache = getattr(store, "_bm25_cache", None)
    if cache is None:
        store._bm25_cache = cache = OrderedDict()
    key = (store.count(), json.dumps(where or {}, sort_keys=True))
    if key not in cache:
        kwargs = {"include": ["documents", "metadatas"]}
        if where:
            kwargs["where"] = where
        rows = store.collection.get(**kwargs)
        corpus = []
        for index, text in enumerate(rows["documents"] or []):
            metadata = (rows["metadatas"][index] or {}) if rows["metadatas"] else {}
            context = " ".join(str(metadata.get(k, "")) for k in ("doc_title", "doc_number", "citation"))
            corpus.append(store._lexical_tokens(context + " " + (text or "")) or [""])
        cache[key] = (rows, corpus, BM25Plus(corpus) if corpus else None)
        if len(cache) > 8:
            cache.popitem(last=False)
    else:
        cache.move_to_end(key)
    rows, corpus, index = cache[key]
    tokens = store._lexical_tokens(query)
    lexical = []
    if index is not None and tokens:
        scores = index.get_scores(tokens)
        terms = set(tokens)
        for i, chunk_id in enumerate(rows["ids"]):
            if not terms.intersection(corpus[i]):
                continue
            lexical.append({"chunk_id": chunk_id, "text": rows["documents"][i] or "",
                            "metadata": store._public_metadata(rows["metadatas"][i]) if rows["metadatas"] else {},
                            "distance": None, "bm25_score": float(scores[i])})
    lexical.sort(key=lambda hit: (-hit["bm25_score"], hit["chunk_id"]))
    lexical = lexical[:bm25_candidates]
    fused = {}
    for rank, hit in enumerate(dense, 1):
        fused[hit["chunk_id"]] = {**hit, "dense_rank": rank, "bm25_rank": None,
                                   "bm25_score": None, "score": dense_weight / (rrf_k + rank)}
    for rank, hit in enumerate(lexical, 1):
        item = fused.setdefault(hit["chunk_id"], {**hit, "dense_rank": None, "score": 0.0})
        item.update(bm25_rank=rank, bm25_score=hit["bm25_score"])
        item["score"] += bm25_weight / (rrf_k + rank)
    return sorted(fused.values(), key=lambda hit: (-hit["score"], hit["chunk_id"]))[:top_k]
