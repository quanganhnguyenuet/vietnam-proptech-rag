"""Evidence-backed legal reference graph. Edges do not establish legal validity."""
from collections import defaultdict, deque
from hashlib import sha256
import json
from pathlib import Path
import re

import networkx as nx

from src.crawler.sources import load_sources
from src.retrieval.config import GraphConfig

NUMBER_RE = re.compile(r"\b\d{1,3}\s*/\s*\d{4}\s*/\s*(?:QH\d+|NĐ-CP|ND-CP)\b", re.I)
ARTICLE_RE = re.compile(r"\b(?:các\s+)?Điều\s+(\d+[a-zđ]?(?:(?:\s*,\s*|\s+và\s+)(?:Điều\s+)?\d+[a-zđ]?)*)\b", re.I)
AMENDMENT_RE = re.compile(r"^(?:Sửa đổi|Bổ sung|Bãi bỏ|Thay thế)\b", re.I)
EXTERNAL_MARKER_RE = re.compile(r"\b(?:Luật|Nghị định|Thông tư|Bộ luật)\b", re.I)


class GraphUnavailable(RuntimeError):
    pass


def article_id(doc_id, article_no):
    return f"article:{doc_id}:{str(article_no).lower()}"


def text_hash(text):
    return sha256(text.encode("utf-8")).hexdigest()


def _number(text):
    return re.sub(r"\s+", "", text).upper().replace("/ND-CP", "/NĐ-CP")


class ReferenceResolver:
    def __init__(self, sources):
        self.numbers = {_number(doc["doc_number"]): doc["id"] for doc in sources}
        self.aliases = {}
        for doc in sources:
            aliases = [*doc.get("aliases", [])]
            if doc["doc_type"] == "luat":
                aliases.extend([doc["title"], re.sub(r"\s+\d{4}$", "", doc["title"])])
            for alias in aliases:
                self.aliases[alias.casefold()] = doc["id"]

    def references(self, text, source_doc, *, amendment=False, target_context=None):
        text = re.sub(r"\s+", " ", text).strip()
        matches = list(ARTICLE_RE.finditer(text))
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            # A document qualifier follows the article number. Never bind every
            # number in a paragraph to every document mentioned in that paragraph.
            suffix = re.split(r"[;.!?]", text[match.end():min(end, match.end() + 220)], maxsplit=1)[0]
            explicit = NUMBER_RE.search(suffix)
            target_doc, scope = None, None
            if explicit:
                target_doc = self.numbers.get(_number(explicit.group()))
                scope = "explicit_document_number" if target_doc else "unknown_document"
            else:
                aliases = [(suffix.casefold().find(alias), -len(alias), doc_id)
                           for alias, doc_id in self.aliases.items() if alias in suffix.casefold()]
                if aliases:
                    target_doc = sorted(aliases)[0][2]
                    scope = "explicit_document_title"
                elif target_context and (not EXTERNAL_MARKER_RE.search(suffix) or re.search(r"\b(?:Luật|Nghị định)\s+này\b", suffix, re.I)):
                    target_doc, scope = target_context, "amendment_document_context"
                elif re.search(r"\b(?:Luật|Nghị định|Bộ luật)\s+này\b", suffix, re.I) and not amendment:
                    target_doc, scope = source_doc, "same_document"
                elif EXTERNAL_MARKER_RE.search(suffix) or amendment:
                    scope = "ambiguous_document"
                else:
                    target_doc, scope = source_doc, "same_document"
            labels = re.findall(r"\d+[a-zđ]?", match.group(1), re.I)
            evidence = text[max(0, match.start() - 70):min(len(text), match.end() + len(suffix))]
            clause = re.search(r"\bkhoản\s+(\d+)\s*$", text[max(0, match.start() - 50):match.start()], re.I)
            for label in labels:
                yield {"target_doc": target_doc, "article_no": label.lower(),
                       "scope": scope, "evidence": evidence,
                       "target_clause_no": int(clause.group(1)) if clause else None}


class LegalGraph:
    def __init__(self, graph, chunks, *, unresolved=None, fingerprint=None):
        self.graph = graph
        self.chunks = chunks
        self.unresolved = unresolved or []
        self.fingerprint = fingerprint
        self.by_article = defaultdict(list)
        for chunk_id, chunk in chunks.items():
            meta = chunk["metadata"]
            if meta.get("section_type") == "main" and meta.get("article_no") is not None:
                self.by_article[article_id(meta["doc_id"], meta["article_no"])].append(chunk_id)
        for values in self.by_article.values():
            values.sort(key=lambda key: (self.chunks[key]["metadata"].get("part_index", 0), key))

    @classmethod
    def build(cls, units, chunks, sources=None, article_contexts=None):
        sources = sources if sources is not None else load_sources()
        resolver = ReferenceResolver(sources)
        graph = nx.MultiDiGraph()
        chunk_map = {}
        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            if chunk_id in chunk_map:
                raise ValueError(f"Duplicate graph chunk ID: {chunk_id}")
            chunk_map[chunk_id] = {"text": chunk["text"], "metadata": chunk["metadata"],
                                   "text_hash": text_hash(chunk["text"])}
            meta = chunk["metadata"]
            doc_node = f"document:{meta['doc_id']}"
            graph.add_node(doc_node, kind="document", doc_id=meta["doc_id"], title=meta.get("doc_title"))
            chunk_node = f"chunk:{chunk_id}"
            graph.add_node(chunk_node, kind="chunk", chunk_id=chunk_id)
            if meta.get("section_type") == "main" and meta.get("article_no") is not None:
                node = article_id(meta["doc_id"], meta["article_no"])
                graph.add_node(node, kind="article", doc_id=meta["doc_id"], article_no=str(meta["article_no"]),
                               title=meta.get("article_title"), source_url=meta.get("source_url"))
                graph.add_edge(doc_node, node, key="contains", relation="contains")
                graph.add_edge(node, chunk_node, key="contains", relation="contains")
            else:
                graph.add_edge(doc_node, chunk_node, key="contains", relation="contains")
        unresolved, seen = [], set()
        for unit in units:
            if unit.get("section_type") != "main":
                continue
            node = article_id(unit["doc_id"], unit["article_no"])
            if node not in graph:
                raise ValueError(f"Graph unit has no indexed chunks: {unit['unit_id']}")
            title = unit.get("article_title", "")
            amendment = bool(AMENDMENT_RE.search(title))
            context_text = title
            target_contexts = {resolver.numbers[_number(match.group())] for match in NUMBER_RE.finditer(title)
                               if _number(match.group()) in resolver.numbers}
            if not target_contexts and article_contexts and node in article_contexts:
                context_text = article_contexts[node]
                target_contexts = {resolver.numbers[_number(match.group())] for match in NUMBER_RE.finditer(context_text)
                                   if _number(match.group()) in resolver.numbers}
            target_context = next(iter(target_contexts)) if amendment and len(target_contexts) == 1 else None
            for text, is_title in ((title, True), (unit["text"], False)):
                for ref in resolver.references(text, unit["doc_id"], amendment=amendment, target_context=target_context):
                    target = article_id(ref["target_doc"], ref["article_no"]) if ref["target_doc"] else None
                    signature = (node, target, ref["evidence"])
                    if signature in seen:
                        continue
                    seen.add(signature)
                    if target is None or target not in graph:
                        unresolved.append({"source": node, "source_unit_id": unit["unit_id"], **ref,
                                           "reason": ref["scope"] if target is None else "target_not_in_corpus"})
                        continue
                    if target == node:
                        continue
                    mutation_reference = bool(re.search(r"(?:Sửa đổi|Bổ sung|Bãi bỏ|Thay thế)[^.]{0,100}$", ref["evidence"].split("Điều", 1)[0], re.I))
                    relation = "amends" if amendment and (is_title or mutation_reference) and (ref["scope"].startswith("explicit") or target_context) else "cites"
                    key = sha256(f"{node}|{target}|{relation}|{ref['evidence']}".encode()).hexdigest()[:24]
                    graph.add_edge(node, target, key=key, relation=relation, evidence=ref["evidence"],
                                   source_unit_id=unit["unit_id"], scope=ref["scope"],
                                   target_clause_no=ref["target_clause_no"],
                                   document_context=context_text if ref["scope"] == "amendment_document_context" else "",
                                   legal_effect_verified=False)
        fingerprint = text_hash(json.dumps(chunk_map, ensure_ascii=False, sort_keys=True))
        return cls(graph, chunk_map, unresolved=unresolved, fingerprint=fingerprint)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema_version": 1, "fingerprint": self.fingerprint,
                   "graph": nx.node_link_data(self.graph, edges="edges"),
                   "chunks": self.chunks, "unresolved": self.unresolved}
        serialized = json.dumps(payload, ensure_ascii=False)
        if path.is_file() and path.read_text(encoding="utf-8") == serialized:
            return
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(serialized, encoding="utf-8")
        temporary.replace(path)

    @classmethod
    def load(cls, path):
        path = Path(path)
        if not path.is_file():
            raise GraphUnavailable(f"Legal graph missing: {path}. Run python build_graph.py first")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise GraphUnavailable("Unsupported graph version; rebuild graph")
        return cls(nx.node_link_graph(payload["graph"], edges="edges"), payload["chunks"],
                   unresolved=payload.get("unresolved"), fingerprint=payload.get("fingerprint"))

    def expand(self, seeds, config: GraphConfig, query: str = ""):
        seed_ids = {hit["chunk_id"] for hit in seeds}
        candidates, paths, visited, queue = {}, [], set(), deque()
        stale_seeds = []
        for hit in seeds[:config.seed_k]:
            saved = self.chunks.get(hit["chunk_id"])
            if saved is None or saved["text_hash"] != text_hash(hit["text"]):
                stale_seeds.append(hit["chunk_id"])
                continue
            meta = saved["metadata"]
            if meta.get("section_type") != "main" or meta.get("article_no") is None:
                continue
            node = article_id(meta["doc_id"], meta["article_no"])
            if node not in visited:
                visited.add(node)
                queue.append((node, hit["chunk_id"], []))
        discovered = 0
        while queue and discovered < config.max_articles and len(candidates) < config.max_chunks:
            node, seed_id, path = queue.popleft()
            if len(path) >= config.max_hops:
                continue
            edges = []
            for left, right, _, data in self.graph.out_edges(node, keys=True, data=True):
                if data.get("relation") in {"cites", "amends"}:
                    edges.append((right, "outgoing", left, right, data))
            for left, right, _, data in self.graph.in_edges(node, keys=True, data=True):
                if data.get("relation") in {"cites", "amends"}:
                    edges.append((left, "incoming", left, right, data))
            edges.sort(key=lambda edge: (edge[4]["relation"] != "amends", edge[1] != "outgoing", edge[0], edge[4].get("evidence", "")))
            for neighbor, direction, left, right, data in edges:
                if neighbor in visited:
                    continue
                if discovered >= config.max_articles or len(candidates) >= config.max_chunks:
                    break
                visited.add(neighbor)
                discovered += 1
                step = {"source": left, "target": right, "direction": direction,
                        **{key: data[key] for key in ("relation", "evidence", "source_unit_id", "scope", "legal_effect_verified")},
                        "document_context": data.get("document_context", "")}
                step["target_clause_no"] = data.get("target_clause_no")
                new_path = [*path, step]
                queue.append((neighbor, seed_id, new_path))
                paths.append({"seed_chunk_id": seed_id, "target_article": neighbor, "steps": new_path})
                chunk_ids = self.by_article.get(neighbor, [])
                if direction == "outgoing" and data.get("target_clause_no") is not None:
                    clause_chunks = [key for key in chunk_ids if self.chunks[key]["metadata"].get("clause_no") == data["target_clause_no"]]
                    chunk_ids = clause_chunks or chunk_ids
                elif direction == "incoming":
                    source_chunks = [key for key in chunk_ids if self.chunks[key]["metadata"].get("unit_id") == data.get("source_unit_id")]
                    chunk_ids = source_chunks or chunk_ids
                # A referenced article can contain dozens of clauses. Select
                # passages by query terms instead of always taking clauses 1–2.
                if query and chunk_ids:
                    from rank_bm25 import BM25Plus
                    tokens = lambda value: re.findall(r"\w+", value.casefold())
                    corpus = [tokens(self.chunks[key]["text"]) or [""] for key in chunk_ids]
                    scores = BM25Plus(corpus).get_scores(tokens(query))
                    chunk_ids = [key for _, key in sorted(zip(scores, chunk_ids), key=lambda item: (-item[0], item[1]))]
                for chunk_id in chunk_ids[:config.chunks_per_article]:
                    if chunk_id in seed_ids or chunk_id in candidates:
                        continue
                    if len(candidates) >= config.max_chunks:
                        break
                    candidates[chunk_id] = {"chunk_id": chunk_id, **self.chunks[chunk_id],
                                            "origin": "graph", "graph_seed_chunk_id": seed_id,
                                            "graph_hops": len(new_path), "graph_path": new_path,
                                            "score": 1.0 / (60 + len(new_path))}
        return list(candidates.values()), {"paths": paths, "visited_articles": len(visited),
                                           "stale_seed_chunks": stale_seeds,
                                           "unresolved_reference_count": len(self.unresolved)}


def build_graph_artifact(structured_dir, chunks_dir, output_path):
    units = [row for path in sorted(Path(structured_dir).glob("*_legal_units.json"))
             for row in json.loads(path.read_text(encoding="utf-8-sig"))]
    chunks = [row for path in sorted(Path(chunks_dir).glob("*_chunks.json"))
              for row in json.loads(path.read_text(encoding="utf-8-sig"))]
    if not units or not chunks:
        raise ValueError("Cannot build legal graph from an empty corpus")
    contexts = {}
    from src.legal_structure import article_matches
    for path in sorted((Path(structured_dir).parent / "processed").glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8-sig"))
        text = doc.get("main_text", "")
        chapters = list(re.finditer(r"(?im)^#[ \t]+Chương\b[^\n]*", text))
        articles = article_matches(text)
        for index, chapter in enumerate(chapters):
            end = chapters[index + 1].start() if index + 1 < len(chapters) else len(text)
            members = [match for match in articles if chapter.start() < match.start() < end]
            if not members:
                continue
            heading = text[chapter.start():members[0].start()].strip()
            if NUMBER_RE.search(heading):
                for match in members:
                    contexts[article_id(doc["doc_id"], match.group(1))] = heading
    graph = LegalGraph.build(units, chunks, article_contexts=contexts)
    graph.save(output_path)
    return {"nodes": graph.graph.number_of_nodes(), "edges": graph.graph.number_of_edges(),
            "chunks": len(graph.chunks), "unresolved_references": len(graph.unresolved),
            "fingerprint": graph.fingerprint, "path": str(output_path)}
