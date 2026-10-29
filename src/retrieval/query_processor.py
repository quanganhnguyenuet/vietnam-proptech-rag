"""Build an explainable retrieval plan from a user's legal question.

This module deliberately contains no database, embedding, or LLM calls.  Its
job is to turn a raw query into a deterministic ``QueryPlan`` that a retriever
can execute later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

from src.retrieval.query_normalizer import normalize_query
from src.crawler.sources import load_sources
from src.retrieval.legal_graph import ReferenceResolver


class QueryIntent(str, Enum):
    EXACT_LOOKUP = "exact_lookup"
    PROCEDURE = "procedure"
    ELIGIBILITY = "eligibility"
    DEFINITION = "definition"
    COMPARISON = "comparison"
    GENERAL = "general"


class QueryTopic(str, Enum):
    LAND = "land"
    LAND_REGISTRATION = "land_registration"
    HOUSING = "housing"
    REAL_ESTATE_BUSINESS = "real_estate_business"
    GENERAL = "general"


class RetrievalStrategy(str, Enum):
    HYBRID = "hybrid"
    FILTERED_HYBRID = "filtered_hybrid"
    MULTI_QUERY_HYBRID = "multi_query_hybrid"


@dataclass(frozen=True)
class QueryEntities:
    """Entities that are safe to carry into retrieval planning."""

    document_ids: tuple[str, ...] = ()
    document_numbers: tuple[str, ...] = ()
    article_numbers: tuple[str, ...] = ()
    clause_numbers: tuple[int, ...] = ()
    point_labels: tuple[str, ...] = ()
    section_type: str | None = None
    canonical_terms: tuple[str, ...] = ()
    article_references: tuple[tuple[str, str, int | None], ...] = ()


@dataclass(frozen=True)
class QueryPlan:
    """Serializable instructions for the retrieval layer."""

    original_query: str
    normalized_query: str
    intent: QueryIntent
    topics: tuple[QueryTopic, ...]
    entities: QueryEntities
    expanded_queries: tuple[str, ...]
    preferred_doc_ids: tuple[str, ...]
    metadata_filter: dict[str, Any]
    strategy: RetrievalStrategy
    top_k: int
    candidate_k: int
    reasons: tuple[str, ...] = field(default_factory=tuple)


ARTICLE_RE = re.compile(r"\bđiều\s+(\d+[a-z]?)\b", re.IGNORECASE)
CLAUSE_RE = re.compile(r"\bkhoản\s+(\d+)\b", re.IGNORECASE)
POINT_RE = re.compile(r"\bđiểm\s+([a-zđ])\b", re.IGNORECASE)
DOCUMENT_NUMBER_RE = re.compile(
    r"\b\d{1,3}\s*/\s*\d{4}\s*/\s*(?:QH\d+|NĐ-CP|ND-CP)\b",
    re.IGNORECASE,
)


DOCUMENT_ALIASES: dict[str, str] = {
    "luật đất đai 2024": "31_2024_QH15",
    "luật đất đai": "31_2024_QH15",
    "31/2024/qh15": "31_2024_QH15",
    "luật nhà ở 2023": "27_2023_QH15",
    "luật nhà ở": "27_2023_QH15",
    "27/2023/qh15": "27_2023_QH15",
    "luật kinh doanh bất động sản 2023": "29_2023_QH15",
    "luật kinh doanh bất động sản": "29_2023_QH15",
    "29/2023/qh15": "29_2023_QH15",
    "nghị định 101/2024/nđ-cp": "101_2024_ND_CP",
    "101/2024/nđ-cp": "101_2024_ND_CP",
    "101/2024/nd-cp": "101_2024_ND_CP",
    "nghị định 101": "101_2024_ND_CP",
    "nghị định 102/2024/nđ-cp": "102_2024_ND_CP",
    "102/2024/nđ-cp": "102_2024_ND_CP",
    "102/2024/nd-cp": "102_2024_ND_CP",
    "nghị định 102": "102_2024_ND_CP",
}

DOCUMENT_NUMBERS: dict[str, str] = {
    "31_2024_QH15": "31/2024/QH15",
    "27_2023_QH15": "27/2023/QH15",
    "29_2023_QH15": "29/2023/QH15",
    "101_2024_ND_CP": "101/2024/NĐ-CP",
    "102_2024_ND_CP": "102/2024/NĐ-CP",
}
for _doc in load_sources():
    DOCUMENT_NUMBERS[_doc["id"]] = _doc["doc_number"]
    for _alias in (_doc["doc_number"], _doc["doc_number"].replace("NĐ", "ND"), *_doc.get("aliases", [])):
        DOCUMENT_ALIASES[_alias.casefold()] = _doc["id"]

INTENT_KEYWORDS: dict[QueryIntent, tuple[str, ...]] = {
    QueryIntent.COMPARISON: (
        "so sánh",
        "khác nhau",
        "phân biệt",
        "giống nhau",
    ),
    QueryIntent.PROCEDURE: (
        "thủ tục",
        "hồ sơ",
        "trình tự",
        "nộp ở đâu",
        "cơ quan nào",
        "bao lâu",
    ),
    QueryIntent.ELIGIBILITY: (
        "điều kiện",
        "được phép",
        "có được",
        "đối tượng nào",
        "trường hợp nào",
    ),
    QueryIntent.DEFINITION: (
        "là gì",
        "được hiểu là",
        "khái niệm",
        "định nghĩa",
    ),
}

TOPIC_KEYWORDS: dict[QueryTopic, tuple[str, ...]] = {
    QueryTopic.LAND_REGISTRATION: (
        "sổ đỏ",
        "sổ hồng",
        "giấy chứng nhận",
        "đăng ký đất đai",
        "cấp giấy",
        "sang tên",
        "đăng ký biến động",
    ),
    QueryTopic.REAL_ESTATE_BUSINESS: (
        "kinh doanh bất động sản",
        "môi giới",
        "chủ đầu tư",
        "đặt cọc",
        "bất động sản hình thành trong tương lai",
        "mua nhà trên giấy",
    ),
    QueryTopic.HOUSING: (
        "nhà ở",
        "chung cư",
        "nhà ở xã hội",
        "quyền sở hữu nhà",
        "nhà ở hình thành trong tương lai",
    ),
    QueryTopic.LAND: (
        "đất đai",
        "quyền sử dụng đất",
        "thu hồi đất",
        "bồi thường",
        "tái định cư",
        "chuyển mục đích sử dụng đất",
        "tách thửa",
    ),
}

TOPIC_DOCUMENT_ROUTES: dict[QueryTopic, tuple[str, ...]] = {
    QueryTopic.LAND: ("31_2024_QH15", "102_2024_ND_CP"),
    QueryTopic.LAND_REGISTRATION: (
        "101_2024_ND_CP",
        "31_2024_QH15",
        "102_2024_ND_CP",
    ),
    QueryTopic.HOUSING: ("27_2023_QH15", "95_2024_ND_CP"),
    QueryTopic.REAL_ESTATE_BUSINESS: ("29_2023_QH15", "96_2024_ND_CP"),
    QueryTopic.GENERAL: (),
}

QUERY_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "làm sổ đỏ": (
        "đăng ký cấp giấy chứng nhận quyền sử dụng đất",
    ),
    "sổ đỏ": (
        "giấy chứng nhận quyền sử dụng đất",
    ),
    "làm sổ hồng": (
        "đăng ký cấp giấy chứng nhận quyền sử dụng đất quyền sở hữu tài sản gắn liền với đất",
    ),
    "sổ hồng": (
        "giấy chứng nhận quyền sử dụng đất quyền sở hữu tài sản gắn liền với đất",
    ),
    "mua nhà trên giấy": (
        "mua bán nhà ở hình thành trong tương lai",
    ),
    "sang tên": (
        "đăng ký biến động quyền sử dụng đất",
    ),
}


def _unique(values: Iterable[Any]) -> tuple[Any, ...]:
    """Return values in first-seen order without duplicates."""

    return tuple(dict.fromkeys(values))


class QueryProcessor:
    """Convert a raw Vietnamese legal query into a retrieval plan."""

    def process(self, raw_query: str) -> QueryPlan:
        cleaned = normalize_query(raw_query)
        query = cleaned.normalized

        intent, intent_reasons = self._classify_intent(query)
        topics, topic_reasons = self._classify_topics(query)
        entities, entity_reasons = self._extract_entities(query)
        expanded_queries, expansion_reasons = self._expand_query(query)
        preferred_doc_ids, routing_reasons = self._route_documents(
            topics,
            entities,
        )
        metadata_filter = self._build_metadata_filter(entities)
        strategy, top_k, candidate_k, strategy_reasons = (
            self._select_retrieval_strategy(
                intent=intent,
                entities=entities,
                expanded_queries=expanded_queries,
                metadata_filter=metadata_filter,
            )
        )

        return QueryPlan(
            original_query=cleaned.original,
            normalized_query=query,
            intent=intent,
            topics=topics,
            entities=entities,
            expanded_queries=expanded_queries,
            preferred_doc_ids=preferred_doc_ids,
            metadata_filter=metadata_filter,
            strategy=strategy,
            top_k=top_k,
            candidate_k=candidate_k,
            reasons=(
                *intent_reasons,
                *topic_reasons,
                *entity_reasons,
                *expansion_reasons,
                *routing_reasons,
                *strategy_reasons,
            ),
        )

    @staticmethod
    def _classify_intent(
        query: str,
    ) -> tuple[QueryIntent, tuple[str, ...]]:
        folded = query.casefold()

        # A task-oriented intent is more informative than the citation that may
        # accompany it.  For example, "thủ tục theo Điều 5" is PROCEDURE.
        for intent, keywords in INTENT_KEYWORDS.items():
            matched = tuple(keyword for keyword in keywords if keyword in folded)
            if matched:
                return intent, (f"Matched {intent.value} keyword: {matched[0]}",)

        has_legal_reference = bool(
            ARTICLE_RE.search(query)
            or CLAUSE_RE.search(query)
            or DOCUMENT_NUMBER_RE.search(query)
            or any(alias in folded for alias in DOCUMENT_ALIASES)
        )
        if has_legal_reference:
            return QueryIntent.EXACT_LOOKUP, ("Detected an explicit legal reference",)

        return QueryIntent.GENERAL, ("No specialized intent rule matched",)

    @staticmethod
    def _classify_topics(
        query: str,
    ) -> tuple[tuple[QueryTopic, ...], tuple[str, ...]]:
        folded = query.casefold()
        topics: list[QueryTopic] = []
        reasons: list[str] = []

        for topic, keywords in TOPIC_KEYWORDS.items():
            matched = tuple(keyword for keyword in keywords if keyword in folded)
            if matched:
                topics.append(topic)
                reasons.append(f"Matched {topic.value} topic: {matched[0]}")

        if not topics:
            return (QueryTopic.GENERAL,), ("No specialized topic rule matched",)

        return tuple(topics), tuple(reasons)

    @staticmethod
    def _extract_entities(
        query: str,
    ) -> tuple[QueryEntities, tuple[str, ...]]:
        folded = query.casefold()
        document_ids = _unique(
            doc_id
            for alias, doc_id in DOCUMENT_ALIASES.items()
            if re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", folded)
        )

        known_numbers = [DOCUMENT_NUMBERS[doc_id] for doc_id in document_ids]
        raw_numbers = (
            match.group(0).replace(" ", "").upper().replace("/ND-CP", "/NĐ-CP")
            for match in DOCUMENT_NUMBER_RE.finditer(query)
        )
        document_numbers = _unique((*known_numbers, *raw_numbers))
        by_number = {number.casefold(): doc_id for doc_id, number in DOCUMENT_NUMBERS.items()}
        document_ids = _unique((*document_ids, *(by_number[number.casefold()] for number in document_numbers if number.casefold() in by_number)))

        article_numbers = _unique(
            match.group(1).lower() for match in ARTICLE_RE.finditer(query)
        )
        clause_numbers = _unique(
            int(match.group(1)) for match in CLAUSE_RE.finditer(query)
        )
        point_labels = _unique(
            match.group(1).lower() for match in POINT_RE.finditer(query)
        )

        section_type = None
        if "phụ lục" in folded or "mẫu số" in folded:
            section_type = "appendix"

        canonical_terms = _unique(
            canonical
            for alias, expansions in QUERY_EXPANSIONS.items()
            if alias in folded
            for canonical in expansions
        )

        entities = QueryEntities(
            document_ids=document_ids,
            document_numbers=document_numbers,
            article_numbers=article_numbers,
            clause_numbers=clause_numbers,
            point_labels=point_labels,
            section_type=section_type,
            canonical_terms=canonical_terms,
            article_references=_unique(
                (ref["target_doc"], ref["article_no"], ref["target_clause_no"])
                for ref in ReferenceResolver(load_sources()).references(query, "__query__")
                if ref["target_doc"] and ref["target_doc"] != "__query__"
            ),
        )

        reasons: list[str] = []
        if document_ids:
            reasons.append(f"Extracted document IDs: {', '.join(document_ids)}")
        if article_numbers:
            reasons.append(f"Extracted article: {', '.join(article_numbers)}")
        if clause_numbers:
            reasons.append(
                "Extracted clause: " + ", ".join(map(str, clause_numbers))
            )
        if point_labels:
            reasons.append(f"Extracted point: {', '.join(point_labels)}")
        if section_type:
            reasons.append(f"Extracted section type: {section_type}")

        return entities, tuple(reasons)

    @staticmethod
    def _expand_query(query: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
        folded = query.casefold()
        queries = [query]
        reasons: list[str] = []

        for alias, canonical_terms in QUERY_EXPANSIONS.items():
            if alias not in folded:
                continue
            for canonical in canonical_terms:
                expanded = re.sub(
                    re.escape(alias),
                    canonical,
                    query,
                    count=1,
                    flags=re.IGNORECASE,
                )
                queries.append(expanded)
            reasons.append(f"Expanded colloquial term: {alias}")

        return _unique(queries), tuple(reasons)

    @staticmethod
    def _route_documents(
        topics: tuple[QueryTopic, ...],
        entities: QueryEntities,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        # Explicit documents take precedence and will also become hard filters.
        if entities.document_ids:
            return entities.document_ids, ("Preferred explicitly referenced documents",)

        preferred = _unique(
            doc_id
            for topic in topics
            for doc_id in TOPIC_DOCUMENT_ROUTES[topic]
        )
        if preferred:
            return preferred, ("Preferred documents inferred from query topics",)

        return (), ("No preferred document route",)

    @staticmethod
    def _build_metadata_filter(entities: QueryEntities) -> dict[str, Any]:
        conditions: list[dict[str, Any]] = []
        references = entities.article_references
        paired = (
            len(entities.document_ids) > 1 and len(entities.article_numbers) > 1
            and {ref[0] for ref in references} == set(entities.document_ids)
            and {ref[1] for ref in references} == set(entities.article_numbers)
        )
        if paired:
            branches = []
            for doc_id, article, clause in references:
                branch = [{"doc_id": doc_id}, {"article_no": int(article) if article.isdigit() else article}]
                if clause is not None:
                    branch.append({"clause_no": clause})
                branches.append({"$and": branch})
            conditions.append({"$or": branches})

        if entities.document_ids and not paired:
            value: Any
            if len(entities.document_ids) == 1:
                value = entities.document_ids[0]
            else:
                value = {"$in": list(entities.document_ids)}
            conditions.append({"doc_id": value})
        elif entities.document_numbers and not paired:
            numbers = list(entities.document_numbers)
            conditions.append({"doc_number": numbers[0] if len(numbers) == 1 else {"$in": numbers}})

        if entities.article_numbers and not paired:
            articles: list[int | str] = [
                int(number) if number.isdigit() else number
                for number in entities.article_numbers
            ]
            if len({type(article) for article in articles}) > 1:
                conditions.append({"$or": [{"article_no": article} for article in articles]})
            else:
                value = articles[0] if len(articles) == 1 else {"$in": articles}
                conditions.append({"article_no": value})

        if entities.clause_numbers and not paired:
            clauses = list(entities.clause_numbers)
            value = clauses[0] if len(clauses) == 1 else {"$in": clauses}
            conditions.append({"clause_no": value})

        if entities.section_type:
            conditions.append({"section_type": entities.section_type})

        if not conditions:
            return {}
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    @staticmethod
    def _select_retrieval_strategy(
        *,
        intent: QueryIntent,
        entities: QueryEntities,
        expanded_queries: tuple[str, ...],
        metadata_filter: dict[str, Any],
    ) -> tuple[RetrievalStrategy, int, int, tuple[str, ...]]:
        if metadata_filter and (
            entities.document_ids
            or entities.article_numbers
            or intent == QueryIntent.EXACT_LOOKUP
        ):
            return (
                RetrievalStrategy.FILTERED_HYBRID,
                5,
                20,
                ("Selected filtered hybrid retrieval for exact metadata",),
            )

        if intent == QueryIntent.COMPARISON or len(expanded_queries) > 1:
            return (
                RetrievalStrategy.MULTI_QUERY_HYBRID,
                8,
                60,
                ("Selected multi-query hybrid retrieval",),
            )

        return (
            RetrievalStrategy.HYBRID,
            6,
            40,
            ("Selected default hybrid retrieval",),
        )


def process_query(raw_query: str) -> QueryPlan:
    """Convenience entry point for callers that do not need custom rules."""

    return QueryProcessor().process(raw_query)
