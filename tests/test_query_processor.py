import unittest

from src.retrieval.query_processor import (
    QueryIntent,
    QueryProcessor,
    QueryTopic,
    RetrievalStrategy,
)


class QueryProcessorTests(unittest.TestCase):
    def setUp(self):
        self.processor = QueryProcessor()

    def test_cross_law_articles_keep_their_own_document_and_clause(self):
        plan = self.processor.process("So sánh khoản 1 Điều 138 Luật Đất đai và khoản 2 Điều 160 Luật Nhà ở")
        self.assertEqual(plan.metadata_filter, {"$or": [
            {"$and": [{"doc_id": "31_2024_QH15"}, {"article_no": 138}, {"clause_no": 1}]},
            {"$and": [{"doc_id": "27_2023_QH15"}, {"article_no": 160}, {"clause_no": 2}]},
        ]})

    def test_exact_legal_reference_builds_hard_filter(self):
        plan = self.processor.process(
            "Khoản 2 Điều 138 Luật Đất đai quy định gì?"
        )

        self.assertEqual(plan.intent, QueryIntent.EXACT_LOOKUP)
        self.assertEqual(plan.entities.document_ids, ("31_2024_QH15",))
        self.assertEqual(plan.entities.article_numbers, ("138",))
        self.assertEqual(plan.entities.clause_numbers, (2,))
        self.assertEqual(
            plan.metadata_filter,
            {
                "$and": [
                    {"doc_id": "31_2024_QH15"},
                    {"article_no": 138},
                    {"clause_no": 2},
                ]
            },
        )
        self.assertEqual(plan.strategy, RetrievalStrategy.FILTERED_HYBRID)

    def test_procedure_alias_uses_soft_routing_and_expansion(self):
        plan = self.processor.process("Thủ tục làm sổ đỏ lần đầu?")

        self.assertEqual(plan.intent, QueryIntent.PROCEDURE)
        self.assertIn(QueryTopic.LAND_REGISTRATION, plan.topics)
        self.assertGreater(len(plan.expanded_queries), 1)
        self.assertEqual(plan.metadata_filter, {})
        self.assertEqual(
            plan.preferred_doc_ids,
            ("101_2024_ND_CP", "31_2024_QH15", "102_2024_ND_CP"),
        )
        self.assertEqual(plan.strategy, RetrievalStrategy.MULTI_QUERY_HYBRID)

    def test_processor_reuses_query_normalizer(self):
        raw = "  Thủ tục\n\ncấp sổ đỏ?  "
        plan = self.processor.process(raw)

        self.assertEqual(plan.original_query, raw)
        self.assertEqual(plan.normalized_query, "Thủ tục cấp sổ đỏ?")

    def test_task_intent_takes_precedence_over_citation(self):
        plan = self.processor.process(
            "Thủ tục theo Điều 5 Nghị định 101/2024/NĐ-CP là gì?"
        )

        self.assertEqual(plan.intent, QueryIntent.PROCEDURE)
        self.assertEqual(plan.entities.document_ids, ("101_2024_ND_CP",))
        self.assertEqual(plan.strategy, RetrievalStrategy.FILTERED_HYBRID)

    def test_comparison_can_have_multiple_topics(self):
        plan = self.processor.process(
            "So sánh nhà ở xã hội và kinh doanh bất động sản"
        )

        self.assertEqual(plan.intent, QueryIntent.COMPARISON)
        self.assertIn(QueryTopic.HOUSING, plan.topics)
        self.assertIn(QueryTopic.REAL_ESTATE_BUSINESS, plan.topics)
        self.assertEqual(plan.strategy, RetrievalStrategy.MULTI_QUERY_HYBRID)


if __name__ == "__main__":
    unittest.main()
