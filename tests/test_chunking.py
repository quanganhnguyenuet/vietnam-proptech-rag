import json
import tempfile
import unittest
from pathlib import Path

from src import chunking
from src.quality_gate import check_processed_doc, load_rows


def sample_doc(**overrides):
    doc = {
        "doc_id": "demo_doc",
        "title": "Lu\u1eadt Demo",
        "doc_number": "01/2026/QH15",
        "doc_type": "luat",
        "source_url": "https://vbpl.vn/demo",
        "expected_article_count": 2,
        "source_provider": "tvpl",
        "source_policy": "tvpl_primary_with_attachment_preference",
        "source_format": "html",
        "downloaded_files": [],
        "clean_text": """
# Ch\u01b0\u01a1ng I

## \u0110i\u1ec1u 1. Ph\u1ea1m vi

1. Quy \u0111\u1ecbnh th\u1ee9 nh\u1ea5t.
2. Quy \u0111\u1ecbnh th\u1ee9 hai.

## \u0110i\u1ec1u 2. \u0110\u1ed1i t\u01b0\u1ee3ng

1. \u00c1p d\u1ee5ng cho t\u1ed5 ch\u1ee9c.

PH\u1ee4 L\u1ee4C

## \u0110i\u1ec1u 1. \u0110i\u1ec1u trong m\u1eabu h\u1ee3p \u0111\u1ed3ng

1. \u0110i\u1ec1u n\u00e0y kh\u00f4ng \u0111\u01b0\u1ee3c t\u00ednh l\u00e0 \u0111i\u1ec1u lu\u1eadt ch\u00ednh.
""".strip(),
    }
    doc.update(overrides)
    return doc


class ChunkingTests(unittest.TestCase):
    def test_appendix_articles_do_not_pollute_main_units(self):
        units = chunking.build_legal_units(sample_doc())

        main_articles = sorted(
            {unit["article_no"] for unit in units if unit.get("section_type") == "main"}
        )
        appendix_units = [unit for unit in units if unit.get("section_type") == "appendix"]

        self.assertEqual(main_articles, [1, 2])
        self.assertEqual(len(appendix_units), 1)
        self.assertEqual(appendix_units[0]["unit_id"], "demo_doc__appendix__appendix_1")
        self.assertEqual(len({unit["unit_id"] for unit in units}), len(units))

    def test_chunk_ids_include_section_type_and_source_metadata(self):
        units = chunking.build_legal_units(sample_doc())
        chunks = chunking.build_chunks(units)

        main_chunk = next(chunk for chunk in chunks if chunk["metadata"]["section_type"] == "main")

        self.assertTrue(main_chunk["chunk_id"].startswith("demo_doc__main__article_1__clause_1"))
        self.assertEqual(main_chunk["metadata"]["source_provider"], "tvpl")
        self.assertEqual(main_chunk["metadata"]["source_format"], "html")

    def test_trailing_structure_headings_do_not_leak_into_previous_article(self):
        units = chunking.build_legal_units(sample_doc())
        chunks = chunking.build_chunks(units)

        article_1_chunks = [
            chunk for chunk in chunks
            if chunk["metadata"].get("article_no") == 1
        ]

        self.assertTrue(article_1_chunks)
        self.assertNotIn("# Chương", " ".join(chunk["text"] for chunk in article_1_chunks))
        self.assertNotIn("PHỤ LỤC", " ".join(chunk["text"] for chunk in article_1_chunks))

    def test_quality_gate_reads_pretty_json_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "rows.json"
            temp_path.write_text(
                json.dumps([{"chunk_id": "demo"}], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            rows = load_rows(temp_path)

        self.assertEqual(rows, [{"chunk_id": "demo"}])

    def test_point_ids_keep_d_and_dd_distinct(self):
        self.assertEqual(chunking.point_label_to_id("d"), "d")
        self.assertEqual(chunking.point_label_to_id("\u0111"), "dd")

    def test_quality_gate_detects_article_count_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            doc = sample_doc(expected_article_count=3)

            errors = check_processed_doc(Path(temp_dir) / "demo.json", doc)

        self.assertIn("article count mismatch", "\n".join(errors))

    def test_quality_gate_rejects_unverified_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            doc = sample_doc(effective_date="01/01/2026")

            errors = check_processed_doc(Path(temp_dir) / "demo.json", doc)

        self.assertIn("unverified metadata field must move to registry: effective_date", "\n".join(errors))


if __name__ == "__main__":
    unittest.main()
