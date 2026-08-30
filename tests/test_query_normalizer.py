import unittest
import unicodedata

from src.retrieval.query_normalizer import (
    InvalidQueryError,
    normalize_query,
)


class QueryNormalizerTests(unittest.TestCase):
    def test_collapses_whitespace(self):
        result = normalize_query(
            "  Quyền sử dụng\n\n đất\t là gì?  "
        )

        self.assertEqual(
            result.normalized,
            "Quyền sử dụng đất là gì?",
        )

    def test_preserves_original_query(self):
        raw = "  Giá đất   tại Hà Nội?  "

        result = normalize_query(raw)

        self.assertEqual(result.original, raw)
        self.assertEqual(
            result.normalized,
            "Giá đất tại Hà Nội?",
        )

    def test_removes_zero_width_characters(self):
        result = normalize_query(
            "quyền sử\u200bdụng đất"
        )

        self.assertEqual(
            result.normalized,
            "quyền sử dụng đất",
        )

    def test_normalizes_unicode_to_nfc(self):
        decomposed = unicodedata.normalize(
            "NFD",
            "Quyền sử dụng đất",
        )

        result = normalize_query(decomposed)

        self.assertTrue(
            unicodedata.is_normalized("NFC", result.normalized)
        )
        self.assertEqual(
            result.normalized,
            "Quyền sử dụng đất",
        )

    def test_rejects_empty_query(self):
        with self.assertRaises(InvalidQueryError):
            normalize_query(" \n\t ")

    def test_rejects_non_string_query(self):
        with self.assertRaises(InvalidQueryError):
            normalize_query(None)  # type: ignore[arg-type]

    def test_rejects_query_over_limit(self):
        with self.assertRaises(InvalidQueryError):
            normalize_query("a" * 11, max_length=10)


if __name__ == "__main__":
    unittest.main()