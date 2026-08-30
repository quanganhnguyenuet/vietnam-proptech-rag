from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

@dataclass(frozen=True)
class NormalizedQuery:
    original: str
    normalized: str

class InvalidQueryError(ValueError):
    """
    Raised when an input query cannot be used for retrieval.
    """

_ZERO_WIDTH_RE = re.compile(r"[\u200B\u200c\u200D\uFEFF]")
_WHITESPACE_RE = re.compile(r"\s+")

def normalize_query(query: str, *, max_length: int = 1000) -> NormalizedQuery:
    #1 Kiem tra co phai chuoi string khong
    if not isinstance(query, str):
        raise InvalidQueryError(f"Query must be a string, got {type(query).__name__}")
    original_query = query
    #2 Loai bo ky tu dac biet
    normalize_query = _ZERO_WIDTH_RE.sub(" ", query)
    #3 Chuyen ve unicode NFC
    normalize_query = unicodedata.normalize("NFC", normalize_query)
    #4 Loai bo khoang trang thua
    normalize_query = _WHITESPACE_RE.sub(" ", normalize_query).strip()
    #5 Kiem tra do dai chuoi
    if not normalize_query:
        raise InvalidQueryError("Query is empty after normalization")   
    if len(normalize_query) > max_length:
        raise InvalidQueryError(f"Query exceeds maximum length of {max_length} characters")
    
    return NormalizedQuery(original=original_query, normalized=normalize_query)
