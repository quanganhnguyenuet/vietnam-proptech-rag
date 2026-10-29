"""Single source registry shared by the crawlers and query processor."""
import json
import re
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_SOURCES = Path(__file__).resolve().parents[2] / "config" / "legal_sources.json"


def load_sources(path: str | Path = DEFAULT_SOURCES) -> list[dict]:
    documents = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(documents, list) or not documents:
        raise ValueError("Source registry must be a nonempty list")
    ids = set()
    for doc in documents:
        for field in ("id", "title", "doc_number", "doc_type", "url"):
            if not isinstance(doc.get(field), str) or not doc[field].strip():
                raise ValueError(f"Source missing {field}")
        if not re.fullmatch(r"[A-Za-z0-9_]+", doc["id"]) or doc["id"] in ids:
            raise ValueError(f"Invalid/duplicate source ID: {doc['id']}")
        ids.add(doc["id"])
        for url in [doc["url"], *doc.get("fallback_urls", [])]:
            if urlparse(url).scheme != "https" or not urlparse(url).hostname:
                raise ValueError(f"Invalid HTTPS source URL: {url}")
    return documents
