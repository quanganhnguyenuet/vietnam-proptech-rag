import json
import re
from collections import Counter
from pathlib import Path

from src.crawler.clean_dataset import split_main_and_appendices


ARTICLE_HEADING_RE = re.compile(
    r"(?m)^\s*(?:#{1,6}\s*)?(?:[·oº\-\s]*)?Điều\s+(\d+[a-zA-Z]?)\s*[.:]",
    re.IGNORECASE,
)

EXPECTED_ARTICLE_COUNTS = {
    "31_2024_QH15": 260,
    "27_2023_QH15": 198,
    "29_2023_QH15": 83,
    "101_2024_ND_CP": 68,
    "102_2024_ND_CP": 113,
}

UNVERIFIED_METADATA_FIELDS = {
    "effective_date",
    "status",
    "amended_by",
    "replaces",
    "related_docs",
}


class QualityGateError(Exception):
    pass


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise QualityGateError(f"{path}: invalid JSONL at line {line_no}: {exc}") from exc
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def run_quality_gate(processed_dir: Path, structured_dir: Path, chunks_dir: Path) -> None:
    errors: list[str] = []

    for processed_path in sorted(processed_dir.glob("*.json")):
        with processed_path.open(encoding="utf-8-sig") as f:
            doc = json.load(f)
        errors.extend(check_processed_doc(processed_path, doc))

    all_units = []
    for units_path in sorted(structured_dir.glob("*_legal_units.jsonl")):
        rows = load_jsonl(units_path)
        all_units.extend(rows)
        errors.extend(check_units(units_path, rows))

    all_chunks = []
    for chunks_path in sorted(chunks_dir.glob("*_chunks.jsonl")):
        rows = load_jsonl(chunks_path)
        all_chunks.extend(rows)
        errors.extend(check_chunks(chunks_path, rows))

    errors.extend(check_duplicate_key("global unit_id", all_units, "unit_id"))
    errors.extend(check_duplicate_key("global chunk_id", all_chunks, "chunk_id"))

    if errors:
        print("\n[QUALITY GATE] FAILED")
        for error in errors[:100]:
            print(f"- {error}")
        if len(errors) > 100:
            print(f"- ... {len(errors) - 100} more errors")
        raise QualityGateError(f"Quality gate failed with {len(errors)} error(s).")

    print("[QUALITY GATE] PASSED")


def check_processed_doc(path: Path, doc: dict) -> list[str]:
    errors = []
    main_text = doc.get("main_text")
    if main_text is None and doc.get("clean_text"):
        main_text, _ = split_main_and_appendices(doc["clean_text"])

    if not main_text or not main_text.strip():
        return [f"{path.name}: empty main_text"]

    article_count = len(ARTICLE_HEADING_RE.findall(main_text))
    expected = doc.get("expected_article_count") or EXPECTED_ARTICLE_COUNTS.get(doc.get("doc_id"))
    if expected is not None and article_count != int(expected):
        errors.append(
            f"{path.name}: article count mismatch, expected {expected}, found {article_count}"
        )
    if article_count == 0:
        errors.append(f"{path.name}: no article headings detected in main_text")

    for field in UNVERIFIED_METADATA_FIELDS:
        if field in doc:
            errors.append(f"{path.name}: unverified metadata field must move to registry: {field}")

    return errors


def check_units(path: Path, rows: list[dict]) -> list[str]:
    errors = []
    required = [
        "unit_id",
        "doc_id",
        "doc_title",
        "doc_number",
        "doc_type",
        "source_url",
        "section_type",
        "level",
        "text",
        "citation",
    ]

    errors.extend(check_duplicate_key(path.name, rows, "unit_id"))

    for row in rows:
        unit_id = row.get("unit_id", f"line {row.get('_line_no')}")
        for field in required:
            if row.get(field) in [None, ""]:
                errors.append(f"{path.name}: {unit_id} missing {field}")
        for field in UNVERIFIED_METADATA_FIELDS:
            if field in row:
                errors.append(f"{path.name}: {unit_id} contains unverified metadata field {field}")

        text = row.get("text", "")
        if not text.strip():
            errors.append(f"{path.name}: {unit_id} has empty text")

        if row.get("section_type") == "main" and ARTICLE_HEADING_RE.search(text):
            errors.append(f"{path.name}: {unit_id} contains another article heading")

    return errors


def check_chunks(path: Path, rows: list[dict]) -> list[str]:
    errors = []
    errors.extend(check_duplicate_key(path.name, rows, "chunk_id"))

    for row in rows:
        chunk_id = row.get("chunk_id", f"line {row.get('_line_no')}")
        metadata = row.get("metadata") or {}
        text = row.get("text", "")

        if not text.strip():
            errors.append(f"{path.name}: {chunk_id} has empty text")
        if not row.get("embedding_text", "").strip():
            errors.append(f"{path.name}: {chunk_id} missing embedding_text")
        if not metadata.get("unit_id"):
            errors.append(f"{path.name}: {chunk_id} missing metadata.unit_id")
        if not metadata.get("citation"):
            errors.append(f"{path.name}: {chunk_id} missing metadata.citation")
        for field in UNVERIFIED_METADATA_FIELDS:
            if field in metadata:
                errors.append(f"{path.name}: {chunk_id} metadata contains unverified field {field}")
        if metadata.get("section_type") == "main" and ARTICLE_HEADING_RE.search(text):
            errors.append(f"{path.name}: {chunk_id} contains another article heading")

    return errors


def check_duplicate_key(scope: str, rows: list[dict], key: str) -> list[str]:
    counts = Counter(row.get(key) for row in rows)
    return [
        f"{scope}: duplicate {key}={value!r} appears {count} times"
        for value, count in counts.items()
        if value and count > 1
    ]


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    run_quality_gate(root / "data" / "processed", root / "data" / "structured", root / "data" / "chunks")
