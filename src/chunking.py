import json
import re
from pathlib import Path


ARTICLE_RE = re.compile(r"(?m)^##\s+Điều\s+(\d+)\.\s*(.+?)\s*$")
CHAPTER_RE = re.compile(r"(?m)^#\s+Chương\s+([0-9IVXLCDM]+)\s*$", re.IGNORECASE)
CLAUSE_RE = re.compile(r"(?m)^(\d+)\s*\.\s+")
POINT_RE = re.compile(r"(?m)^\s*(?:[-–—]\s*)?([a-zđ])\)\s+")

MAX_CHARS = 2200
SHORT_TEXT_CHARS = 30
POINT_SPLIT_MIN_CHARS = 1200
POINT_SPLIT_MIN_POINTS = 3


def is_inside_quote(text: str, pos: int) -> bool:
    prefix = text[:pos]
    return prefix.count("“") > prefix.count("”")


def current_chapter(text_before_article: str):
    matches = list(CHAPTER_RE.finditer(text_before_article))
    if not matches:
        return None
    return matches[-1].group(1)


def parse_articles(doc: dict, text: str) -> list[dict]:
    matches = list(ARTICLE_RE.finditer(text))
    articles = []

    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        article_no = int(match.group(1))
        article_title = match.group(2).strip()
        body = text[match.end():end].strip()

        articles.append(
            {
                "unit_id": f"{doc['doc_id']}__dieu_{article_no}",
                "doc_id": doc["doc_id"],
                "doc_title": doc["title"],
                "doc_number": doc["doc_number"],
                "doc_type": doc["doc_type"],
                "source_url": doc["source_url"],
                "chapter": current_chapter(text[:start]),
                "article_no": article_no,
                "article_title": article_title,
                "text": body,
                "citation": f"{doc['title']}, Điều {article_no}",
            }
        )

    return articles


def parse_clauses(article: dict) -> list[dict]:
    body = article["text"]
    matches = [m for m in CLAUSE_RE.finditer(body) if not is_inside_quote(body, m.start())]

    if not matches:
        return [{**article, "level": "article"}]

    first_clause_no = int(matches[0].group(1))
    intro = body[:matches[0].start()].strip()
    if first_clause_no != 1 and intro:
        return [{**article, "level": "article", "parse_warning": "first_clause_is_not_1"}]

    clauses = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(body)
        clause_no = int(match.group(1))
        clause_text = body[start:end].strip()

        if intro:
            clause_text = intro + "\n" + clause_text

        clauses.append(
            {
                **{k: v for k, v in article.items() if k != "text"},
                "unit_id": f"{article['unit_id']}__khoan_{clause_no}",
                "level": "clause",
                "clause_no": clause_no,
                "text": clause_text,
                "citation": f"{article['citation']} khoản {clause_no}",
            }
        )

    return clauses


def build_legal_units(doc: dict) -> list[dict]:
    articles = parse_articles(doc, doc["clean_text"])
    legal_units = []
    for article in articles:
        legal_units.extend(parse_clauses(article))
    return legal_units


def split_long_text(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current = []
    current_len = 0

    for paragraph in paragraphs:
        if current and current_len + len(paragraph) + 1 > max_chars:
            chunks.append("\n".join(current))
            current = []
            current_len = 0
        current.append(paragraph)
        current_len += len(paragraph) + 1

    if current:
        chunks.append("\n".join(current))

    return chunks


def make_context_header(unit: dict) -> str:
    lines = [
        unit["doc_title"],
        f"Số hiệu: {unit['doc_number']}",
    ]
    if unit.get("chapter"):
        lines.append(f"Chương {unit['chapter']}")
    lines.append(f"Điều {unit['article_no']}. {unit['article_title']}")
    if unit.get("clause_no"):
        lines.append(f"Khoản {unit['clause_no']}")
    return "\n".join(lines)


def make_chunk_text(unit: dict, body: str) -> str:
    return f"{make_context_header(unit)}\n\n{body}".strip()


def point_label_to_id(label: str) -> str:
    return label.lower().replace("đ", "d")


def format_point_range(labels: list[str]) -> str:
    if len(labels) == 1:
        return labels[0]
    return f"{labels[0]}-{labels[-1]}"


def split_by_points(text: str, max_chars: int = MAX_CHARS) -> list[dict]:
    matches = list(POINT_RE.finditer(text))
    if len(matches) < POINT_SPLIT_MIN_POINTS:
        return []

    intro = text[:matches[0].start()].strip()
    point_parts = []
    for idx, match in enumerate(matches):
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        label = match.group(1).lower()
        point_parts.append({"labels": [label], "text": text[match.start():end].strip()})

    groups = []
    current_labels = []
    current_parts = []
    current_len = len(intro)

    for part in point_parts:
        part_len = len(part["text"]) + 1
        if current_parts and current_len + part_len > max_chars:
            body_parts = []
            if intro:
                body_parts.append(intro)
            body_parts.extend(current_parts)
            groups.append({"labels": current_labels, "text": "\n".join(body_parts).strip()})
            current_labels = []
            current_parts = []
            current_len = len(intro)

        current_labels.extend(part["labels"])
        current_parts.append(part["text"])
        current_len += part_len

    if current_parts:
        body_parts = []
        if intro:
            body_parts.append(intro)
        body_parts.extend(current_parts)
        groups.append({"labels": current_labels, "text": "\n".join(body_parts).strip()})

    return groups


def chunk_unit(unit: dict) -> list[dict]:
    text = unit["text"]
    point_count = len(POINT_RE.findall(text))
    should_split_points = len(text) > POINT_SPLIT_MIN_CHARS and point_count >= POINT_SPLIT_MIN_POINTS

    if should_split_points:
        point_groups = split_by_points(text)
        if point_groups:
            chunks = []
            for idx, group in enumerate(point_groups, 1):
                point_range = format_point_range(group["labels"])
                point_id = "_".join(point_label_to_id(label) for label in group["labels"])
                citation = f"{unit['citation']} điểm {point_range}"
                metadata = {
                    **{k: v for k, v in unit.items() if k != "text"},
                    "parent_unit_id": unit["unit_id"],
                    "level": "point_group" if len(group["labels"]) > 1 else "point",
                    "point_labels": group["labels"],
                    "point_range": point_range,
                    "citation": citation,
                    "split_reason": "long_clause_with_points",
                }
                chunks.append(
                    {
                        "chunk_id": f"{unit['unit_id']}__diem_{point_id}",
                        "text": group["text"],
                        "embedding_text": make_chunk_text(metadata, group["text"]),
                        "metadata": metadata,
                    }
                )
            return chunks

    parts = split_long_text(text)
    chunks = []
    for idx, part in enumerate(parts, 1):
        chunk_id = unit["unit_id"] if len(parts) == 1 else f"{unit['unit_id']}__part_{idx}"
        metadata = {k: v for k, v in unit.items() if k != "text"}
        if len(text.strip()) < SHORT_TEXT_CHARS:
            metadata["context_enrichment"] = "short_text_parent_context"
        elif len(parts) > 1:
            metadata["split_reason"] = "long_text_by_paragraph"

        chunks.append(
            {
                "chunk_id": chunk_id,
                "text": part,
                "embedding_text": make_chunk_text(unit, part),
                "metadata": metadata,
            }
        )
    return chunks


def build_chunks(units: list[dict]) -> list[dict]:
    chunks = []
    for unit in units:
        chunks.extend(chunk_unit(unit))
    return chunks


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_from_processed_file(processed_path: Path, structured_dir: Path, chunks_dir: Path) -> tuple[Path, Path]:
    with processed_path.open("r", encoding="utf-8") as f:
        doc = json.load(f)

    legal_units = build_legal_units(doc)
    chunks = build_chunks(legal_units)

    units_path = structured_dir / f"{doc['doc_id']}_legal_units.jsonl"
    chunks_path = chunks_dir / f"{doc['doc_id']}_chunks.jsonl"
    write_jsonl(units_path, legal_units)
    write_jsonl(chunks_path, chunks)

    return units_path, chunks_path


def build_all(processed_dir: Path, structured_dir: Path, chunks_dir: Path) -> None:
    for processed_path in sorted(processed_dir.glob("*.json")):
        units_path, chunks_path = build_from_processed_file(processed_path, structured_dir, chunks_dir)
        print(f"[OK] Built legal units: {units_path}")
        print(f"[OK] Built chunks: {chunks_path}")



