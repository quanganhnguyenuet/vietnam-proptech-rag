import json
import re
from pathlib import Path

from src.crawler.clean_dataset import split_main_and_appendices
from src.legal_structure import ARTICLE_RE, article_matches, is_inside_quote


CHAPTER_RE = re.compile(
    r"(?m)^\s*(?:#{1,6}\s*)?(?:[·oº\-\s]*)?Chương\s+([0-9IVXLCDM]+)\b.*$",
    re.IGNORECASE,
)
STRUCTURE_HEADING_RE = re.compile(
    r"(?m)^\s*#{1,6}\s*(?:Chương|Mục|Tiểu mục)\b.*$",
    re.IGNORECASE,
)
CLAUSE_RE = re.compile(r"(?m)^[ \t]*(\d+)[ \t]*\.[ \t]+")
POINT_RE = re.compile(r"(?m)^\s*(?:[-–—]\s*)?([a-zđ])\)\s+", re.IGNORECASE)
MARKDOWN_HEADING_RE = re.compile(r"(?m)^\s*#{1,6}\s*")
LIST_MARKER_RE = re.compile(r"(?m)^\s*[-*+]\s+")
INLINE_MARKDOWN_RE = re.compile(r"[*_`]+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;:!?])\s+")

MAX_CHARS = 2200
SHORT_TEXT_CHARS = 30
POINT_SPLIT_MIN_CHARS = 1200
POINT_SPLIT_MIN_POINTS = 3

DOC_METADATA_FIELDS = [
    "expected_article_count",
    "source_provider",
    "source_policy",
    "source_format",
    "downloaded_files",
]


def current_chapter(text_before_article: str):
    matches = list(CHAPTER_RE.finditer(text_before_article))
    if not matches:
        return None
    return matches[-1].group(1)


def strip_trailing_structure_headings(text: str) -> str:
    match = next((m for m in STRUCTURE_HEADING_RE.finditer(text) if not is_inside_quote(text, m.start())), None)
    if not match:
        return text.strip()
    return text[: match.start()].strip()


def doc_metadata(doc: dict) -> dict:
    return {field: doc.get(field) for field in DOC_METADATA_FIELDS if field in doc}


def make_unit_id(
    doc_id: str,
    section_type: str,
    article_no=None,
    clause_no=None,
    appendix_id=None,
) -> str:
    parts = [doc_id, section_type]
    if section_type == "appendix":
        parts.append(appendix_id or "appendix")
    if article_no is not None:
        parts.append(f"article_{article_no}")
    if clause_no is not None:
        parts.append(f"clause_{clause_no}")
    return "__".join(str(part) for part in parts)


def parse_articles(doc: dict, text: str) -> list[dict]:
    matches = article_matches(text)
    articles = []

    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        article_no = match.group(1)
        article_title = match.group(2).strip()
        body = strip_trailing_structure_headings(text[match.end():end])
        if not body.strip():
            # Some sources put the entire provision on the heading line.
            body = article_title

        articles.append(
            {
                **doc_metadata(doc),
                "unit_id": make_unit_id(doc["doc_id"], "main", article_no=article_no),
                "doc_id": doc["doc_id"],
                "doc_title": doc["title"],
                "doc_number": doc["doc_number"],
                "doc_type": doc["doc_type"],
                "source_url": doc["source_url"],
                "section_type": "main",
                "chapter": current_chapter(text[:start]),
                "article_no": int(article_no) if article_no.isdigit() else article_no,
                "article_label": article_no,
                "article_title": article_title,
                "text": body,
                "citation": f"{doc['title']}, Điều {article_no}",
            }
        )

    return articles


def parse_clauses(article: dict) -> list[dict]:
    body = article["text"]
    if body.count("“") != body.count("”") or body.count('"') % 2:
        return [{**article, "level": "article", "parse_warning": "unbalanced_quotes"}]
    matches = [m for m in CLAUSE_RE.finditer(body) if not is_inside_quote(body, m.start())]

    if not matches:
        return [{**article, "level": "article"}]

    clause_numbers = [int(match.group(1)) for match in matches]
    if clause_numbers != list(range(1, len(matches) + 1)):
        return [{**article, "level": "article", "parse_warning": "non_sequential_clauses"}]
    first_clause_no = int(matches[0].group(1))
    intro = body[: matches[0].start()].strip()
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
                "unit_id": make_unit_id(
                    article["doc_id"],
                    "main",
                    article_no=article["article_label"],
                    clause_no=clause_no,
                ),
                "level": "clause",
                "clause_no": clause_no,
                "text": clause_text,
                "citation": f"{article['citation']} khoản {clause_no}",
            }
        )

    return clauses


def appendix_units(doc: dict, appendices: list[dict]) -> list[dict]:
    rows = []
    for appendix in appendices:
        text = appendix.get("text", "").strip()
        if not text:
            continue

        appendix_id = appendix.get("appendix_id") or f"appendix_{len(rows) + 1}"
        title = appendix.get("title") or appendix_id
        rows.append(
            {
                **doc_metadata(doc),
                "unit_id": make_unit_id(doc["doc_id"], "appendix", appendix_id=appendix_id),
                "doc_id": doc["doc_id"],
                "doc_title": doc["title"],
                "doc_number": doc["doc_number"],
                "doc_type": doc["doc_type"],
                "source_url": doc["source_url"],
                "section_type": "appendix",
                "appendix_id": appendix_id,
                "appendix_title": title,
                "level": "appendix",
                "text": text,
                "citation": f"{doc['title']}, {title}",
            }
        )
    return rows


def build_legal_units(doc: dict) -> list[dict]:
    main_text = doc.get("main_text")
    appendices = doc.get("appendices")
    if main_text is None:
        main_text, detected_appendices = split_main_and_appendices(doc["clean_text"])
        appendices = appendices if appendices is not None else detected_appendices

    legal_units = []
    for article in parse_articles(doc, main_text):
        legal_units.extend(parse_clauses(article))
    legal_units.extend(appendix_units(doc, appendices or []))
    return legal_units


def split_long_text(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current = []
    current_len = 0

    def flush_current():
        nonlocal current, current_len
        if current:
            chunks.append("\n".join(current))
            current = []
            current_len = 0

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            flush_current()
            chunks.extend(split_long_paragraph(paragraph, max_chars))
            continue

        if current and current_len + len(paragraph) + 1 > max_chars:
            flush_current()
        current.append(paragraph)
        current_len += len(paragraph) + 1

    flush_current()

    return chunks


def split_long_paragraph(paragraph: str, max_chars: int = MAX_CHARS) -> list[str]:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    pieces = [piece.strip() for piece in SENTENCE_SPLIT_RE.split(paragraph) if piece.strip()]
    if not pieces:
        return []

    chunks = []
    current = []
    current_len = 0

    for piece in pieces:
        if len(piece) > max_chars:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_len = 0
            while len(piece) > max_chars:
                boundary = piece.rfind(" ", 0, max_chars + 1)
                if boundary <= 0:
                    boundary = max_chars
                chunks.append(piece[:boundary].strip())
                piece = piece[boundary:].strip()
            if piece:
                chunks.append(piece)
            continue

        if current and current_len + len(piece) + 1 > max_chars:
            chunks.append(" ".join(current))
            current = []
            current_len = 0

        current.append(piece)
        current_len += len(piece) + 1

    if current:
        chunks.append(" ".join(current))

    return chunks


def make_context_header(unit: dict) -> str:
    lines = [
        unit["doc_title"],
        f"Số hiệu: {unit['doc_number']}",
    ]
    if unit.get("section_type") == "appendix":
        lines.append(unit.get("appendix_title") or unit.get("appendix_id") or "Phụ lục")
        return "\n".join(lines)

    if unit.get("chapter"):
        lines.append(f"Chương {unit['chapter']}")
    lines.append(f"Điều {unit['article_label']}. {unit['article_title']}")
    if unit.get("clause_no"):
        lines.append(f"Khoản {unit['clause_no']}")
    return "\n".join(lines)


def make_chunk_text(unit: dict, body: str) -> str:
    return f"{make_context_header(unit)}\n\n{body}".strip()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_embedding_text(text: str) -> str:
    text = MARKDOWN_HEADING_RE.sub("", text)
    text = LIST_MARKER_RE.sub("", text)
    text = INLINE_MARKDOWN_RE.sub("", text)
    return normalize_text(text)


def point_label_to_id(label: str) -> str:
    return label.lower().replace("đ", "dd")


def format_point_range(labels: list[str]) -> str:
    if len(labels) == 1:
        return labels[0]
    return f"{labels[0]}-{labels[-1]}"


def split_by_points(text: str, max_chars: int = MAX_CHARS) -> list[dict]:
    matches = [m for m in POINT_RE.finditer(text) if not is_inside_quote(text, m.start())]
    if len(matches) < POINT_SPLIT_MIN_POINTS:
        return []
    if len({m.group(1).lower() for m in matches}) != len(matches):
        return []

    intro = text[: matches[0].start()].strip()
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
            body_parts = [intro] if intro else []
            body_parts.extend(current_parts)
            groups.append({"labels": current_labels, "text": "\n".join(body_parts).strip()})
            current_labels = []
            current_parts = []
            current_len = len(intro)

        current_labels.extend(part["labels"])
        current_parts.append(part["text"])
        current_len += part_len

    if current_parts:
        body_parts = [intro] if intro else []
        body_parts.extend(current_parts)
        groups.append({"labels": current_labels, "text": "\n".join(body_parts).strip()})

    return groups


def chunk_unit(unit: dict) -> list[dict]:
    text = unit["text"]
    point_count = len(POINT_RE.findall(text))
    should_split_points = (
        unit.get("section_type") == "main"
        and not unit.get("parse_warning")
        and len(text) > POINT_SPLIT_MIN_CHARS
        and point_count >= POINT_SPLIT_MIN_POINTS
    )

    if should_split_points:
        point_groups = split_by_points(text)
        if point_groups:
            chunks = []
            for group in point_groups:
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
                parts = split_long_text(group["text"])
                for idx, part in enumerate(parts, 1):
                    part_metadata = dict(metadata)
                    chunk_id = f"{unit['unit_id']}__point_{point_id}"
                    if len(parts) > 1:
                        chunk_id = f"{chunk_id}__part_{idx}"
                        part_metadata["part_index"] = idx
                        part_metadata["part_count"] = len(parts)
                    chunks.append(
                        {
                            "chunk_id": chunk_id,
                            "text": normalize_text(part),
                            "embedding_text": normalize_embedding_text(make_chunk_text(part_metadata, part)),
                            "metadata": part_metadata,
                        }
                    )
            return chunks

    parts = split_long_text(text)
    chunks = []
    for idx, part in enumerate(parts, 1):
        chunk_id = unit["unit_id"] if len(parts) == 1 else f"{unit['unit_id']}__part_{idx}"
        metadata = {k: v for k, v in unit.items() if k != "text"}
        metadata["parent_unit_id"] = unit["unit_id"]
        if len(parts) > 1:
            metadata["part_index"] = idx
            metadata["part_count"] = len(parts)
        if len(text.strip()) < SHORT_TEXT_CHARS:
            metadata["context_enrichment"] = "short_text_parent_context"
        elif len(parts) > 1:
            metadata["split_reason"] = "long_text_by_paragraph"

        chunks.append(
            {
                "chunk_id": chunk_id,
                "text": normalize_text(part),
                "embedding_text": normalize_embedding_text(make_chunk_text(unit, part)),
                "metadata": metadata,
            }
        )
    return chunks


def build_chunks(units: list[dict]) -> list[dict]:
    chunks = []
    for unit in units:
        chunks.extend(chunk_unit(unit))
    return chunks


def write_json(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

def build_from_processed_file(processed_path: Path, structured_dir: Path, chunks_dir: Path) -> tuple[Path, Path]:
    with processed_path.open("r", encoding="utf-8-sig") as f:
        doc = json.load(f)

    legal_units = build_legal_units(doc)
    chunks = build_chunks(legal_units)

    units_path = structured_dir / f"{doc['doc_id']}_legal_units.json"
    chunks_path = chunks_dir / f"{doc['doc_id']}_chunks.json"
    write_json(units_path, legal_units)
    write_json(chunks_path, chunks)

    return units_path, chunks_path


def build_all(processed_dir: Path, structured_dir: Path, chunks_dir: Path) -> None:
    for processed_path in sorted(processed_dir.glob("*.json")):
        units_path, chunks_path = build_from_processed_file(processed_path, structured_dir, chunks_dir)
        print(f"[OK] Built legal units: {units_path}")
        print(f"[OK] Built chunks: {chunks_path}")
