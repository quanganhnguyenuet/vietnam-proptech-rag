import json
import re
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]

RAW_DIR = _ROOT / "data" / "raw"
PROCESSED_DIR = _ROOT / "data" / "processed"
UNVERIFIED_METADATA_FIELDS = {
    "effective_date",
    "status",
    "amended_by",
    "replaces",
    "related_docs",
}

APPENDIX_START_RE = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*)?(?:PHỤ\s+LỤC|PHU\s+LUC|MẪU\s+SỐ|Mẫu\s+số|DANH\s+MỤC\s+MẪU|"
    r"DANH\s+MỤC\s+BIỂU|BIỂU\s+MẪU)\b"
)


def _normalize(text: str) -> str:
    text = re.sub(r"\r\n\s*", " ", text)
    text = re.sub(r"\n\s*", " ", text)
    text = text.replace("\xa0", " ")
    text = text.replace("\u200b", "")
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"^(\d+)\s+\.", r"\1.", text)
    text = re.sub(r"^([a-zđ])\s+\)", r"\1)", text, flags=re.IGNORECASE)
    text = re.sub(r"\bĐi\s+ều\b", "Điều", text)
    text = re.sub(r"\bCh\s+ương\b", "Chương", text)
    text = re.sub(r"\bM\s+ục\b", "Mục", text)
    return text.strip()


def clean_html(raw_html: str) -> str:
    """Convert HTML to plain legal text while preserving article/clause boundaries."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(raw_html, "lxml")

    paragraphs = []
    for tag in soup.find_all(["p", "h1", "h2", "h3", "h4", "li"]):
        text = tag.get_text(separator=" ")
        text = _normalize(text)
        if text:
            paragraphs.append(text)

    text = "\n".join(paragraphs)
    return normalize_legal_headings(text)


def normalize_legal_headings(text: str) -> str:
    text = re.sub(
        r"(?m)^\s*(?:[·oº\-\s]*)?(Chương\s+[0-9IVXLCDM]+.*)$",
        r"\n# \1\n",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(?m)^\s*(?:[·oº\-\s]*)?(Mục\s+\d+.*)$",
        r"\n## \1\n",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(?m)^\s*(?:[·oº\-\s]*)?(Điều\s+\d+[a-zA-Z]?\.\s.*)$",
        r"\n## \1\n",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"(?m)^([a-zđ]\)\s+.*)$", r"- \1", text, flags=re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_main_and_appendices(text: str) -> tuple[str, list[dict]]:
    """Split enforceable text from appendices/forms."""
    matches = list(APPENDIX_START_RE.finditer(text))
    if not matches:
        return text.strip(), []

    main_text = text[: matches[0].start()].strip()
    appendices = []

    for idx, match in enumerate(matches, 1):
        start = match.start()
        end = matches[idx].start() if idx < len(matches) else len(text)
        appendix_text = text[start:end].strip()
        title = appendix_text.splitlines()[0].strip() if appendix_text else f"Phụ lục {idx}"
        appendices.append(
            {
                "appendix_id": f"appendix_{idx}",
                "title": title,
                "text": appendix_text,
            }
        )

    return main_text, appendices


def clean_file(input_path: Path, output_path: Path):
    print(f"[*] Processing: {input_path.name}")

    with open(input_path, encoding="utf-8-sig") as f:
        data = json.load(f)

    raw_text = data.get("raw_text", "")
    raw_html = data.get("raw_html", "")
    if raw_text:
        clean_text = normalize_legal_headings(raw_text.strip())
    elif raw_html:
        clean_text = clean_html(raw_html)
    else:
        print("  [!] Missing raw_text/raw_html")
        return

    main_text, appendices = split_main_and_appendices(clean_text)
    num_articles = len(re.findall(r"(?m)^##\s+Điều\s+\d+[a-zA-Z]?\.", main_text))
    print(
        f"  -> {len(clean_text):,} chars | main articles={num_articles} | "
        f"main={len(main_text):,} | appendices={len(appendices)}"
    )

    excluded_fields = {"raw_html", "raw_text", *UNVERIFIED_METADATA_FIELDS}
    output_data = {k: v for k, v in data.items() if k not in excluded_fields}
    output_data["clean_text"] = clean_text
    output_data["main_text"] = main_text
    output_data["appendices"] = appendices

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    txt_path = output_path.with_suffix(".txt")
    txt_path.write_text(main_text, encoding="utf-8")

    print(f"  [OK] {output_path}")
    print(f"  [OK] {txt_path}")
