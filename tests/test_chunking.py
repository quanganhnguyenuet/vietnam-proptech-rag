import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "data" / "structured" / "31_2024_QH15_legal_units.jsonl"

rows = []
with PATH.open("r", encoding="utf-8") as f:
    for line_no, line in enumerate(f, 1):
        try:
            obj = json.loads(line)
            obj["_line_no"] = line_no
            rows.append(obj)
        except Exception as e:
            print(f"[JSON ERROR] line={line_no}: {e}")

required_fields = [
    "unit_id", "doc_id", "doc_title", "doc_number",
    "article_no", "article_title", "citation",
    "level", "text", "source_url"
]

errors = []
warnings = []

# Schema check
for r in rows:
    for field in required_fields:
        if field not in r or r[field] in [None, ""]:
            errors.append(f"Missing field {field}: line {r.get('_line_no')}")

# Duplicate check
for field in ["unit_id", "citation"]:
    counter = Counter(r.get(field) for r in rows)
    duplicates = [(k, v) for k, v in counter.items() if v > 1]
    for k, v in duplicates:
        errors.append(f"Duplicate {field}: {k} appears {v} times")

# Article coverage
article_numbers = sorted(set(r["article_no"] for r in rows if "article_no" in r))
if article_numbers:
    missing_articles = [
        i for i in range(min(article_numbers), max(article_numbers) + 1)
        if i not in article_numbers
    ]
    for a in missing_articles:
        errors.append(f"Missing article: Điều {a}")

# Clause sequence check
by_article = defaultdict(list)
for r in rows:
    by_article[r["article_no"]].append(r)

for article_no, units in by_article.items():
    clause_nos = sorted(
        r.get("clause_no")
        for r in units
        if r.get("clause_no") is not None
    )
    if clause_nos:
        if min(clause_nos) != 1:
            warnings.append(
                f"Article {article_no} clause sequence starts at {min(clause_nos)}"
            )

        expected = set(range(min(clause_nos), max(clause_nos) + 1))
        missing = sorted(expected - set(clause_nos))
        if missing:
            warnings.append(
                f"Article {article_no} missing clause numbers: {missing}"
            )

# Text length check
lengths = [len(r.get("text", "")) for r in rows]
for r in rows:
    text = r.get("text", "")

    if len(text.strip()) == 0:
        errors.append(f"Empty text: {r['unit_id']}")

    if len(text) < 30:
        warnings.append(f"Very short unit: {r['unit_id']} len={len(text)}")

    if len(text) > 2200:
        warnings.append(f"Very long unit: {r['unit_id']} len={len(text)}")

# Boundary check
for r in rows:
    text = r.get("text", "")
    article_no = r.get("article_no")
    clause_no = r.get("clause_no")

    # If another article heading appears inside text, likely boundary error
    other_article_heading = re.search(r"(?m)^Điều\s+(\d+)\.", text)
    if other_article_heading:
        found_article = int(other_article_heading.group(1))
        if found_article != article_no:
            errors.append(
                f"Boundary error: {r['unit_id']} contains Điều {found_article}"
            )

    # Clause marker consistency
    if clause_no is not None:
        first_clause_marker = re.search(r"(?m)^\s*(\d+)\.\s+", text)
        if first_clause_marker:
            found_clause = int(first_clause_marker.group(1))
            if found_clause != clause_no:
                errors.append(
                    f"Clause mismatch: {r['unit_id']} metadata={clause_no}, text={found_clause}"
                )

# Point marker check
point_re = re.compile(r"(?m)^\s*(?:[-–—]\s*)?([a-zđ])\)\s+")
point_units = []
for r in rows:
    points = point_re.findall(r.get("text", ""))
    if points:
        point_units.append((r, points))

for r, points in point_units:
    if len(points) >= 5 and len(r.get("text", "")) > 1200:
        warnings.append(
            f"Long clause with many points should be split: {r['unit_id']} "
            f"points={len(points)} len={len(r['text'])}"
        )

print("===== CHUNK QUALITY REPORT =====")
print(f"Total units: {len(rows)}")
print(f"Articles found: {len(article_numbers)}")
print(f"Length min/avg/median/max: {min(lengths)} / {mean(lengths):.1f} / {median(lengths)} / {max(lengths)}")
print(f"Units with point markers: {len(point_units)}")
print(f"Errors: {len(errors)}")
print(f"Warnings: {len(warnings)}")

print("\n--- ERRORS ---")
for e in errors[:50]:
    print(e)

print("\n--- WARNINGS ---")
for w in warnings[:50]:
    print(w)
