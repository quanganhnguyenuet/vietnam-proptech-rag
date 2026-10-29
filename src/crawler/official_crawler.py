import html
import json
import re
import time
import zipfile
from pathlib import Path
from urllib.parse import urljoin, urlparse, parse_qs
from xml.etree import ElementTree

import requests
from bs4 import BeautifulSoup

from src.crawler import tvpl_crawler
from src.crawler.sources import load_sources
from src.crawler.http import create_session
from src.legal_structure import article_matches


_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]

RAW_DATA_DIR = _ROOT / "data" / "raw"
ATTACHMENT_DIR = RAW_DATA_DIR / "official_attachments"

HEADERS = tvpl_crawler.HEADERS.copy()

EXPECTED_ARTICLE_COUNTS = {doc["id"]: doc["expected_article_count"] for doc in load_sources() if doc.get("expected_article_count")}

SOURCE_POLICY = "registry_sources_with_validated_content"

OFFICIAL_DOCUMENTS = [
    {
        **doc,
        "expected_article_count": EXPECTED_ARTICLE_COUNTS.get(doc["id"]),
        "source_policy": SOURCE_POLICY,
    }
    for doc in load_sources()
]


def crawl_official_document(doc_info: dict, session: requests.Session) -> bool:
    for url in [doc_info["url"], *doc_info.get("fallback_urls", [])]:
        try:
            if _crawl_source({**doc_info, "url": url}, session):
                return True
        except (requests.RequestException, ValueError, zipfile.BadZipFile) as exc:
            print(f"[!] Source failed: {url}: {exc}")
    return False


def _crawl_source(doc_info: dict, session: requests.Session) -> bool:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ATTACHMENT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[*] Fetching source: {doc_info['doc_number']}...")
    response = session.get(doc_info["url"], timeout=30)
    response.encoding = "utf-8"
    print(f"    Status: {response.status_code}")

    if response.status_code != 200:
        print(f"[!] HTTP {response.status_code}: {doc_info['url']}")
        return False

    soup = BeautifulSoup(response.text, "html.parser")
    # Attachment links outside the legal-content area can belong to related laws.
    if source_provider(doc_info["url"]) == "tvpl":
        attachments = []
    else:
        attachments = find_attachment_links(soup, doc_info["url"])
    downloaded_files = []
    raw_text = ""
    raw_html = ""
    source_format = "html"

    # Prefer source attachments only when exposed. Otherwise keep the TVPL HTML path.
    for attachment_url in attachments:
        suffix = attachment_suffix(attachment_url)
        try:
            saved_path = download_attachment(session, attachment_url, doc_info["id"], suffix)
        except requests.RequestException as exc:
            print(f"    [!] Attachment download failed: {exc}")
            continue
        if not saved_path:
            continue

        downloaded_files.append(str(saved_path.relative_to(_ROOT)))
        try:
            extracted = extract_text_from_attachment(saved_path)
        except Exception as exc:
            print(f"    [!] Attachment extraction failed: {exc}")
            continue
        expected = doc_info.get("expected_article_count")
        from src.crawler.clean_dataset import split_main_and_appendices
        count = len(article_matches(split_main_and_appendices(extracted)[0]))
        if count and matches_document_number(extracted, doc_info["doc_number"]) and (expected is None or count == expected):
            raw_text = extracted
            source_format = suffix.lstrip(".") or "attachment"
            break

    if not raw_text:
        raw_html = extract_best_html_content(soup, doc_info["url"])
        from src.crawler.clean_dataset import clean_html, split_main_and_appendices
        extracted = clean_html(raw_html)
        count = len(article_matches(split_main_and_appendices(extracted)[0]))
        expected = doc_info.get("expected_article_count")
        if not count or not matches_document_number(extracted, doc_info["doc_number"]) or (expected is not None and count != expected):
            (ATTACHMENT_DIR / f"{doc_info['id']}.source.html").write_text(response.text, encoding="utf-8")
            print(f"[!] Incomplete legal content for {doc_info['id']}: {count} articles; HTML/OCR or source review required")
            return False

    data = {
        "doc_id": doc_info["id"],
        "title": doc_info["title"],
        "doc_number": doc_info["doc_number"],
        "doc_type": doc_info["doc_type"],
        "source_provider": source_provider(doc_info["url"]),
        "source_policy": doc_info.get("source_policy"),
        "source_url": doc_info["url"],
        "source_format": source_format,
        "downloaded_files": downloaded_files,
        "crawled_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "expected_article_count": doc_info.get("expected_article_count"),
    }

    if raw_text:
        data["raw_text"] = raw_text
    else:
        data["raw_html"] = raw_html

    output_file = RAW_DATA_DIR / f"{doc_info['id']}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved: {output_file}")
    return True


def crawl_all(documents: list[dict] | None = None) -> None:
    session = create_session(HEADERS)

    docs = documents or OFFICIAL_DOCUMENTS
    for idx, doc in enumerate(docs, 1):
        print(f"\n[Progress {idx}/{len(docs)}]")
        try:
            crawl_official_document(doc, session)
        except Exception as exc:
            print(f"[!] Failed {doc['id']}: {exc}")
        time.sleep(2)


def source_provider(url: str) -> str:
    if "thuvienphapluat.vn" in url:
        return "tvpl"
    if "congbao.chinhphu.vn" in url:
        return "congbao"
    if "vbpl.vn" in url:
        return "vbpl"
    if "vanban.chinhphu.vn" in url:
        return "chinhphu"
    return "unknown"


def matches_document_number(text: str, doc_number: str) -> bool:
    compact = re.sub(r"\s+", "", text[:6000]).casefold()
    number = re.sub(r"\s+", "", doc_number).casefold()
    return f"số:{number}" in compact


def extract_best_html_content(soup: BeautifulSoup, url: str) -> str:
    selectors = []
    if "thuvienphapluat.vn" in url:
        selectors.extend(
            [
                ("div", {"class": "content1"}),
                ("div", {"id": "divContentDoc"}),
                ("div", {"class": "content"}),
            ]
        )
    selectors.extend(
        [
            ("div", {"id": re.compile("toanvan|content", re.IGNORECASE)}),
            ("article", {}),
            ("body", {}),
        ]
    )

    for name, attrs in selectors:
        content = soup.find(name, attrs=attrs)
        if content and len(content.get_text(" ", strip=True)) > 500:
            return str(content)

    return str(soup)


def find_attachment_links(soup: BeautifulSoup, base_url: str) -> list[str]:
    candidates = []
    for link in soup.find_all("a", href=True):
        href = html.unescape(link["href"])
        label = link.get_text(" ", strip=True)
        target = urljoin(base_url, href)
        haystack = f"{href} {label}".lower()
        if any(ext in haystack for ext in [".docx", ".pdf", ".doc"]) or attachment_suffix(target) in {".docx", ".pdf", ".doc"}:
            candidates.append(target)

    def rank(url: str) -> int:
        suffix = attachment_suffix(url)
        if suffix == ".docx":
            return 0
        if suffix == ".pdf":
            return 1
        if suffix == ".doc":
            return 2
        return 3

    return sorted(dict.fromkeys(candidates), key=rank)


def attachment_suffix(url: str) -> str:
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix.lower()
    if suffix in {".docx", ".pdf", ".doc"}:
        return suffix
    fmt = parse_qs(parsed.query).get("format", [""])[0].lower()
    filename = parse_qs(parsed.query).get("file_name", [""])[0]
    if Path(filename).suffix.lower() in {".docx", ".pdf", ".doc"}:
        return Path(filename).suffix.lower()
    return f".{fmt}" if fmt in {"docx", "pdf", "doc"} else ""


def download_attachment(
    session: requests.Session,
    url: str,
    doc_id: str,
    fallback_suffix: str,
) -> Path | None:
    response = session.get(url, timeout=45)
    if response.status_code != 200 or not response.content:
        return None

    suffix = fallback_suffix if fallback_suffix in [".docx", ".pdf", ".doc"] else ".bin"
    path = ATTACHMENT_DIR / f"{doc_id}{suffix}"
    path.write_bytes(response.content)
    return path


def extract_text_from_attachment(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return extract_docx_text(path)
    if suffix == ".pdf":
        return extract_pdf_text(path)
    return ""


def extract_docx_text(path: Path) -> str:
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        xml = archive.read("word/document.xml")
    root = ElementTree.fromstring(xml)
    paragraphs = []
    for paragraph in root.findall(".//w:p", namespace):
        texts = [node.text or "" for node in paragraph.findall(".//w:t", namespace)]
        line = "".join(texts).strip()
        if line:
            paragraphs.append(line)
    return "\n".join(paragraphs)


def extract_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        print("    [!] pypdf is not installed; falling back to HTML.")
        return ""

    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages).strip()


if __name__ == "__main__":
    crawl_all()
