import html
import json
import re
import time
import zipfile
from pathlib import Path
from urllib.parse import urljoin
from xml.etree import ElementTree

import requests
from bs4 import BeautifulSoup

from src.crawler import tvpl_crawler


_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]

RAW_DATA_DIR = _ROOT / "data" / "raw"
ATTACHMENT_DIR = RAW_DATA_DIR / "official_attachments"

HEADERS = tvpl_crawler.HEADERS.copy()

EXPECTED_ARTICLE_COUNTS = {
    "31_2024_QH15": 260,
    "27_2023_QH15": 198,
    "29_2023_QH15": 83,
    "101_2024_ND_CP": 68,
    "102_2024_ND_CP": 113,
}

SOURCE_POLICY = "tvpl_primary_with_attachment_preference"

OFFICIAL_DOCUMENTS = [
    {
        **doc,
        "expected_article_count": EXPECTED_ARTICLE_COUNTS.get(doc["id"]),
        "source_policy": SOURCE_POLICY,
    }
    for doc in tvpl_crawler.SEED_DOCUMENTS
]


def crawl_official_document(doc_info: dict, session: requests.Session) -> bool:
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
    attachments = find_attachment_links(soup, doc_info["url"])
    downloaded_files = []
    raw_text = ""
    raw_html = ""
    source_format = "html"

    # Prefer source attachments only when exposed. Otherwise keep the TVPL HTML path.
    for attachment_url in attachments:
        suffix = Path(attachment_url.split("?", 1)[0]).suffix.lower()
        saved_path = download_attachment(session, attachment_url, doc_info["id"], suffix)
        if not saved_path:
            continue

        downloaded_files.append(str(saved_path.relative_to(_ROOT)))
        extracted = extract_text_from_attachment(saved_path)
        if extracted:
            raw_text = extracted
            source_format = suffix.lstrip(".") or "attachment"
            break

    if not raw_text:
        raw_html = extract_best_html_content(soup, doc_info["url"])

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
    session = requests.Session()
    session.headers.update(HEADERS)

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
    return "unknown"


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
        if any(ext in haystack for ext in [".docx", ".pdf", ".doc"]):
            candidates.append(target)

    def rank(url: str) -> int:
        clean = url.lower().split("?", 1)[0]
        if clean.endswith(".docx"):
            return 0
        if clean.endswith(".pdf"):
            return 1
        if clean.endswith(".doc"):
            return 2
        return 3

    return sorted(dict.fromkeys(candidates), key=rank)


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
