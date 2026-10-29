from pathlib import Path
from unittest.mock import MagicMock, patch

from bs4 import BeautifulSoup

from src.crawler import official_crawler as crawler
from src.crawler.clean_dataset import clean_html, clean_extracted_text, split_main_and_appendices
from src.legal_structure import article_matches


def test_source_line_wrapping_does_not_split_article_headings():
    text = clean_html('<p><b>Điều\n 1. Phạm vi\n điều chỉnh</b></p><p>1. Nội dung <span>pháp</span> luật.</p>')
    assert len(article_matches(text)) == 1
    assert "Điều 1. Phạm vi điều chỉnh" in text


def test_wrapped_appendix_references_are_not_appendix_headings():
    text = "Điều 1. Quy định\nTheo Mẫu số 03\nPhụ lục I ban hành kèm theo Nghị định này.\nĐiều 2. Hiệu lực\nÁp dụng.\nPhụ lục I\nMẫu số 01\nNội dung mẫu"
    main, appendices = split_main_and_appendices(text)
    assert "Điều 2" in main
    assert len(appendices) == 2


def test_quoted_appendix_stays_with_amendment():
    text = 'Điều 1. Sửa đổi\n“Nội dung\nPHỤ LỤC I\nQuy định sửa đổi.”\nĐiều 2. Hiệu lực\nÁp dụng.'
    main, appendices = split_main_and_appendices(text)
    assert main == text
    assert not appendices


def test_pdf_gazette_headers_do_not_pollute_body():
    result = clean_extracted_text("Điều 1. Quy định\nNội dung.\n12 CÔNG BÁO/Số 1 + 2/Ngày 1-1-2024\nTiếp theo.", "pdf")
    assert "CÔNG BÁO" not in result
    assert "Tiếp theo." in result


def test_attachment_format_query_is_detected():
    url = "https://congbao.chinhphu.vn/download?format=pdf"
    links = crawler.find_attachment_links(BeautifulSoup(f'<a href="{url}">Tải</a>', "html.parser"), "https://congbao.chinhphu.vn")
    assert links == [url]
    assert crawler.attachment_suffix(url) == ".pdf"
    assert crawler.attachment_suffix("https://g7.cdnchinhphu.vn/api/download/stream?file_name=law.docx") == ".docx"


def test_crawler_tls_uses_system_trust_and_keeps_verification():
    import requests
    from src.crawler.http import create_session
    session = create_session({})
    adapter = session.get_adapter("https://")
    request = requests.Request("GET", "https://congbao.chinhphu.vn").prepare()
    _, pool = adapter.build_connection_pool_key_attributes(request, True)
    assert pool["ssl_context"] is adapter._ssl_context
    assert pool["cert_reqs"] == "CERT_REQUIRED"
    assert session.verify is True
    session.close()


def test_content_must_have_its_own_document_number():
    assert crawler.matches_document_number("Số: 95/2024/NĐ-CP\nĐiều 1. Nội dung", "95/2024/NĐ-CP")
    assert not crawler.matches_document_number("Số: 96/2024/NĐ-CP\nCăn cứ 95/2024/NĐ-CP", "95/2024/NĐ-CP")


def test_crawler_uses_fallback_after_primary_failure():
    doc = {"id": "demo", "url": "https://primary.vn", "fallback_urls": ["https://fallback.vn"]}
    with patch.object(crawler, "_crawl_source", side_effect=[False, True]) as fetch:
        assert crawler.crawl_official_document(doc, MagicMock())
    assert fetch.call_args.args[0]["url"] == "https://fallback.vn"
