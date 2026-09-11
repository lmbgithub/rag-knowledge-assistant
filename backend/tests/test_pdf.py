from __future__ import annotations

import pytest
from pdfs import make_pdf, simple_pdf

from rag_assistant.errors import BadRequest, UnprocessableDocument
from rag_assistant.pdf import (
    MIN_CHARS_PER_DOC,
    check_extraction,
    extract,
    looks_like_pdf,
    pages_from_texts,
)

LONG = "The migration to PostgreSQL completed in March after two false starts."


def test_magic_bytes_are_checked_not_the_extension():
    assert looks_like_pdf(b"%PDF-1.7\n...")
    assert not looks_like_pdf(b"PK\x03\x04")  # a DOCX renamed to .pdf
    assert not looks_like_pdf(b"")


def test_a_non_pdf_is_a_400_not_a_422():
    with pytest.raises(BadRequest):
        extract(b"PK\x03\x04 this is a zip", filename="cv.pdf")


def test_pages_are_numbered_from_one():
    extraction = pages_from_texts(["first", "second"])
    assert [p.number for p in extraction.pages] == [1, 2]


def test_blank_pages_are_dropped_but_do_not_renumber_the_rest():
    extraction = pages_from_texts(["first", "   ", "third"])
    assert [(p.number, p.text) for p in extraction.pages] == [(1, "first"), (3, "third")]


def test_page_text_is_stripped():
    assert pages_from_texts(["  hello  "]).pages[0].text == "hello"


def test_total_chars_counts_only_kept_pages():
    assert pages_from_texts(["abc", "   "]).total_chars == 3


def test_a_document_under_the_character_floor_is_a_scan():
    extraction = pages_from_texts(["7"])
    with pytest.raises(UnprocessableDocument, match="no text layer"):
        check_extraction(extraction, filename="scan.pdf")


def test_the_floor_is_over_the_whole_document_not_per_page():
    # Many nearly-empty pages that add up are a real document, not a scan.
    extraction = pages_from_texts(["abcd"] * 20)
    assert extraction.total_chars > MIN_CHARS_PER_DOC
    assert len(check_extraction(extraction, filename="ok.pdf")) == 20


def test_extract_reads_a_real_pdf():
    pages = extract(simple_pdf(LONG, "A second page about caching and Redis."), filename="p.pdf")
    assert len(pages) == 2
    assert "PostgreSQL" in pages[0].text
    assert "Redis" in pages[1].text


def test_an_encrypted_pdf_says_so_rather_than_blaming_the_scanner():
    data = make_pdf([[LONG]], encrypted=True)
    with pytest.raises(UnprocessableDocument, match="password-protected"):
        extract(data, filename="locked.pdf")


def test_a_truncated_pdf_is_unprocessable_not_a_crash():
    data = simple_pdf(LONG)[:120]
    with pytest.raises(UnprocessableDocument):
        extract(data, filename="broken.pdf")


def test_an_image_only_pdf_is_reported_as_a_scan():
    with pytest.raises(UnprocessableDocument, match="OCR"):
        extract(make_pdf([[]]), filename="scan.pdf")


def test_any_parser_failure_is_a_422_not_an_escaping_exception(monkeypatch):
    """The catch used to name pypdf's exception types and missed one.

    `DependencyError` — raised for an AES-encrypted file when `cryptography`
    is absent — is not a subclass of any of them, so it escaped this module's
    whole contract as an uncaught traceback.
    """

    from pypdf.errors import DependencyError

    class Exploding:
        def __init__(self, *_: object) -> None:
            raise DependencyError("cryptography is required for AES algorithms")

    monkeypatch.setattr("pypdf.PdfReader", Exploding)
    with pytest.raises(UnprocessableDocument, match="could not be parsed"):
        extract(simple_pdf(LONG), filename="aes.pdf")
