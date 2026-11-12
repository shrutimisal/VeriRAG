import pytest

from src.ingestion.pdf_loader import load_pdf
from src.utils.errors import EmptyDocumentError, PDFExtractionError
from tests.conftest import make_pdf


def test_pages_and_numbers_preserved(pdf_bytes):
    doc = load_pdf(pdf_bytes, "geo.pdf")
    assert doc.name == "geo.pdf" and doc.total_pages == 2
    assert [p.page_number for p in doc.pages] == [1, 2]
    assert "Eiffel" in doc.pages[0].text


def test_invalid_pdf_raises():
    with pytest.raises(PDFExtractionError):
        load_pdf(b"this is not a pdf", "bad.pdf")


def test_empty_bytes_raises():
    with pytest.raises(PDFExtractionError):
        load_pdf(b"", "empty.pdf")


def test_scanned_like_pdf_raises_empty_document():
    with pytest.raises(EmptyDocumentError):
        load_pdf(make_pdf(["", ""]), "scan.pdf")


def test_blank_page_is_skipped_but_page_numbers_stay_original():
    data = make_pdf(["", "This page has plenty of readable text in it."])
    doc = load_pdf(data, "x.pdf")
    assert doc.empty_pages == [1] and doc.pages[0].page_number == 2
