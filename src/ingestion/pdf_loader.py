"""PDF text extraction with PyMuPDF, page by page."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import pymupdf  # PyMuPDF (import name "pymupdf"; legacy alias "fitz" also exists)

from src.utils.errors import EmptyDocumentError, PDFExtractionError
from src.utils.text import normalize_whitespace

# Pages with fewer characters than this are treated as "no text" (cover images, scans...).
MIN_PAGE_CHARS = 20


@dataclass
class PageContent:
    page_number: int  # 1-based, matches the page number a reader sees in a PDF viewer
    text: str


@dataclass
class LoadedDocument:
    doc_id: str
    name: str
    total_pages: int
    pages: list[PageContent]
    empty_pages: list[int] = field(default_factory=list)


def compute_doc_id(data: bytes) -> str:
    """Stable ID from file content: the same PDF uploaded twice gets the same ID."""
    return hashlib.sha256(data).hexdigest()[:16]


def load_pdf(data: bytes, name: str) -> LoadedDocument:
    """Extract text from PDF bytes.

    Raises:
        PDFExtractionError: empty file, corrupt file, or password-protected.
        EmptyDocumentError: valid PDF but no extractable text (probably scanned).
    """
    if not data:
        raise PDFExtractionError(f"'{name}' is empty (0 bytes).")
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises several different types for bad files
        raise PDFExtractionError(f"'{name}' is not a valid PDF: {exc}") from exc

    try:
        if doc.needs_pass:
            raise PDFExtractionError(f"'{name}' is password-protected and cannot be read.")
        if doc.page_count == 0:
            raise PDFExtractionError(f"'{name}' has no pages.")

        pages: list[PageContent] = []
        empty: list[int] = []
        for index in range(doc.page_count):
            try:
                text = normalize_whitespace(doc[index].get_text("text"))
            except Exception:  # a single broken page should not kill the whole document
                text = ""
            if len(text) < MIN_PAGE_CHARS:
                empty.append(index + 1)
            else:
                pages.append(PageContent(page_number=index + 1, text=text))
        total = doc.page_count
    finally:
        doc.close()

    if not pages:
        raise EmptyDocumentError(
            f"'{name}' has no extractable text. It is probably a scanned PDF (images only). "
            "OCR is not supported in this MVP."
        )
    return LoadedDocument(doc_id=compute_doc_id(data), name=name, total_pages=total, pages=pages, empty_pages=empty)
