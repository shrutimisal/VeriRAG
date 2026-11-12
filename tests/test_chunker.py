import pytest

from src.ingestion.pdf_loader import load_pdf
from src.processing.chunker import chunk_document, chunk_text


def test_chunks_respect_size_and_overlap():
    text = "word " * 500
    chunks = chunk_text(text, chunk_size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    # overlap: end of chunk i shares text with start of chunk i+1
    assert chunks[0][-20:].strip().split()[-1] in chunks[1]


def test_short_text_single_chunk():
    assert chunk_text("hello world", 100, 10) == ["hello world"]


def test_always_terminates_with_large_overlap():
    assert len(chunk_text("a" * 1000, chunk_size=100, overlap=99)) > 1


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (100, -1)])
def test_invalid_params(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("abc", size, overlap)


def test_metadata_preserved(pdf_bytes):
    doc = load_pdf(pdf_bytes, "geo.pdf")
    chunks = chunk_document(doc, 80, 10)
    assert chunks and all(c.document == "geo.pdf" and c.doc_id == doc.doc_id for c in chunks)
    assert {c.page for c in chunks} == {1, 2}
    assert len({c.chunk_id for c in chunks}) == len(chunks)
