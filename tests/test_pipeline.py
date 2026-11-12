import pytest

from src.config import NOT_FOUND_MESSAGE
from src.utils.errors import EmptyDocumentError, PDFExtractionError
from tests.conftest import FakeEmbedder, FakeLLM, make_pdf


def test_ingest_is_idempotent_and_does_not_re_embed(pipeline, pdf_bytes):
    r1 = pipeline.ingest("geo.pdf", pdf_bytes)
    assert r1.chunks_added == r1.chunks_total > 0 and not r1.already_indexed
    before = FakeEmbedder.calls
    r2 = pipeline.ingest("geo.pdf", pdf_bytes)
    assert r2.chunks_added == 0 and r2.already_indexed
    assert FakeEmbedder.calls == before  # no duplicate embeddings
    assert pipeline.list_documents()[0]["document"] == "geo.pdf"


def test_retrieval_returns_right_page_with_metadata(pipeline, pdf_bytes):
    pipeline.ingest("geo.pdf", pdf_bytes)
    hits = pipeline.retriever.retrieve("What is the capital of France?", top_k=2)
    assert hits[0].page == 1 and hits[0].document == "geo.pdf" and hits[0].score >= hits[1].score


def test_ask_produces_citation_linked_to_retrieved_chunk(pipeline, pdf_bytes):
    pipeline.ingest("geo.pdf", pdf_bytes)
    res = pipeline.ask("capital of France")
    assert res.found and res.citations
    retrieved_ids = {c.chunk_id for c in res.retrieved}
    assert all(c.chunk_id in retrieved_ids for c in res.citations)


def test_not_found_answer_has_no_citations(pipeline, pdf_bytes):
    pipeline.ingest("geo.pdf", pdf_bytes)
    pipeline.generator.llm = FakeLLM(NOT_FOUND_MESSAGE)
    res = pipeline.ask("quantum chromodynamics?")
    assert not res.found and res.citations == []


def test_invented_citation_is_discarded_with_warning(pipeline, pdf_bytes):
    pipeline.ingest("geo.pdf", pdf_bytes)
    pipeline.generator.llm = FakeLLM("Paris is big [1]. Something else [42].")
    res = pipeline.ask("capital of France")
    assert res.invalid_refs == [42] and all(c.ref != 42 for c in res.citations) and res.warnings


def test_empty_index_does_not_call_llm(pipeline):
    pipeline.generator.llm = FakeLLM("SHOULD NOT BE USED")
    res = pipeline.ask("anything")
    assert not res.found and pipeline.generator.llm.prompts == []


def test_document_filter(pipeline, pdf_bytes):
    a = pipeline.ingest("a.pdf", pdf_bytes)
    pipeline.ingest("b.pdf", make_pdf(["Completely different content about volcanoes and magma flows today."]))
    hits = pipeline.retriever.retrieve("volcanoes", top_k=5, doc_ids=[a.doc_id])
    assert hits and all(h.doc_id == a.doc_id for h in hits)


def test_bad_files_raise_friendly_errors(pipeline):
    with pytest.raises(PDFExtractionError):
        pipeline.ingest("x.pdf", b"nope")
    with pytest.raises(EmptyDocumentError):
        pipeline.ingest("scan.pdf", make_pdf([""]))


def test_delete_document(pipeline, pdf_bytes):
    r = pipeline.ingest("geo.pdf", pdf_bytes)
    pipeline.delete_document(r.doc_id)
    assert pipeline.list_documents() == []
