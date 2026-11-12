"""High-level orchestration: ingest PDFs, answer questions, verify citations."""
from __future__ import annotations

from dataclasses import dataclass

from src.citations.citation_builder import build_citations, is_not_found
from src.citations.claim_splitter import split_claims
from src.citations.verifier import CitationVerifier
from src.config import NOT_FOUND_MESSAGE, Settings, get_settings
from src.embeddings.embedder import Embedder
from src.generation.generator import AnswerGenerator
from src.generation.llm_client import GeminiClient
from src.ingestion.pdf_loader import load_pdf
from src.models import AnswerResult, ClaimVerdict
from src.processing.chunker import chunk_document
from src.retrieval.retriever import Retriever
from src.utils.logger import get_logger
from src.vectorstore.chroma_store import ChromaStore

log = get_logger(__name__)


@dataclass
class IngestReport:
    doc_id: str
    name: str
    total_pages: int
    empty_pages: list[int]
    chunks_total: int
    chunks_added: int
    already_indexed: bool


class VeriRAGPipeline:
    """Owns the embedder, vector store and Gemini client. Dependencies can be injected for tests."""

    def __init__(self, settings: Settings | None = None, embedder=None, store=None, llm=None):
        self.settings = settings or get_settings()
        self.embedder = embedder or Embedder(self.settings.embedding_model)
        self.store = store or ChromaStore(self.settings.chroma_dir)
        self.llm = llm or GeminiClient(self.settings.gemini_api_key, self.settings.gemini_model)
        self.retriever = Retriever(self.store, self.embedder)
        self.generator = AnswerGenerator(self.llm)
        self.verifier = CitationVerifier(self.llm)

    # --------------------------------------------------------------- config
    def set_api_key(self, key: str) -> None:
        """Override the API key at runtime (e.g. pasted in the UI sidebar)."""
        self.llm.api_key = key.strip()
        self.llm._client = None

    @property
    def has_api_key(self) -> bool:
        return bool(self.llm.api_key)

    # ------------------------------------------------------------ ingestion
    def ingest(self, name: str, data: bytes, chunk_size: int | None = None, overlap: int | None = None) -> IngestReport:
        """PDF bytes -> pages -> chunks -> embeddings -> ChromaDB. Idempotent per file content + chunk settings."""
        doc = load_pdf(data, name)  # raises PDFExtractionError / EmptyDocumentError
        chunks = chunk_document(doc, chunk_size or self.settings.chunk_size, self.settings.chunk_overlap if overlap is None else overlap)
        already = self.store.has_document(doc.doc_id)
        added = self.store.add_chunks(chunks, self.embedder)
        self._save_copy(doc.doc_id, data)
        return IngestReport(doc.doc_id, name, doc.total_pages, doc.empty_pages, len(chunks), added, already and added == 0)

    def _save_copy(self, doc_id: str, data: bytes) -> None:
        try:
            self.settings.documents_dir.mkdir(parents=True, exist_ok=True)
            (self.settings.documents_dir / f"{doc_id}.pdf").write_bytes(data)
        except OSError as exc:  # non-fatal: the index is what matters
            log.warning("Could not save PDF copy: %s", exc)

    def list_documents(self) -> list[dict]:
        return self.store.list_documents()

    def delete_document(self, doc_id: str) -> None:
        self.store.delete_document(doc_id)
        try:
            (self.settings.documents_dir / f"{doc_id}.pdf").unlink(missing_ok=True)
        except OSError:
            pass

    # --------------------------------------------------------------- Q & A
    def ask(self, question: str, doc_ids: list[str] | None = None, top_k: int | None = None) -> AnswerResult:
        """Retrieve -> generate -> build citations. Raises LLMError / MissingAPIKeyError on LLM problems."""
        retrieved = self.retriever.retrieve(question, top_k or self.settings.top_k, doc_ids)
        answer = self.generator.generate(question, retrieved)
        found = bool(retrieved) and not is_not_found(answer)
        citations, invalid = build_citations(answer, retrieved) if found else ([], [])
        warnings: list[str] = []
        if not retrieved:
            warnings.append("No documents are indexed (or none match the selected filter).")
        if found and not citations:
            warnings.append("The model gave an answer without any valid citation. Treat it with caution.")
        if invalid:
            warnings.append(f"The model referenced non-existent passages {invalid}; those citations were discarded.")
        return AnswerResult(question, answer, found, citations, retrieved, invalid, warnings)

    def verify(self, result: AnswerResult) -> list[ClaimVerdict]:
        """Claim-level verification of an answer against its cited passages."""
        if not result.found:
            return []
        claims = split_claims(result.answer, result.retrieved)
        return self.verifier.verify(claims)
