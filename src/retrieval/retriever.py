"""Semantic retrieval: question -> embedding -> nearest chunks."""
from __future__ import annotations

from src.embeddings.embedder import Embedder
from src.models import RetrievedChunk
from src.vectorstore.chroma_store import ChromaStore


class Retriever:
    def __init__(self, store: ChromaStore, embedder: Embedder):
        self.store = store
        self.embedder = embedder

    def retrieve(
        self, question: str, top_k: int = 5, doc_ids: list[str] | None = None
    ) -> list[RetrievedChunk]:
        """Return the top_k most similar chunks (ref numbers 1..k, best first)."""
        question = question.strip()
        if not question:
            return []
        return self.store.query(self.embedder.embed_query(question), top_k=top_k, doc_ids=doc_ids)
