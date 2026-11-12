"""ChromaDB persistent vector store."""
from __future__ import annotations

from pathlib import Path

import chromadb

from src.embeddings.embedder import Embedder
from src.models import Chunk, RetrievedChunk
from src.utils.errors import EmbeddingError, VectorStoreError
from src.utils.logger import get_logger

log = get_logger(__name__)
_BATCH = 256


class ChromaStore:
    """Stores chunk text + embeddings + metadata (document, doc_id, page) on local disk."""

    def __init__(self, persist_dir: Path | str, collection_name: str = "verirag_chunks"):
        try:
            Path(persist_dir).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(persist_dir))
            # cosine space: distance = 1 - cosine_similarity
            self._col = self._client.get_or_create_collection(
                name=collection_name, metadata={"hnsw:space": "cosine"}
            )
        except Exception as exc:
            raise VectorStoreError(f"Could not open ChromaDB at '{persist_dir}': {exc}") from exc

    # ------------------------------------------------------------------ write
    def has_document(self, doc_id: str) -> bool:
        res = self._col.get(where={"doc_id": doc_id}, limit=1, include=[])
        return len(res["ids"]) > 0

    def add_chunks(self, chunks: list[Chunk], embedder: Embedder) -> int:
        """Index chunks, skipping ones already stored (no duplicate embeddings). Returns #new chunks."""
        if not chunks:
            return 0
        try:
            existing: set[str] = set()
            for i in range(0, len(chunks), 1000):
                ids = [c.chunk_id for c in chunks[i : i + 1000]]
                existing.update(self._col.get(ids=ids, include=[])["ids"])
            new = [c for c in chunks if c.chunk_id not in existing]
            for i in range(0, len(new), _BATCH):
                batch = new[i : i + _BATCH]
                vectors = embedder.embed_texts([c.text for c in batch])
                self._col.upsert(
                    ids=[c.chunk_id for c in batch],
                    embeddings=vectors,
                    documents=[c.text for c in batch],
                    metadatas=[
                        {
                            "doc_id": c.doc_id,
                            "document": c.document,
                            "page": c.page,
                            "chunk_index": c.chunk_index,
                        }
                        for c in batch
                    ],
                )
            return len(new)
        except (VectorStoreError, EmbeddingError):
            raise
        except Exception as exc:
            raise VectorStoreError(f"Failed to index chunks: {exc}") from exc

    def delete_document(self, doc_id: str) -> None:
        try:
            self._col.delete(where={"doc_id": doc_id})
        except Exception as exc:
            raise VectorStoreError(f"Failed to delete document: {exc}") from exc

    # ------------------------------------------------------------------- read
    def count(self) -> int:
        return self._col.count()

    def list_documents(self) -> list[dict]:
        """Summaries of indexed documents: doc_id, name, #chunks, #pages."""
        try:
            res = self._col.get(include=["metadatas"])
        except Exception as exc:
            raise VectorStoreError(f"Failed to list documents: {exc}") from exc
        docs: dict[str, dict] = {}
        for meta in res["metadatas"] or []:
            d = docs.setdefault(
                meta["doc_id"], {"doc_id": meta["doc_id"], "document": meta["document"], "chunks": 0, "pages": set()}
            )
            d["chunks"] += 1
            d["pages"].add(meta["page"])
        out = []
        for d in docs.values():
            d["pages"] = len(d["pages"])
            out.append(d)
        return sorted(out, key=lambda d: d["document"].lower())

    def query(
        self, query_embedding: list[float], top_k: int = 5, doc_ids: list[str] | None = None
    ) -> list[RetrievedChunk]:
        """Nearest-neighbour search, optionally restricted to specific documents."""
        total = self._col.count()
        if total == 0:
            return []
        kwargs: dict = {}
        if doc_ids:
            kwargs["where"] = {"doc_id": {"$in": list(doc_ids)}}
        try:
            res = self._col.query(
                query_embeddings=[query_embedding],
                n_results=max(1, min(top_k, total)),
                include=["documents", "metadatas", "distances"],
                **kwargs,
            )
        except Exception as exc:
            raise VectorStoreError(f"Vector search failed: {exc}") from exc

        hits: list[RetrievedChunk] = []
        for i, cid in enumerate(res["ids"][0]):
            meta = res["metadatas"][0][i]
            dist = float(res["distances"][0][i])
            hits.append(
                RetrievedChunk(
                    ref=i + 1,
                    chunk_id=cid,
                    doc_id=meta["doc_id"],
                    document=meta["document"],
                    page=int(meta["page"]),
                    text=res["documents"][0][i],
                    score=1.0 - dist,
                    distance=dist,
                )
            )
        return hits
