"""Split page text into overlapping chunks while keeping source metadata."""
from __future__ import annotations

from src.ingestion.pdf_loader import LoadedDocument
from src.models import Chunk

_BREAK_MARKERS = (". ", "? ", "! ", "; ", ", ", " ")  # preferred split points, best first


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Sliding-window character chunker that prefers to cut at sentence/word boundaries.

    Args:
        chunk_size: max characters per chunk.
        overlap: characters shared between consecutive chunks (keeps context across cuts).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    text = text.strip()
    n = len(text)
    chunks: list[str] = []
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            # Only search the back half of the window (and past the overlap) so we always advance.
            lo = start + max(chunk_size // 2, overlap + 1)
            if lo < end:
                window = text[lo:end]
                for marker in _BREAK_MARKERS:
                    pos = window.rfind(marker)
                    if pos != -1:
                        end = lo + pos + len(marker)
                        break
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_document(doc: LoadedDocument, chunk_size: int = 1000, overlap: int = 200) -> list[Chunk]:
    """Chunk every page separately so each chunk maps to exactly one original page."""
    result: list[Chunk] = []
    for page in doc.pages:
        for idx, piece in enumerate(chunk_text(page.text, chunk_size, overlap)):
            result.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}_p{page.page_number}_c{idx}",
                    doc_id=doc.doc_id,
                    document=doc.name,
                    page=page.page_number,
                    chunk_index=idx,
                    text=piece,
                )
            )
    return result
