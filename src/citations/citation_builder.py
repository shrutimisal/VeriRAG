"""Turn [n] markers in an answer into real citations backed by retrieved chunks."""
from __future__ import annotations

from src.config import NOT_FOUND_MESSAGE
from src.models import Citation, RetrievedChunk
from src.utils.text import extract_refs


def is_not_found(answer: str) -> bool:
    """True when the model declined to answer from the documents."""
    return NOT_FOUND_MESSAGE.lower() in answer.lower()


def build_citations(answer: str, retrieved: list[RetrievedChunk]) -> tuple[list[Citation], list[int]]:
    """Map markers to chunks.

    Returns (citations, invalid_refs). A marker pointing to a passage number that was never
    retrieved is reported in `invalid_refs` and produces NO citation, so citations can only
    ever reference real retrieved chunks.
    """
    by_ref = {c.ref: c for c in retrieved}
    citations: list[Citation] = []
    invalid: list[int] = []
    for ref in extract_refs(answer):
        chunk = by_ref.get(ref)
        if chunk is None:
            invalid.append(ref)
            continue
        citations.append(
            Citation(
                ref=ref, document=chunk.document, page=chunk.page,
                passage=chunk.text, chunk_id=chunk.chunk_id, score=chunk.score,
            )
        )
    return citations, invalid
