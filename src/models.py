"""Plain dataclasses passed between pipeline stages.

Keeping them in one file avoids circular imports and documents the data flow:
PageContent -> Chunk -> RetrievedChunk -> Citation / Claim -> ClaimVerdict -> AnswerResult
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A slice of one PDF page. Carries its origin so citations can never lose it."""
    chunk_id: str      # deterministic: "<doc_id>_p<page>_c<index>"  -> re-indexing is idempotent
    doc_id: str        # hash of the PDF bytes
    document: str      # human-readable file name
    page: int          # 1-based ORIGINAL PDF page number
    chunk_index: int   # position within the page
    text: str


@dataclass
class RetrievedChunk:
    """A chunk returned from the vector store with its similarity score."""
    ref: int           # 1-based number shown to the LLM as [ref]
    chunk_id: str
    doc_id: str
    document: str
    page: int
    text: str
    score: float       # cosine similarity (1 - cosine distance); higher = more similar
    distance: float


@dataclass
class Citation:
    ref: int
    document: str
    page: int
    passage: str
    chunk_id: str
    score: float


@dataclass
class Claim:
    id: int
    text: str
    refs: list[int]                              # valid refs cited by this claim
    passages: list[RetrievedChunk] = field(default_factory=list)  # evidence used for checking
    uncited: bool = False                        # True if the claim had no citation marker


@dataclass
class ClaimVerdict:
    claim: Claim
    label: str          # one of CLAIM_LABELS or "Unverified"
    explanation: str


@dataclass
class AnswerResult:
    question: str
    answer: str
    found: bool                       # False when the model said the docs don't contain the answer
    citations: list[Citation]
    retrieved: list[RetrievedChunk]
    invalid_refs: list[int] = field(default_factory=list)  # markers the model invented (dropped)
    warnings: list[str] = field(default_factory=list)
