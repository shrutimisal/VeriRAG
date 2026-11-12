"""Split an answer into individual claims (sentences) and attach each one's cited passages."""
from __future__ import annotations

import re

from src.models import Claim, RetrievedChunk
from src.utils.text import extract_refs, strip_markers

# Sentence boundary: . ! ? (+ optional trailing [n] markers) followed by whitespace and a new sentence start.
# Requiring whitespace after the punctuation keeps decimals like "28.4" intact.
_BOUNDARY = re.compile(r"([.!?])((?:\s*\[\d+(?:\s*,\s*\d+)*\])*)\s+(?=[A-Z0-9\"“(\[])")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for line in text.splitlines():
        line = _BULLET.sub("", line).strip()
        if not line:
            continue
        pos = 0
        for m in _BOUNDARY.finditer(line):
            sentences.append(line[pos : m.end(2)].strip())
            pos = m.end()
        tail = line[pos:].strip()
        if tail:
            sentences.append(tail)
    return sentences


def split_claims(answer: str, retrieved: list[RetrievedChunk]) -> list[Claim]:
    """Create Claim objects. Evidence = the passages the claim cites; if it cites nothing,
    fall back to all retrieved passages and flag it `uncited`."""
    by_ref = {c.ref: c for c in retrieved}
    claims: list[Claim] = []
    for sentence in split_sentences(answer):
        text = strip_markers(sentence)
        if len(re.findall(r"[A-Za-z]{2,}", text)) < 3:  # skip fragments / headings
            continue
        refs = [r for r in extract_refs(sentence) if r in by_ref]
        passages = [by_ref[r] for r in refs]
        uncited = not passages
        if uncited:
            passages = list(retrieved)
        claims.append(Claim(id=len(claims) + 1, text=text, refs=refs, passages=passages, uncited=uncited))
    return claims
