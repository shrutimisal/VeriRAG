"""Text helpers shared by citation parsing and claim splitting."""
from __future__ import annotations

import re

# Matches [1], [2,3], [1, 2]. Individual markers like [1][2] match separately.
MARKER_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace into single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def extract_refs(text: str) -> list[int]:
    """Return citation numbers found in `text`, in order of first appearance, de-duplicated."""
    seen: list[int] = []
    for match in MARKER_RE.finditer(text):
        for num in match.group(1).split(","):
            n = int(num.strip())
            if n not in seen:
                seen.append(n)
    return seen


def strip_markers(text: str) -> str:
    """Remove citation markers and tidy leftover spacing/punctuation."""
    cleaned = MARKER_RE.sub("", text)
    cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)
    return normalize_whitespace(cleaned)
