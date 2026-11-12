"""Shared fixtures: a fake embedder and fake LLM so tests need no model download or API key."""
import hashlib
import sys
from pathlib import Path

import pymupdf
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Settings  # noqa: E402


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder (dim 64) - similar words => similar vectors."""
    DIM = 64
    calls = 0

    def embed_texts(self, texts, batch_size=32):
        FakeEmbedder.calls += len(texts)
        out = []
        for t in texts:
            v = [0.0] * self.DIM
            for w in t.lower().split():
                w = w.strip(".,;:!?()[]\"'")
                if w:
                    v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.DIM] += 1.0
            norm = sum(x * x for x in v) ** 0.5 or 1.0
            out.append([x / norm for x in v])
        return out

    def embed_query(self, q):
        return self.embed_texts([q])[0]


class FakeLLM:
    """Returns canned text; records prompts."""
    def __init__(self, reply="Paris is the capital of France [1]."):
        self.reply, self.prompts, self.api_key, self._client = reply, [], "fake", None

    def generate(self, prompt, system=None, json_mode=False):
        self.prompts.append(prompt)
        return self.reply


def make_pdf(pages: list[str]) -> bytes:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_textbox(pymupdf.Rect(50, 50, 550, 780), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def pdf_bytes():
    return make_pdf([
        "Paris is the capital of France. It is known for the Eiffel Tower and many museums around the city.",
        "The Amazon rainforest covers much of northwestern Brazil and extends into Colombia and Peru regions.",
    ])


@pytest.fixture
def pipeline(tmp_path):
    from src.pipeline import VeriRAGPipeline
    settings = Settings("fake", "fake-model", "fake-emb", 300, 50, 3, tmp_path / "chroma", tmp_path / "docs", tmp_path / "eval", 1)
    return VeriRAGPipeline(settings=settings, embedder=FakeEmbedder(), llm=FakeLLM())
