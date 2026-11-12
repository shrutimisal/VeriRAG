"""Sentence-Transformers wrapper (local, free, no API key)."""
from __future__ import annotations

from src.utils.errors import EmbeddingError
from src.utils.logger import get_logger

log = get_logger(__name__)


class Embedder:
    """Lazy-loading embedder. The model is downloaded/loaded on first use only."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                log.info("Loading embedding model %s", self.model_name)
                self._model = SentenceTransformer(self.model_name)
            except Exception as exc:
                raise EmbeddingError(
                    f"Could not load embedding model '{self.model_name}'. "
                    f"Check your internet connection for the first download. ({exc})"
                ) from exc
        return self._model

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Embed many texts. Vectors are L2-normalised so cosine similarity == dot product."""
        if not texts:
            return []
        try:
            vectors = self.model.encode(
                texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
            )
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"Embedding failed: {exc}") from exc
        return [v.tolist() for v in vectors]

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query])[0]
