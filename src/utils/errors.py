"""Project-specific exceptions so the UI can show friendly messages."""


class VeriRAGError(Exception):
    """Base class for all expected, user-presentable errors."""


class PDFExtractionError(VeriRAGError):
    """The file is not a readable PDF (corrupt, encrypted, wrong type)."""


class EmptyDocumentError(VeriRAGError):
    """The PDF opened fine but has no extractable text (likely scanned images)."""


class MissingAPIKeyError(VeriRAGError):
    """GEMINI_API_KEY is not configured."""


class LLMError(VeriRAGError):
    """Gemini call failed (network, quota, blocked response...)."""


class VectorStoreError(VeriRAGError):
    """ChromaDB operation failed."""


class EmbeddingError(VeriRAGError):
    """Embedding model could not be loaded or run."""


class EvaluationError(VeriRAGError):
    """RAGAS evaluation could not be completed."""
