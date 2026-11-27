"""RAG Multidocs: plugin-based document ingestion and retrieval."""

__version__ = "0.1.0"

from .model import Chunk, Document
from .core import RAGCore

__all__ = ["Chunk", "Document", "RAGCore", "__version__"]
