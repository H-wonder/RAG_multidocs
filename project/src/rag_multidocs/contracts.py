"""Public Connector contract.

Core only depends on this module. Built-in and external connectors return the
same ``ConnectorResult``; adding a data source must not require changing core.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, TypedDict

from .model import Document


@dataclass(frozen=True)
class ConnectorConfig:
    name: str
    path: str
    options: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "path": self.path, "options": dict(self.options)}


@dataclass
class ConnectorIssue:
    code: str
    message: str
    source: str = ""

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message, "source": self.source}


@dataclass
class ConnectorResult:
    documents: list[Document] = field(default_factory=list)
    issues: list[ConnectorIssue] = field(default_factory=list)


class ConnectorError(RuntimeError):
    """Fatal connector error; the CLI converts it to a structured message."""

    def __init__(self, code: str, message: str, source: str = ""):
        self.code = code
        self.message = message
        self.source = source
        super().__init__(f"{code}: {message}")


class Connector(Protocol):
    name: str

    def load(self, config: ConnectorConfig) -> ConnectorResult: ...


__all__ = [
    "Connector",
    "ConnectorConfig",
    "ConnectorError",
    "ConnectorIssue",
    "ConnectorResult",
]


# Reranker contract v1. Only these data types, model.py and documented public
# services may be imported by external reranker plugins.
@dataclass(frozen=True)
class SearchHit:
    candidate_id: str
    text: str
    source: str = ""
    doc_id: str = ""
    score: float = 0.0
    semantic_score: float | None = None


@dataclass(frozen=True)
class RerankRequest:
    query: str
    candidates: tuple[SearchHit, ...]
    config: dict[str, Any] = field(default_factory=dict)
    contract_version: str = "1"


@dataclass(frozen=True)
class RankedCandidate:
    candidate_id: str
    rank: int
    score: float  # finite normalized relevance in [0,1], not a probability


@dataclass(frozen=True)
class RerankWarning:
    code: str
    message: str
    plugin: str = ""
    fallback: str = "rrf"


class Reranker(Protocol):
    """Function entry adapter:rank (callable), not a required plugin class."""

    def __call__(self, request: RerankRequest) -> list[RankedCandidate]: ...


__all__ += ["SearchHit", "RerankRequest", "RankedCandidate", "RerankWarning", "Reranker"]


class ConnectorAdapter(Protocol):
    """External function entry; Connector.load remains the legacy object API."""

    def __call__(self, payload: dict, context: dict | None = None) -> ConnectorResult: ...


class Block(TypedDict, total=False):
    """Typed view of Document.segments dicts, not a second runtime document model."""

    id: str
    type: str
    content: str
    reading_order: int
    page: int
    page_end: int
    bbox: list[float] | str
    ocr_mode: str
    origin: str
    needs_review: bool
    warnings: list[dict]


class Citation(TypedDict, total=False):
    index: int
    evidence_id: str
    chunk_id: str
    doc_id: str
    version: int | None
    source: str
    page: int
    page_end: int
    bbox: str
    block_ids: list[str]
    excerpts: list[str]
    fact_verified: bool


__all__ += ["ConnectorAdapter", "Block", "Citation"]
