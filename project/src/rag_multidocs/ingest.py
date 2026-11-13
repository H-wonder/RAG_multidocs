"""Ingest orchestration: Connector -> chunks -> index."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .chunking import ChunkingOptions, chunk_docparse_ir, chunk_document
from .contracts import ConnectorConfig, ConnectorIssue
from .index import KeywordIndex
from .model import Document
from .registry import ConnectorRegistry


@dataclass
class IngestReport:
    connector: str
    path: str
    documents: list[dict[str, Any]] = field(default_factory=list)
    chunks: int = 0
    issues: list[ConnectorIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.documents)

    def as_dict(self) -> dict[str, Any]:
        return {
            "connector": self.connector,
            "path": self.path,
            "documents": self.documents,
            "chunks": self.chunks,
            "issues": [issue.as_dict() for issue in self.issues],
        }


class IngestService:
    def __init__(
        self,
        registry: ConnectorRegistry,
        index: KeywordIndex,
        chunk_options: ChunkingOptions | None = None,
    ):
        self.registry = registry
        self.index = index
        self.chunk_options = chunk_options or ChunkingOptions()

    def ingest(
        self,
        connector_name: str,
        path: str,
        options: dict[str, Any] | None = None,
    ) -> IngestReport:
        config = ConnectorConfig(
            name=connector_name,
            path=path,
            options=dict(options or {}),
        )
        result = self.registry.load(connector_name, config)
        report = IngestReport(
            connector=connector_name,
            path=path,
            issues=list(result.issues),
        )

        for document in result.documents:
            chunks = self._chunk(document)
            if not chunks:
                report.issues.append(
                    ConnectorIssue(
                        "EMPTY_DOCUMENT",
                        "解析成功但没有可索引内容",
                        document.source,
                    )
                )
                continue

            # 先删后插：同一 doc_id 重跑不会留下旧块
            self.index.remove_doc(document.doc_id)
            self.index.add(chunks, document)
            report.chunks += len(chunks)
            report.documents.append(
                {
                    "doc_id": document.doc_id,
                    "source": document.source,
                    "title": document.title,
                    "chunks": len(chunks),
                }
            )

        self.index.persist()
        return report

    def _chunk(self, document: Document):
        if document.metadata.get("docparse_ir"):
            return chunk_docparse_ir(
                document.metadata["docparse_ir"], document, self.chunk_options
            )
        return chunk_document(document, self.chunk_options)


__all__ = ["IngestReport", "IngestService"]
