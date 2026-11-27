"""Connector -> standard Document -> quality gate -> chunks -> optional publication.

Base Index backends keep their documented legacy path with an explicit warning;
only declared version-capable backends get publication/recovery guarantees.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..contracts import ConnectorConfig, ConnectorIssue
from ..model import Document, normalize_warning
from .batch import BatchIngestReport, collect_files
from .chunking import ChunkingOptions
from .structured import StructuredOptions, structured_chunks

_VOLATILE = {
    "latency",
    "latency_ms",
    "duration",
    "usage",
    "created_at",
    "temp_path",
    "request_id",
    "timestamp",
}


def stable_value(value):
    if isinstance(value, dict):
        return {k: stable_value(v) for k, v in value.items() if k not in _VOLATILE}
    if isinstance(value, (list, tuple)):
        return [stable_value(v) for v in value]
    return value


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()


def canonical_content(document):
    """Only content/structure fields consumed by the core; no runtime IR metadata."""
    from .structured import public_blocks

    fields = {
        "id",
        "type",
        "content",
        "text",
        "html",
        "page",
        "page_end",
        "bbox",
        "reading_order",
        "section_path",
        "level",
        "caption",
        "ocr_text",
        "description",
        "source",
    }
    blocks = []
    for b in public_blocks(document):
        block = {k: v for k, v in b.items() if k in fields}
        meta = b.get("meta", b.get("metadata", {}))
        stable_meta = {k: stable_value(meta[k]) for k in ["location", "table_data", "latex", "variables"] if k in meta}
        if stable_meta:
            block["meta"] = stable_meta
        blocks.append(block)
    return {
        "text": document.content,
        "blocks": blocks,
    }


def source_bytes(path):
    """Format-agnostic source change detection; opaque URI plugins opt out."""
    p = Path(path)
    if p.is_file():
        return hashlib.sha256(p.read_bytes()).hexdigest()
    if p.is_dir():
        return digest(
            [
                (str(f.relative_to(p)), hashlib.sha256(f.read_bytes()).hexdigest())
                for f in sorted(p.rglob("*"))
                if f.is_file()
            ]
        )
    return None


def quality(document, *, allow_partial=True, strict=False):
    ir = document.metadata.get("docparse_ir") or document.metadata.get("ir") or {}
    if str(document.metadata.get("status", ir.get("status", ""))).lower() in {"failed", "fatal"}:
        raise ValueError("PARSE_FATAL_STATUS")
    pages = ir.get("pages", [])
    failed = [
        p.get("number", 0)
        for p in pages
        if p.get("mode") == "failed"
        or any(w.get("code") == "page_failed" for w in p.get("warnings", []) if isinstance(w, dict))
    ]
    warnings = [
        normalize_warning(w) for w in document.metadata.get("warnings", ir.get("warnings", []))
    ]
    for page in pages:
        warnings.extend(
            {**normalize_warning(w), "page": page.get("number", 0)}
            for w in page.get("warnings", [])
        )
    if failed:
        warnings.append(
            {
                "code": "PARTIAL_PARSE",
                "message": "Missing parsed pages",
                "severity": "error",
                "pages": failed,
            }
        )
    if pages and len(failed) == len(pages):
        raise ValueError("PARSE_ALL_PAGES_FAILED")
    if failed and not allow_partial:
        raise ValueError("PARTIAL_PARSE_REJECTED")
    if strict and (
        document.metadata.get("needs_review") or any(w["severity"] == "error" for w in warnings)
    ):
        raise ValueError("QUALITY_GATE_REJECTED")
    return warnings


@dataclass
class IngestReport:
    connector: str
    path: str
    documents: list[dict[str, Any]] = field(default_factory=list)
    chunks: int = 0
    issues: list[ConnectorIssue] = field(default_factory=list)

    @property
    def ok(self):
        return bool(self.documents)

    def as_dict(self):
        return {
            "connector": self.connector,
            "path": self.path,
            "documents": self.documents,
            "chunks": self.chunks,
            "issues": [i.as_dict() for i in self.issues],
        }


class IngestService:
    def __init__(
        self,
        registry,
        index,
        chunk_options: ChunkingOptions | None = None,
        catalog=None,
        *,
        allow_partial=True,
        strict=False,
        near_duplicate_policy="report",
    ):
        self.registry, self.index = registry, index
        # Native PDFs often expose one block per visual line. Coalesce adjacent
        # short blocks with the existing section/page-aware packing logic.
        self.chunk_options = chunk_options or StructuredOptions(
            max_tokens=480, min_tokens=240, overlap_tokens=60
        )
        self.catalog = catalog or getattr(index, "version_catalog", None)
        # Keep the separately selected legacy keyword backend compatible.
        # The normal Milvus path never imports or creates a SQLite catalog.
        if self.catalog is None and getattr(index, "name", "") == "keyword":
            from .catalog import DocumentCatalog
            self.catalog = DocumentCatalog(index)
        if near_duplicate_policy not in {"report", "skip"}:
            raise ValueError("invalid near_duplicate_policy")
        self.allow_partial, self.strict = allow_partial, strict
        self.near_duplicate_policy = near_duplicate_policy

    def ingest_many(self, connector_name, paths, options=None):
        """Dispatch each distinct file through the existing single-input publication path."""
        if isinstance(paths, (str, Path)):
            paths = [paths]
        report = BatchIngestReport()
        auto = connector_name == "auto"
        routes = self.registry.file_routes() if auto else {}
        base = Path(self.registry.context.get("source_base", Path.cwd())).resolve()
        skipped_messages = {
            "UNSUPPORTED_TYPE": "没有 Connector 声明该文件后缀",
            "CONNECTOR_DISABLED": "支持该后缀的 Connector 均未启用",
            "CONNECTOR_DUPLICATE": "支持该后缀的 Connector 名称冲突",
        }
        for file, issue in collect_files(paths, report):
            name = None if auto else connector_name
            if auto and issue is None:
                name, code = routes.get(file.suffix.casefold(), (None, "UNSUPPORTED_TYPE"))
                if name is None:
                    issue = ConnectorIssue(code, skipped_messages[code], str(file))
            entry = {"path": str(file), "connector": name,
                     "status": "skipped", "document_count": 0, "chunks": 0,
                     "documents": [], "issues": []}
            report.files.append(entry)
            if issue:
                entry["issues"] = [issue.as_dict()]
                continue

            file_options = dict(options or {})
            # Use a fixed base, regardless of which ancestor the caller scanned,
            # so separate folders with the same filename keep separate versions.
            if file.parent != base and not file_options.get("source_namespace"):
                try:
                    parent = Path(os.path.relpath(file.parent, base))
                except ValueError:  # Windows paths on separate drives
                    parent = file.parent
                file_options["source_namespace"] = parent.as_posix()
            try:
                if not file.is_file():
                    raise ValueError("不是普通文件，或文件在导入前已移除")
                result = self._ingest_one(name, str(file), file_options)
            except Exception as exc:
                entry["status"] = "failed"
                entry["issues"] = [ConnectorIssue(
                    "INGEST_FAILED", f"文件导入失败: {type(exc).__name__}", str(file)
                ).as_dict()]
                continue
            issues = [issue.as_dict() for issue in result.issues]
            if result.documents:
                status = "partial" if issues else "succeeded"
            else:
                status = "failed"
                if not issues:
                    issues = [ConnectorIssue("EMPTY_SOURCE", "Connector 没有返回可入库文档", str(file)).as_dict()]
            entry.update(status=status, documents=result.documents, document_count=len(result.documents),
                         chunks=result.chunks, issues=issues)
        return report

    def ingest(self, connector_name, path, options=None):
        if connector_name == "auto":
            return self.ingest_many(connector_name, path, options)
        if isinstance(path, (list, tuple)):
            if len(path) != 1:
                return self.ingest_many(connector_name, path, options)
            path = path[0]
        return self._ingest_one(connector_name, path, options)

    def _ingest_one(self, connector_name, path, options=None):
        options = dict(options or {})
        report = IngestReport(connector_name, path)
        # A source cache is optional; unavailable files still go through the Connector
        # so public FILE_NOT_FOUND/READ_ERROR behavior remains unchanged.
        try:
            raw_hash = source_bytes(path)
        except OSError:
            raw_hash = None
        plugin_signature = []
        for info in self.registry.plugin_infos():
            if info.name == connector_name:
                plugin_signature = [
                    info.version,
                    info.contract_version,
                    info.config,
                    info.requires,
                    [
                        (p.name, hashlib.sha256(p.read_bytes()).hexdigest())
                        for p in sorted(info.path.rglob("*.py"))
                    ],
                ]
        parser_repo = self.registry.context.get("settings", {}).get("docparse_repo")
        parser_signature = []
        if parser_repo:
            parser_root = Path(parser_repo) / "docparse"
            parser_signature = [(str(p.relative_to(parser_root)), hashlib.sha256(p.read_bytes()).hexdigest())
                                for p in sorted(parser_root.rglob("*.py"))]
        processing = {
            "revision": "structured-v3.3-complete-source-text",
            "chunking": asdict(self.chunk_options),
            "options": stable_value(options),
            "plugin": plugin_signature,
            "parser_code": parser_signature,
            "quality": [self.allow_partial, self.strict, self.near_duplicate_policy],
            "parser_settings": {
                k: self.registry.context.get("settings", {}).get(k)
                for k in (
                    "docparse_mode",
                    "parser_revision",
                    "ocr_model",
                    "layout_model",
                    "dpi",
                    "max_pages",
                )
            },
            "processing_identity": stable_value(
                self.registry.context.get("processing_identity", {})
            ),
        }
        key = digest([connector_name, str(Path(path).resolve())])
        signature = digest([raw_hash, processing])
        # Do not let a cache bypass disabling a plugin.
        enabled = (
            getattr(self.registry, "enabled", None) is None
            or connector_name in self.registry.enabled
        )
        matching = [i for i in self.registry.plugin_infos() if i.name == connector_name]
        if matching:
            enabled = enabled and len(matching) == 1 and matching[0].enabled
        cache = self.catalog or self.index
        cached_ingest = getattr(cache, "cached_ingest", None)
        if cached_ingest and raw_hash and enabled:
            cached = cached_ingest(key, signature)
            if cached:
                report.documents = cached
                return report
        result = self.registry.load(connector_name, ConnectorConfig(connector_name, path, options))
        report.issues.extend(result.issues)
        for document in result.documents:
            try:
                document.metadata["content_sha256"] = digest(canonical_content(document))
                # A single input file is not necessarily a single logical document (e.g. JSONL).
                document.metadata["raw_sha256"] = (
                    document.metadata.get("file_sha256")
                    or document.metadata.get("raw_sha256")
                    or (raw_hash if Path(path).is_file() else document.metadata.get("sha256"))
                )
                warnings = quality(document, allow_partial=self.allow_partial, strict=self.strict)
                document.metadata["warnings"] = warnings
                if options.get("source_namespace"):
                    from ..model import stable_doc_id

                    document.source = (
                        str(options["source_namespace"]).strip("/") + "/" + document.source
                    )
                    document.doc_id = stable_doc_id(document.source)
                chunks = self._chunk(document)
                # Structural parsing can discover issues after the first quality
                # gate. Surface these before publication, including strict mode.
                seen_warnings = {digest(w) for w in warnings}
                for chunk in chunks:
                    for value in chunk.metadata.get("warnings", []):
                        warning = normalize_warning(value)
                        if digest(warning) not in seen_warnings:
                            warnings.append(warning)
                            seen_warnings.add(digest(warning))
                document.metadata["needs_review"] = bool(document.metadata.get("needs_review")) or any(
                    c.metadata.get("needs_review") for c in chunks
                )
                if self.strict and (document.metadata["needs_review"] or any(w["severity"] == "error" for w in warnings)):
                    raise ValueError("QUALITY_GATE_REJECTED")
                if not chunks:
                    report.issues.append(
                        ConnectorIssue(
                            "EMPTY_DOCUMENT",
                            "Parsed document has no indexable content",
                            document.source,
                        )
                    )
                    continue
                if raw_hash and source_bytes(path) != raw_hash:
                    raise ValueError("SOURCE_CHANGED_DURING_INGEST")
                fingerprint = digest(
                    [
                        processing,
                        canonical_content(document),
                        document.metadata["content_sha256"],
                    ]
                )
                near = []
                analysis_warning = None
                if self.catalog:
                    try:
                        near = self.catalog.near_duplicates(document)
                    except Exception:
                        analysis_warning = {
                            "code": "NEAR_DUPLICATE_CHECK_FAILED",
                            "message": "Optional near-duplicate analysis unavailable",
                            "severity": "warning",
                        }
                    if near and self.near_duplicate_policy == "skip":
                        report.issues.append(
                            ConnectorIssue(
                                "NEAR_DUPLICATE_SKIPPED",
                                "Explicit skip policy matched another document",
                                document.source,
                            )
                        )
                        continue
                    count, version = self.catalog.publish(document, chunks, fingerprint)
                else:
                    document.metadata.update(processing_fingerprint=fingerprint,
                                             ingest_signature=signature, ingest_key=key)
                    if not getattr(self.index, "replaces_document", False):
                        self.index.remove_doc(document.doc_id)
                    count = self.index.add(chunks, document=document)
                    self.index.persist()
                    version = None
                    if not getattr(self.index, "replaces_document", False):
                        warnings.append({"code": "LEGACY_INDEX_NON_ATOMIC",
                                         "message": "Base Index path has no staged-publication guarantee",
                                         "severity": "warning"})
                # Publication is committed. Auxiliary diagnostics must not misreport rollback.
                if analysis_warning:
                    warnings.append(analysis_warning)
                report.chunks += count
                report.documents.append(
                    {
                        "doc_id": document.doc_id,
                        "source": document.source,
                        "title": document.title,
                        "chunks": len(chunks),
                        "version": version,
                        "status": "indexed" if count else "unchanged",
                        "needs_review": bool(document.metadata.get("needs_review")),
                        "warnings": warnings,
                        "quality_status": "partial_or_review"
                        if warnings or document.metadata.get("needs_review")
                        else "complete",
                        "near_duplicate_candidates": near,
                        "raw_sha256": document.metadata.get("raw_sha256"),
                        "content_sha256": document.metadata["content_sha256"],
                        "processing_fingerprint": fingerprint,
                    }
                )
            except Exception as exc:
                code = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
                report.issues.append(
                    ConnectorIssue(
                        "INGEST_FAILED",
                        f"Publication failed: {code[:100]}; prior active version retained on version-capable indexes",
                        document.source,
                    )
                )
        if self.catalog and raw_hash and report.documents and not report.issues:
            try:
                self.catalog.remember_ingest(key, signature, report.documents)
            except Exception:
                for d in report.documents:
                    d["warnings"].append(
                        {
                            "code": "INGEST_CACHE_FAILED",
                            "message": "Published successfully; parse cache unavailable",
                            "severity": "warning",
                        }
                    )
        return report

    def _chunk(self, document: Document):
        from .chunking import estimate_tokens

        embedder = getattr(self.index, "embedder", None)
        count = getattr(embedder, "count_tokens", estimate_tokens)
        limit = getattr(embedder, "max_input_tokens", None)
        return structured_chunks(document, self.chunk_options, count, limit)
