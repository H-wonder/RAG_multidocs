"""Independent Markdown directory connector.

This adapter is loaded by the core registry from ``manifest.json``. It only
depends on the public ``rag_multidocs.model`` / ``rag_multidocs.contracts``
modules; it never imports core ingestion or CLI internals.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    from rag_multidocs.contracts import (
        ConnectorError,
        ConnectorIssue,
        ConnectorResult,
    )
    from rag_multidocs.model import Document
except ImportError:  # allow running from a source checkout without installing
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from rag_multidocs.contracts import (
        ConnectorError,
        ConnectorIssue,
        ConnectorResult,
    )
    from rag_multidocs.model import Document

_HEADING = re.compile(r"^\s*#\s+(.+?)\s*$", re.MULTILINE)


def _title(text: str, fallback: str) -> str:
    match = _HEADING.search(text)
    return match.group(1).strip() if match else fallback


def load(payload: dict, context: dict | None = None) -> ConnectorResult:
    """Read a Markdown file or a directory tree of .md/.markdown files."""
    raw_path = payload.get("path") or (payload.get("options") or {}).get("path")
    if not raw_path:
        raise ConnectorError("INVALID_CONFIG", "缺少 path")

    root = Path(raw_path)
    if not root.exists():
        raise ConnectorError("FILE_NOT_FOUND", f"路径不存在: {root}", str(root))

    if root.is_file():
        if root.suffix.lower() not in {".md", ".markdown"}:
            raise ConnectorError(
                "UNSUPPORTED_TYPE", f"不是 Markdown 文件: {root}", str(root)
            )
        files = [root]
    else:
        files = sorted(
            path
            for path in list(root.rglob("*.md")) + list(root.rglob("*.markdown"))
            if path.is_file()
        )
        if not files:
            raise ConnectorError(
                "EMPTY_SOURCE", f"目录中没有 Markdown 文件: {root}", str(root)
            )

    documents: list[Document] = []
    issues: list[ConnectorIssue] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            issues.append(ConnectorIssue("READ_ERROR", f"读取失败: {exc}", str(path)))
            continue
        if not text.strip():
            issues.append(ConnectorIssue("EMPTY_FILE", "文件为空", str(path)))
            continue

        source = path.name if root.is_file() else path.relative_to(root).as_posix()
        documents.append(
            Document.from_text(
                text,
                source,
                title=_title(text, path.stem),
                media_type="text/markdown",
                metadata={
                    "connector": "connector.markdown",
                    "absolute_path": str(path.resolve()),
                },
            )
        )

    return ConnectorResult(documents=documents, issues=issues)


__all__ = ["load"]
