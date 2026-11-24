"""Independent HTML connector: extract visible text from .html/.htm files.

This adapter is loaded from ``manifest.json`` by the core registry. It only
depends on the public ``rag_multidocs.contracts`` / ``rag_multidocs.model``
modules and never imports core ingestion/CLI internals.
"""

from __future__ import annotations

import sys
from html.parser import HTMLParser
from pathlib import Path

try:
    from rag_multidocs.contracts import ConnectorIssue, ConnectorResult
    from rag_multidocs.model import Document
except ImportError:  # allow running from a source checkout without installing
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from rag_multidocs.contracts import ConnectorIssue, ConnectorResult
    from rag_multidocs.model import Document

_BLOCK_TAGS = {
    "address", "article", "aside", "blockquote", "br", "dd", "div", "dl",
    "dt", "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6", "header",
    "hr", "li", "main", "nav", "ol", "p", "pre", "section", "table", "td",
    "th", "tr", "ul",
}
_SKIP_TAGS = {"script", "style", "noscript", "template"}


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.parts: list[str] = []
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        elif tag == "title":
            self._in_title = False
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = " ".join(data.split())
        if not text:
            return
        if self._in_title:
            self.title_parts.append(text)
        else:
            self.parts.append(text)

    def title(self) -> str:
        return " ".join(self.title_parts).strip()

    def text(self) -> str:
        raw = " ".join(self.parts)
        lines = [" ".join(line.split()) for line in raw.splitlines()]
        return "\n".join(line for line in lines if line)


def _extract(path: Path) -> tuple[str, str]:
    parser = _VisibleTextParser()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))
    parser.close()
    return parser.title(), parser.text()


def _files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".html", ".htm"}
    )


def load(payload: dict, context: dict | None = None) -> ConnectorResult:
    """Read an HTML file or directory and return a unified Document list."""
    raw_path = payload.get("path") or (payload.get("options") or {}).get("path")
    if not raw_path:
        return ConnectorResult(
            issues=[ConnectorIssue("INVALID_CONFIG", "缺少 path")]
        )

    root = Path(str(raw_path))
    if not root.exists():
        return ConnectorResult(
            issues=[ConnectorIssue("FILE_NOT_FOUND", f"路径不存在: {root}", str(root))]
        )
    if root.is_file() and root.suffix.lower() not in {".html", ".htm"}:
        return ConnectorResult(
            issues=[
                ConnectorIssue(
                    "UNSUPPORTED_TYPE",
                    f"HTML Connector 不支持 {root.suffix or '无后缀'} 文件",
                    str(root),
                )
            ]
        )

    files = _files(root)
    if not files:
        return ConnectorResult(
            issues=[ConnectorIssue("EMPTY_SOURCE", f"目录中没有 HTML 文件: {root}", str(root))]
        )

    documents: list[Document] = []
    issues: list[ConnectorIssue] = []
    for path in files:
        try:
            title, text = _extract(path)
        except OSError as exc:
            issues.append(ConnectorIssue("READ_ERROR", f"读取失败: {exc}", str(path)))
            continue
        if not text.strip():
            issues.append(ConnectorIssue("EMPTY_FILE", "HTML 没有可见正文", str(path)))
            continue
        source = path.name if root.is_file() else path.relative_to(root).as_posix()
        documents.append(
            Document.from_text(
                text,
                source,
                title=title or path.stem,
                media_type="text/html",
                metadata={
                    "connector": "connector.html",
                    "absolute_path": str(path.resolve()),
                },
            )
        )

    if not documents and not issues:
        issues.append(ConnectorIssue("EMPTY_SOURCE", "没有可导入的 HTML", str(root)))
    return ConnectorResult(documents=documents, issues=issues)


__all__ = ["load"]
