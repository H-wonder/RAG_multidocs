"""独立 PDF/docparse 插件：PDF / 扫描件 / Office → 统一 Document。

核心通过 ``manifest.json`` 发现本插件，按 ``adapter:load`` 调用 ``load``。
插件只依赖公开的 ``rag_multidocs.contracts`` / ``rag_multidocs.model``，
不 import 核心的 ingest、cli 或旧 ``connectors.pdf`` 实现。

多模态识别（OCR、版面分析、表格/公式）留在外部 docparse；
本插件只负责子进程调用和 IR → Document 映射。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from rag_multidocs.contracts import (
        ConnectorConfig,
        ConnectorError,
        ConnectorIssue,
        ConnectorResult,
    )
    from rag_multidocs.model import Document, sha256_file, stable_doc_id
except ImportError:  # 源码目录直接运行时，允许从 project/src 导入
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from rag_multidocs.contracts import (
        ConnectorConfig,
        ConnectorError,
        ConnectorIssue,
        ConnectorResult,
    )
    from rag_multidocs.model import Document, sha256_file, stable_doc_id


class DocparseUnavailable(RuntimeError):
    """docparse 仓库或运行环境不可用（可重试/需要部署）。"""


class DocparseError(RuntimeError):
    """docparse 执行失败（文件损坏、解析失败等，不应无脑重试）。"""


def load_ir(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _first_title(ir: dict[str, Any]) -> str:
    for page in ir.get("pages", []) or []:
        for block in sorted(page.get("blocks", []) or [],
                            key=lambda item: item.get("reading_order") or 0):
            if block.get("type") == "title" and (block.get("content") or "").strip():
                return str(block["content"]).strip()
    return ""


def document_from_ir(
    ir: dict[str, Any],
    source: str,
    *,
    content: str | None = None,
) -> Document:
    """把 docparse IR 映射成统一 Document。

    ``content`` 优先用 docparse 生成的 Markdown；没有 Markdown 时按阅读顺序拼接
    block 的 content/html。结构化切块请把原 IR 交给 ``chunking.chunk_docparse_ir``，
    不要先展平成纯文本再切。
    """
    if content is None:
        parts: list[str] = []
        for page in sorted(ir.get("pages", []) or [],
                           key=lambda item: item.get("number", 0)):
            for block in sorted(page.get("blocks", []) or [],
                                key=lambda item: item.get("reading_order") or 0):
                text = (block.get("content") or block.get("html") or "").strip()
                if text:
                    parts.append(text)
        content = "\n\n".join(parts)

    pages = len(ir.get("pages", []) or [])
    media_type = "application/pdf" if source.lower().endswith(".pdf") else (
        "application/octet-stream"
    )
    metadata = {
        "sha256": ir.get("sha256") or "",
        "doc_type": ir.get("doc_type", ""),
        "pages": pages,
        "needs_review": bool(ir.get("needs_review")),
        "warnings": ir.get("warnings", []),
    }
    document = Document.from_text(
        content,
        source,
        title=_first_title(ir) or Path(source).stem,
        media_type=media_type,
        metadata=metadata,
    )
    # Document.from_text 会用正文重新算 sha256；docparse 的 sha 是原文件哈希，更准确。
    if metadata["sha256"]:
        document.metadata["sha256"] = metadata["sha256"]
    return document


@dataclass
class DocparseOptions:
    repo: str = os.environ.get("DOCPARSE_REPO", "")
    python: str = os.environ.get("DOCPARSE_PYTHON", sys.executable)
    mode: str = os.environ.get("DOCPARSE_MODE", "auto")  # auto | page | region
    timeout_s: int = int(os.environ.get("DOCPARSE_TIMEOUT_S", "1800"))
    extra_args: tuple[str, ...] = ()


class DocparseConnector:
    def __init__(self, options: DocparseOptions | None = None):
        self.options = options or DocparseOptions()

    def available(self) -> bool:
        repo = Path(self.options.repo) if self.options.repo else None
        return bool(repo and (repo / "docparse" / "__main__.py").exists())

    def parse(
        self,
        path: str | Path,
        outdir: str | Path,
        *,
        mode: str | None = None,
    ) -> tuple[Document, dict[str, Any]]:
        """同步解析一个文件，返回 (Document, 原始 IR)。

        失败语义：
        - 找不到 docparse → ``DocparseUnavailable``；
        - 文件不存在 → ``FileNotFoundError``；
        - 解析器退出码非 0 → ``DocparseError``（带 stdout/stderr 尾部）。
        """
        source_path = Path(path)
        if not source_path.is_file():
            raise FileNotFoundError(str(source_path))
        if not self.available():
            raise DocparseUnavailable(
                f"docparse 不可用：repo={self.options.repo!r}；"
                "请指向含 docparse/ 包的目录并安装其依赖"
            )

        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        command = [
            self.options.python,
            "-m",
            "docparse",
            "parse",
            str(source_path),
            "-o",
            str(out),
            "--format",
            "md,json",
            "--mode",
            mode or self.options.mode,
            *self.options.extra_args,
        ]
        env = {
            **os.environ,
            "PYTHONPATH": self.options.repo,
            "TOKENIZERS_PARALLELISM": "false",
        }
        started = time.time()
        try:
            process = subprocess.run(
                command,
                cwd=self.options.repo,
                env=env,
                capture_output=True,
                text=True,
                timeout=self.options.timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise DocparseError(
                f"docparse 超过 {self.options.timeout_s}s 未完成"
            ) from exc

        tail = ((process.stdout or "")[-800:] + (process.stderr or "")[-800:]).strip()
        if process.returncode != 0:
            raise DocparseError(
                f"docparse 退出码 {process.returncode}: {tail[-300:]}"
            )

        json_files = sorted(out.glob("*.json"))
        md_files = sorted(out.glob("*.md"))
        if not json_files:
            raise DocparseError("docparse 退出码 0 但没有产出 IR JSON")

        ir = load_ir(json_files[0])
        markdown = (
            md_files[0].read_text(encoding="utf-8", errors="replace")
            if md_files
            else None
        )
        document = document_from_ir(
            ir, source_path.name, content=markdown
        )
        document.metadata.update(
            {
                "ir_path": str(json_files[0]),
                "md_path": str(md_files[0]) if md_files else "",
                "duration_ms": int((time.time() - started) * 1000),
                "file_sha256": sha256_file(source_path),
                "parse_mode": mode or self.options.mode,
            }
        )
        # 原文件哈希用于增量更新；IR 里的 sha256 有时为空
        if not document.metadata.get("sha256"):
            document.metadata["sha256"] = document.metadata["file_sha256"]
        return document, ir


class PdfConnector:
    """Built-in PDF/scan/Office connector backed by docparse (subprocess)."""

    name = "pdf"
    SUPPORTED = {
        ".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff",
        ".bmp", ".webp", ".docx", ".pptx", ".xlsx",
    }

    def __init__(self, options: DocparseOptions | None = None):
        self.engine = DocparseConnector(options)

    def load(self, config: ConnectorConfig) -> ConnectorResult:
        root = Path(config.path)
        if not root.exists():
            return ConnectorResult(
                issues=[ConnectorIssue("FILE_NOT_FOUND", f"路径不存在: {root}", config.path)]
            )

        if root.is_file():
            if root.suffix.lower() not in self.SUPPORTED:
                return ConnectorResult(
                    issues=[ConnectorIssue(
                        "UNSUPPORTED_TYPE",
                        f"PDF Connector 不支持 {root.suffix or '无后缀'} 文件",
                        config.path,
                    )]
                )
            files = [root]
        else:
            files = sorted(
                path for path in root.rglob("*")
                if path.is_file() and path.suffix.lower() in self.SUPPORTED
            )
            if not files:
                return ConnectorResult(
                    issues=[ConnectorIssue(
                        "EMPTY_SOURCE", f"目录中没有可解析文件: {root}", config.path
                    )]
                )

        documents: list[Document] = []
        issues: list[ConnectorIssue] = []
        for path in files:
            source = path.name if root.is_file() else path.relative_to(root).as_posix()
            try:
                with tempfile.TemporaryDirectory(prefix="rag_multidocs_parse_") as outdir:
                    document, ir = self.engine.parse(path, outdir, mode=config.options.get("mode"))
            except DocparseUnavailable as exc:
                issues.append(ConnectorIssue("CONNECTOR_UNAVAILABLE", str(exc), str(path)))
                continue
            except DocparseError as exc:
                issues.append(ConnectorIssue("PARSE_ERROR", str(exc), str(path)))
                continue

            document.source = source
            document.doc_id = stable_doc_id(source)
            document.metadata["connector"] = self.name
            document.metadata["docparse_ir"] = ir
            documents.append(document)

        return ConnectorResult(documents=documents, issues=issues)


def _settings(context: dict[str, Any] | None) -> dict[str, Any]:
    """核心传入的公共配置 dict；插件不依赖 Settings 类型。"""
    value = (context or {}).get("settings", {})
    return value if isinstance(value, dict) else {}


def load(payload: dict[str, Any], context: dict[str, Any] | None = None) -> ConnectorResult:
    """Plugin entrypoint: load(payload, context) -> ConnectorResult."""
    options = dict(payload.get("options") or {})
    raw_path = payload.get("path") or options.get("path")
    if not raw_path:
        raise ConnectorError("INVALID_CONFIG", "缺少 path")

    settings = _settings(context)
    repo = (
        options.get("docparse_repo")
        or settings.get("docparse_repo")
        or os.environ.get("DOCPARSE_REPO", "")
    )
    python = (
        options.get("docparse_python")
        or settings.get("docparse_python")
        or os.environ.get("DOCPARSE_PYTHON")
        or sys.executable
    )
    mode = (
        options.get("mode")
        or settings.get("docparse_mode")
        or os.environ.get("DOCPARSE_MODE", "auto")
    )
    timeout_raw = (
        options.get("docparse_timeout_s")
        or settings.get("docparse_timeout_s")
        or os.environ.get("DOCPARSE_TIMEOUT_S", 1800)
    )
    try:
        timeout_s = int(timeout_raw)
    except (TypeError, ValueError):
        timeout_s = 1800

    connector = PdfConnector(
        DocparseOptions(
            repo=str(repo or ""),
            python=str(python),
            mode=str(mode),
            timeout_s=timeout_s,
        )
    )
    config = ConnectorConfig(
        name=str(payload.get("name") or "pdf"),
        path=str(raw_path),
        options=options,
    )
    return connector.load(config)


__all__ = [
    "DocparseConnector",
    "DocparseOptions",
    "DocparseError",
    "DocparseUnavailable",
    "PdfConnector",
    "document_from_ir",
    "load",
    "load_ir",
]
