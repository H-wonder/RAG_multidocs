"""Built-in TXT connector: single file or a directory of .txt files."""

from __future__ import annotations

from pathlib import Path

from ..contracts import (
    ConnectorConfig,
    ConnectorIssue,
    ConnectorResult,
)
from ..model import Document


class TxtConnector:
    name = "txt"

    def load(self, config: ConnectorConfig) -> ConnectorResult:
        root = Path(config.path)
        if not root.exists():
            return ConnectorResult(
                issues=[
                    ConnectorIssue(
                        "FILE_NOT_FOUND", f"路径不存在: {root}", config.path
                    )
                ]
            )

        if root.is_file():
            if root.suffix.lower() != ".txt":
                return ConnectorResult(
                    issues=[
                        ConnectorIssue(
                            "UNSUPPORTED_TYPE",
                            f"TXT Connector 不支持 {root.suffix or '无后缀'} 文件",
                            config.path,
                        )
                    ]
                )
            files = [root]
        else:
            files = sorted(path for path in root.rglob("*.txt") if path.is_file())
            if not files:
                return ConnectorResult(
                    issues=[
                        ConnectorIssue(
                            "EMPTY_SOURCE",
                            f"目录中没有 .txt 文件: {root}",
                            config.path,
                        )
                    ]
                )

        documents: list[Document] = []
        issues: list[ConnectorIssue] = []
        for path in files:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                issues.append(
                    ConnectorIssue("READ_ERROR", f"读取失败: {exc}", str(path))
                )
                continue
            if not text.strip():
                issues.append(
                    ConnectorIssue("EMPTY_FILE", "文件为空", str(path))
                )
                continue

            if root.is_file():
                source = path.name
            else:
                source = path.relative_to(root).as_posix()
            documents.append(
                Document.from_text(
                    text,
                    source,
                    title=path.stem,
                    metadata={
                        "connector": self.name,
                        "absolute_path": str(path.resolve()),
                    },
                )
            )

        if not documents and not issues:
            issues.append(
                ConnectorIssue("EMPTY_SOURCE", "没有可导入的文本", config.path)
            )
        return ConnectorResult(documents=documents, issues=issues)


__all__ = ["TxtConnector"]
