"""File enumeration and reporting for the shared batch ingestion service."""

from dataclasses import dataclass, field
import os
from pathlib import Path

from ..contracts import ConnectorIssue


@dataclass
class BatchIngestReport:
    files: list[dict] = field(default_factory=list)
    input_issues: list[ConnectorIssue] = field(default_factory=list)

    @property
    def ok(self):
        return any(entry["document_count"] for entry in self.files)

    def as_dict(self):
        return {
            "discovered": len(self.files),
            "succeeded": sum(row["status"] == "succeeded" for row in self.files),
            "failed": sum(row["status"] in {"failed", "partial"} for row in self.files),
            "skipped": sum(row["status"] == "skipped" for row in self.files),
            "partial": sum(row["status"] == "partial" for row in self.files),
            "document_count": sum(row["document_count"] for row in self.files),
            "chunks": sum(row["chunks"] for row in self.files),
            "input_issues": [issue.as_dict() for issue in self.input_issues],
            "files": self.files,
        }


def collect_files(paths, report):
    """Preserve input order and sort recursion. Resolve aliases before deduplication."""
    seen = set()

    def report_error(exc):
        report.input_issues.append(ConnectorIssue(
            "PATH_READ_ERROR", f"无法读取路径: {exc.strerror or type(exc).__name__}", str(exc.filename or ""),
        ))

    for raw in paths:
        path = Path(raw).expanduser()
        try:
            path = path.resolve(strict=True)
            if path.is_file():
                candidates = [path]
            elif path.is_dir():
                candidates = []
                issue_count = len(report.input_issues)
                for root, directories, files in os.walk(path, onerror=report_error):
                    directories.sort()
                    candidates.extend(Path(root) / name for name in sorted(files))
                if not candidates and len(report.input_issues) == issue_count:
                    report.input_issues.append(ConnectorIssue("EMPTY_SOURCE", "目录中没有文件", str(path)))
            else:
                report.input_issues.append(ConnectorIssue("INVALID_PATH", "需要普通文件或目录", str(path)))
                continue
        except FileNotFoundError:
            report.input_issues.append(ConnectorIssue("FILE_NOT_FOUND", "路径不存在", str(path)))
            continue
        except OSError as exc:
            report_error(exc)
            continue
        except RuntimeError:
            report.input_issues.append(ConnectorIssue("INVALID_PATH", "无法解析路径链接", str(path)))
            continue

        for candidate in candidates:
            # Explicit symlinks are already resolved. Do not follow links found
            # inside recursively scanned directories.
            if candidate.is_symlink():
                if candidate not in seen:
                    seen.add(candidate)
                    yield candidate, ConnectorIssue("SYMLINK_SKIPPED", "递归导入跳过符号链接", str(candidate))
            elif candidate not in seen:
                seen.add(candidate)
                yield candidate, None
