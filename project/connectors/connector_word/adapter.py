"""Independent word connector; only public contracts/models on the host."""
import json
import os
import subprocess
import sys
from pathlib import Path
from rag_multidocs.contracts import ConnectorResult, ConnectorIssue
from rag_multidocs.model import Document, sha256_file


def load(payload, context=None):
    options = dict(payload.get("options") or {})
    settings = (context or {}).get("settings", {})
    python = options.get("parser_python") or settings.get("docparse_python") or sys.executable
    repo = options.get("docparse_repo") or settings.get("docparse_repo") or ""
    path = Path(payload["path"]).expanduser().resolve()
    result = ConnectorResult()
    if not path.exists():
        return ConnectorResult(issues=[ConnectorIssue("FILE_NOT_FOUND", "Input does not exist", str(path))])
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() == ".docx")
    if not files:
        result.issues.append(ConnectorIssue("EMPTY_SOURCE", "No supported documents", str(path)))
    for file in files:
        source = file.name if path.is_file() else file.relative_to(path).as_posix()
        if file.suffix.lower() != ".docx":
            result.issues.append(ConnectorIssue("UNSUPPORTED_TYPE", "Expected .docx", source))
            continue
        request = {"path": str(file), "options": options, "docparse_repo": repo}
        try:
            process = subprocess.run([str(python), str(Path(__file__).with_name("worker.py"))], input=json.dumps(request), capture_output=True, text=True, timeout=float(options.get("timeout_s", 180)), env={**os.environ, "PYTHONPATH": str(repo), "TOKENIZERS_PARALLELISM": "false"})
            if process.returncode:
                raise RuntimeError("parser process exited")
            parsed = json.loads(process.stdout)
            if parsed.get("error"):
                result.issues.append(ConnectorIssue(parsed["error"], parsed.get("message", "Parse failed"), source))
                continue
            doc = Document.from_text(parsed["content"], source, title=parsed.get("title", file.stem), media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document", metadata=parsed.get("metadata", {}))
            doc.metadata.update({"sha256": sha256_file(file), "file_sha256": sha256_file(file), "connector": "connector.word", "parser": "python-docx"})
            doc.segments = parsed["blocks"]
            result.documents.append(doc)
        except FileNotFoundError:
            result.issues.append(ConnectorIssue("CONNECTOR_UNAVAILABLE", "解析解释器不可用，请检查 DOCPARSE_PYTHON", source))
        except subprocess.TimeoutExpired:
            result.issues.append(ConnectorIssue("PARSE_TIMEOUT", "Parser exceeded deadline", source))
        except (OSError, RuntimeError, ValueError, KeyError, TypeError):
            result.issues.append(ConnectorIssue("PARSE_ERROR", "Invalid or unsupported document", source))
    return result
