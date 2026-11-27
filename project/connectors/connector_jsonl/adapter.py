"""Architecture example: standardized JSONL records, independent of ingestion/index/QA."""

import json
from pathlib import Path
from rag_multidocs.contracts import ConnectorIssue, ConnectorResult
from rag_multidocs.model import Document


def load(payload, context=None):
    path = Path(payload["path"])
    result = ConnectorResult()
    try:
        for number, line in enumerate(path.read_text(encoding="utf8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                source = str(record.get("source", f"{path.name}#{number}"))
                doc = Document.from_text(
                    str(record.get("text", "")), source, title=str(record.get("title", "")),
                    metadata={k: record[k] for k in ("provenance", "source_sha256", "annotation", "origin") if k in record}
                )
                doc.segments = list(record.get("blocks", []))
                if not doc.content and not doc.segments:
                    raise ValueError("empty record")
                result.documents.append(doc)
            except (ValueError, TypeError, AttributeError):
                result.issues.append(
                    ConnectorIssue("RECORD_ERROR", f"invalid record at line {number}", path.name)
                )
    except OSError:
        result.issues.append(ConnectorIssue("READ_ERROR", "cannot read JSONL", path.name))
    return result
