"""Image OCR Connector: existing docparse public CLI, no new OCR client."""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image
from rag_multidocs.contracts import ConnectorResult, ConnectorIssue
from rag_multidocs.model import Document, sha256_file


def load(payload, context=None):
    options = dict(payload.get("options") or {})
    settings = (context or {}).get("settings", {})
    repo = str(options.get("docparse_repo") or settings.get("docparse_repo") or "")
    python = str(options.get("docparse_python") or settings.get("docparse_python") or sys.executable)
    root = Path(payload["path"]).expanduser().resolve()
    result = ConnectorResult()
    if not root.exists():
        return ConnectorResult(issues=[ConnectorIssue("FILE_NOT_FOUND", "Input image does not exist", str(root))])
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not files:
        result.issues.append(ConnectorIssue("EMPTY_SOURCE", "No PNG/JPEG images", str(root)))
    for path in files:
        source = path.name if root.is_file() else path.relative_to(root).as_posix()
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            result.issues.append(ConnectorIssue("UNSUPPORTED_TYPE", "Expected PNG/JPEG", source))
            continue
        try:
            with Image.open(path) as image:
                image.verify()
            if not repo or not (Path(repo) / "docparse/__main__.py").is_file():
                result.issues.append(ConnectorIssue("CONNECTOR_UNAVAILABLE", "请用 DOCPARSE_REPO 指向含 docparse/ 包的目录", source))
                continue
            with tempfile.TemporaryDirectory(prefix="image_connector_") as temp:
                process = subprocess.run([python, "-m", "docparse", "parse", str(path), "-o", temp, "--format", "json", "--mode", "page"], cwd=repo, env={**os.environ, "PYTHONPATH": repo, "TOKENIZERS_PARALLELISM": "false"}, capture_output=True, text=True, timeout=float(options.get("timeout_s", 120)))
                if process.returncode:
                    result.issues.append(ConnectorIssue("PARSE_ERROR", "OCR 子进程失败，请检查 DOCPARSE_PYTHON 环境的 Pillow、PyMuPDF 和本地 OCR 服务", source))
                    continue
                ir = json.loads(next(Path(temp).glob("*.json")).read_text())
            segments = []
            for page in ir.get("pages", []):
                for b in page.get("blocks", []):
                    # page mode has no region localization: do not inherit its
                    # synthetic whole-page rectangle as if an OCR word box.
                    segments.append({**b, "page": 1, "bbox": [], "reading_order": len(segments), "origin": "ocr_result", "ocr_mode": "page", "warnings": list(b.get("warnings", [])) + [{"code": "IMAGE_NO_BBOX", "severity": "info", "message": "Whole-image OCR; no localized box"}]})
            text = "\n\n".join(str(b.get("content") or b.get("html") or "") for b in segments)
            if not text.strip():
                result.issues.append(ConnectorIssue("OCR_EMPTY", "OCR returned no text", source))
                continue
            warnings = list(ir.get("warnings", [])) + [w for p in ir.get("pages", []) for w in p.get("warnings", [])]
            doc = Document.from_text(text, source, title=path.stem, media_type="image/png" if path.suffix.lower() == ".png" else "image/jpeg", metadata={"sha256": sha256_file(path), "file_sha256": sha256_file(path), "origin": "ocr_result", "ocr_mode": "page", "warnings": warnings, "needs_review": bool(ir.get("needs_review")), "connector": "connector.image"})
            doc.metadata["sha256"] = sha256_file(path)
            doc.metadata["docparse_ir"] = ir
            doc.segments = segments
            result.documents.append(doc)
        except subprocess.TimeoutExpired:
            result.issues.append(ConnectorIssue("PARSE_TIMEOUT", "OCR exceeded deadline", source))
        except FileNotFoundError:
            result.issues.append(ConnectorIssue("CONNECTOR_UNAVAILABLE", "解析解释器不可用，请检查 DOCPARSE_PYTHON", source))
        except (OSError, ValueError, StopIteration, KeyError, TypeError):
            result.issues.append(ConnectorIssue("PARSE_ERROR", "Corrupt image or OCR failure", source))
    return result
