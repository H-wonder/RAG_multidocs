import sys
import textwrap

from rag_multidocs.cli import build_runtime
from rag_multidocs.contracts import ConnectorConfig


def _fake_docparse_repo(tmp_path):
    repo = tmp_path / "fake_docparse"
    package = repo / "docparse"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "__main__.py").write_text(
        textwrap.dedent(
            """
            import json
            import sys
            from pathlib import Path

            if not Path(sys.argv[2]).is_absolute():
                raise SystemExit("input path is not absolute")
            out = Path(sys.argv[sys.argv.index("-o") + 1])
            out.mkdir(parents=True, exist_ok=True)
            ir = {
                "path": "sample.pdf",
                "sha256": "f" * 64,
                "doc_type": "pdf",
                "needs_review": False,
                "warnings": [],
                "pages": [
                    {
                        "number": 1,
                        "kind": "native",
                        "mode": "native",
                        "blocks": [
                            {
                                "id": "p1b0",
                                "page": 1,
                                "type": "title",
                                "bbox": [100, 80, 900, 160],
                                "reading_order": 0,
                                "content": "测试 PDF",
                                "source": "native_text",
                                "confidence": 0.99,
                                "warnings": [],
                            },
                            {
                                "id": "p1b1",
                                "page": 1,
                                "type": "text",
                                "bbox": [100, 200, 900, 420],
                                "reading_order": 1,
                                "content": "第一段内容，包含检索词。",
                                "source": "native_text",
                                "confidence": 0.99,
                                "warnings": [],
                            },
                        ],
                    }
                ],
            }
            (out / "result.json").write_text(
                json.dumps(ir, ensure_ascii=False), encoding="utf-8"
            )
            (out / "result.md").write_text(
                "# 测试 PDF\\n\\n第一段内容，包含检索词。\\n", encoding="utf-8"
            )
            """
        ),
        encoding="utf-8",
    )
    return repo


def test_pdf_plugin_runs_docparse_and_maps_ir(settings, tmp_path):
    repo = _fake_docparse_repo(tmp_path)
    settings.enabled_connectors = ["pdf"]
    settings.docparse_repo = str(repo)
    settings.docparse_python = sys.executable
    settings.docparse_mode = "auto"
    runtime = build_runtime(settings)

    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    result = runtime.registry.load(
        "pdf", ConnectorConfig(name="pdf", path=str(pdf))
    )

    assert result.documents
    document = result.documents[0]
    assert document.source == "sample.pdf"
    assert document.title == "测试 PDF"
    assert document.metadata["docparse_ir"]["pages"][0]["number"] == 1
    assert document.metadata["connector"] == "pdf"

    report = runtime.ingest.ingest("pdf", str(pdf))
    assert report.ok
    assert report.chunks >= 1
    hits = runtime.search.search("检索词")
    assert hits
    assert hits[0]["source"] == "sample.pdf"


def test_pdf_plugin_is_disabled_without_core_changes(settings, tmp_path):
    repo = _fake_docparse_repo(tmp_path)
    settings.enabled_connectors = ["txt"]
    settings.docparse_repo = str(repo)
    settings.docparse_python = sys.executable
    runtime = build_runtime(settings)

    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    report = runtime.ingest.ingest("pdf", str(pdf))

    assert not report.ok
    assert report.issues[0].code == "CONNECTOR_DISABLED"
