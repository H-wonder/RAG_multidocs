from rag_multidocs.cli import build_runtime
from rag_multidocs.connectors.pdf import PdfConnector, DocparseOptions
from rag_multidocs.contracts import ConnectorConfig


def test_missing_path_returns_structured_issue(runtime):
    report = runtime.ingest.ingest("txt", "/definitely/missing/path.txt")
    assert not report.ok
    assert report.issues[0].code == "FILE_NOT_FOUND"


def test_empty_file_and_empty_directory(runtime, tmp_path):
    ingest = runtime.ingest
    empty = tmp_path / "empty.txt"
    empty.write_text("   \n", encoding="utf-8")
    report = ingest.ingest("txt", str(empty))
    assert not report.ok
    assert report.issues[0].code == "EMPTY_FILE"

    directory = tmp_path / "empty-dir"
    directory.mkdir()
    report = ingest.ingest("txt", str(directory))
    assert not report.ok
    assert report.issues[0].code == "EMPTY_SOURCE"


def test_unknown_connector_is_reported(runtime):
    report = runtime.ingest.ingest("does-not-exist", "x")
    assert not report.ok
    assert report.issues[0].code == "CONNECTOR_NOT_FOUND"


def test_disabled_markdown_connector_does_not_break_txt(settings, tmp_path, project_root):
    txt = tmp_path / "a.txt"
    txt.write_text("检索器负责取回段落。", encoding="utf-8")

    settings.enabled_connectors = ["txt", "pdf"]
    ingest = build_runtime(settings).ingest

    md_dir = tmp_path / "md"
    md_dir.mkdir()
    (md_dir / "a.md").write_text("# A\n\ntext\n", encoding="utf-8")
    report = ingest.ingest("connector.markdown", str(md_dir))
    assert not report.ok
    assert report.issues[0].code == "CONNECTOR_DISABLED"

    report = ingest.ingest("txt", str(txt))
    assert report.ok


def test_pdf_connector_without_docparse_returns_unavailable(tmp_path):
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    connector = PdfConnector(DocparseOptions(repo=""))
    result = connector.load(ConnectorConfig(name="pdf", path=str(pdf)))
    assert not result.documents
    assert result.issues[0].code == "CONNECTOR_UNAVAILABLE"
