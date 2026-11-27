import json
import sys

from rag_multidocs.cli.app import build_registry, build_runtime, main
from rag_multidocs.cli.repl import Session
from rag_multidocs.config import Settings
from rag_multidocs.services import create_core


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_mixed_directory_and_paths_continue_after_failure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = Settings(index_backend="keyword", data_dir=str(tmp_path / "index"))
    with_core = create_core(settings, base_dir=tmp_path)
    text = write(tmp_path / "new corpus/子目录/更深层/欢迎 信.TXT", "Welcome to the greenhouse.")
    markdown = write(tmp_path / "new corpus/导入 指南.md", "# Guide\nUse rainwater for plants.")
    html = write(tmp_path / "new corpus/sub/page.html", "<h1>HTML</h1><p>A different document</p>")
    bad = write(tmp_path / "new corpus/sub/broken.PNG", "not an image")
    other = write(tmp_path / "new corpus/skip.xyz", "opaque")
    try:
        report = with_core.ingest_many(
            "auto", ["new corpus", text, markdown.relative_to(tmp_path), "missing.txt"]
        ).as_dict()
        assert (report["discovered"], report["succeeded"], report["failed"], report["skipped"]) == (5, 3, 1, 1)
        assert report["input_issues"][0]["code"] == "FILE_NOT_FOUND"
        rows = {row["path"]: row for row in report["files"]}
        for path, name in [(text, "txt"), (markdown, "connector.markdown"), (html, "connector.html")]:
            assert rows[str(path)]["connector"] == name
            assert rows[str(path)]["document_count"] == 1
            assert rows[str(path)]["chunks"] > 0
        assert rows[str(bad)]["connector"] == "connector.image"
        assert rows[str(bad)]["issues"][0]["code"] == "PARSE_ERROR"
        assert rows[str(other)]["issues"][0]["code"] == "UNSUPPORTED_TYPE"
    finally:
        with_core.close()


def test_unique_namespace_duplicate_and_update_share_single_ingest_logic(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runtime = build_runtime(Settings(data_dir=str(tmp_path / "index"), index_backend="keyword"))
    first = write(tmp_path / "group/a/同名 文件.txt", "first content")
    second = write(tmp_path / "group/b/同名 文件.txt", "second content")
    initial = runtime.ingest.ingest("auto", "group").as_dict()
    ids = [row["documents"][0]["doc_id"] for row in initial["files"]]
    assert len(set(ids)) == 2
    duplicate = runtime.ingest.ingest("auto", first).files[0]
    assert duplicate["chunks"] == 0 and duplicate["documents"][0]["status"] == "unchanged"
    assert duplicate["documents"][0]["doc_id"] == ids[0]
    write(first, "new revision")
    updated = runtime.ingest.ingest("auto", [first, second]).as_dict()
    docs = [row["documents"][0] for row in updated["files"]]
    assert [d["doc_id"] for d in docs] == ids
    assert [d["version"] for d in docs] == [2, 1]
    assert [d["status"] for d in docs] == ["indexed", "unchanged"]
    # Explicit import still keeps the original report and logical source name.
    legacy = runtime.ingest.ingest("txt", str(second))
    assert legacy.documents[0]["source"] == second.name


def test_third_party_manifest_auto_matches_and_disabled_reports_reason(tmp_path):
    plugin = tmp_path / "plugins/third"
    write(plugin / "manifest.json", json.dumps({
        "name": "third.records", "entry": "adapter:load", "file_extensions": [".ABC"],
    }))
    write(plugin / "adapter.py", """
from pathlib import Path
from rag_multidocs.model import Document
def load(payload, context=None):
    path = Path(payload["path"])
    return [Document.from_text(path.read_text(), path.name)]
""")
    file = write(tmp_path / "新 格式.Abc", "A third-party format.")
    settings = Settings(data_dir=str(tmp_path / "index"), index_backend="keyword",
                        connectors_dir=str(plugin.parent), enabled_connectors=["third.records"])
    runtime = build_runtime(settings)
    assert runtime.ingest.ingest("auto", file).files[0]["connector"] == "third.records"
    runtime.registry.enabled = set()
    skipped = runtime.ingest.ingest("auto", file).files[0]
    assert skipped["status"] == "skipped"
    assert skipped["issues"][0]["code"] == "CONNECTOR_DISABLED"


def test_specialized_connectors_take_priority_over_pdf(project_root):
    registry = build_registry(Settings(connectors_dir=str(project_root / "connectors")))
    routes = registry.file_routes()
    assert {ext: routes[ext][0] for ext in (".docx", ".xlsx", ".png", ".pdf")} == {
        ".docx": "connector.word", ".xlsx": "connector.excel", ".png": "connector.image", ".pdf": "pdf",
    }
    registry.enabled.discard("connector.image")
    assert registry.file_routes()[".png"] == ("pdf", "")


def test_cli_repl_use_shared_ingest_many(tmp_path, monkeypatch, capsys):
    from rag_multidocs.ingestion.service import IngestService

    config = tmp_path / "config.json"
    Settings(data_dir=str(tmp_path / "index"), index_backend="keyword",
             credentials_file=str(tmp_path / "unused.json")).save(config)
    a = write(tmp_path / "中文 路径.txt", "one distinct paragraph")
    b = write(tmp_path / "other file.md", "# Hello\nAnother distinct paragraph")
    seen = []
    implementation = IngestService.ingest_many

    def observe(self, connector, paths, options=None):
        seen.append((connector, list(paths)))
        return implementation(self, connector, paths, options)

    monkeypatch.setattr(IngestService, "ingest_many", observe)
    monkeypatch.setenv("RAG_CONFIG", str(config))
    assert main(["ingest", "--path", str(a), str(b), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["succeeded"] == 2
    session = Session(str(config))
    session.dispatch(f'/ingest auto "{a}" "{b}"')
    output = capsys.readouterr().out
    assert "发现 2 个文件，成功 2" in output and "unchanged" in output
    assert seen == [("auto", [str(a), str(b)])] * 2


def test_connector_paths_config_relative_and_existing_conda_environment(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    plugins = write(root / "project/connectors/demo/manifest.json", "{}").parent.parent
    repo = write(root / "已有资源/可选组件/docparse/docparse/__main__.py", "").parents[1]
    prefixes = tmp_path / "conda/envs"
    python = write(prefixes / "docparse/bin/python", "")
    python.chmod(0o755)
    config = write(root / "config.json", json.dumps({
        "chat_provider": "dashscope", "chat_model": "deepseek-v4.1-flash",
        "milvus_collection": "existing", "embed_dim": 512,
    }))
    monkeypatch.setattr(sys, "prefix", str(prefixes / "rag"))
    monkeypatch.chdir(tmp_path)
    settings = Settings.load(config)
    resolved = settings.resolved_connector_settings()
    assert resolved["connectors_dir"] == str(plugins)
    assert resolved["docparse_repo"] == str(repo)
    assert resolved["docparse_python"] == str(python)
    assert (settings.chat_provider, settings.chat_model, settings.milvus_collection, settings.embed_dim) == (
        "dashscope", "deepseek-v4.1-flash", "existing", 512,
    )
    monkeypatch.setenv("RAG_CONNECTORS_DIR", "third party")
    monkeypatch.setenv("DOCPARSE_REPO", "custom parser")
    monkeypatch.setenv("DOCPARSE_PYTHON", "own environment/bin/python")
    override = Settings.load(config).resolved_connector_settings()
    assert override["connectors_dir"] == str(root / "third party")
    assert override["docparse_repo"] == str(root / "custom parser")
    assert override["docparse_python"] == str(root / "own environment/bin/python")


def test_display_documents_accepts_milvus_active_catalog_record(capsys):
    from rag_multidocs.cli.app import print_documents

    print_documents([{"source": "项目/文件.pdf", "chunks": 4, "doc_id": "abc"}])
    output = capsys.readouterr().out
    assert "项目/文件.pdf" in output and "4      -  abc" in output
