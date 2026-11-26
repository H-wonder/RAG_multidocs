from rag_multidocs.cli import main
from rag_multidocs.config import Settings


def _config(tmp_path, project_root):
    config = Settings(
        data_dir=str(tmp_path / "data"),
        connectors_dir=str(project_root / "connectors"),
        index_backend="keyword",
    )
    path = tmp_path / "config.json"
    config.save(path)
    return path


def test_cli_ingest_search_and_docs(tmp_path, project_root, capsys):
    config = _config(tmp_path, project_root)
    txt = tmp_path / "notes.txt"
    txt.write_text("检索增强生成需要检索器和生成器。", encoding="utf-8")

    assert main(["ingest", "--source", "txt", "--path", str(txt), "--config", str(config)]) == 0
    out = capsys.readouterr().out
    assert "documents: 1" in out
    assert "chunks: 1" in out

    assert main(["search", "检索器", "--config", str(config)]) == 0
    out = capsys.readouterr().out
    assert "notes.txt" in out
    assert "§" not in out or "p0" in out

    assert main(["docs", "list", "--config", str(config)]) == 0
    assert "notes.txt" in capsys.readouterr().out

    assert main(["connectors", "list", "--config", str(config)]) == 0
    out = capsys.readouterr().out
    assert "txt" in out and "connector.markdown" in out


def test_cli_missing_path_returns_nonzero(tmp_path, project_root, capsys):
    config = _config(tmp_path, project_root)
    code = main(
        [
            "ingest",
            "--source",
            "txt",
            "--path",
            str(tmp_path / "missing.txt"),
            "--config",
            str(config),
        ]
    )
    assert code == 2
    assert "FILE_NOT_FOUND" in capsys.readouterr().err


def test_cli_can_disable_and_enable_connector(tmp_path, project_root, capsys):
    config = _config(tmp_path, project_root)

    assert main(["connectors", "disable", "connector.markdown", "--config", str(config)]) == 0
    settings = Settings.load(config)
    assert "connector.markdown" not in settings.enabled_connectors

    assert main(["connectors", "enable", "connector.markdown", "--config", str(config)]) == 0
    settings = Settings.load(config)
    assert "connector.markdown" in settings.enabled_connectors


def test_cli_ask_without_generation_returns_hits(tmp_path, project_root, capsys):
    config = _config(tmp_path, project_root)
    txt = tmp_path / "notes.txt"
    txt.write_text("检索增强生成需要检索器和生成器。", encoding="utf-8")
    assert main(["ingest", "--source", "txt", "--path", str(txt), "--config", str(config)]) == 0
    capsys.readouterr()

    code = main(["ask", "检索器", "--no-generate", "--config", str(config)])
    out = capsys.readouterr().out
    assert code == 0
    assert "notes.txt" in out


def test_cli_ask_with_fake_chat_model(tmp_path, project_root, capsys, monkeypatch):
    config = _config(tmp_path, project_root)
    txt = tmp_path / "notes.txt"
    txt.write_text("检索增强生成需要检索器和生成器。", encoding="utf-8")
    assert main(["ingest", "--source", "txt", "--path", str(txt), "--config", str(config)]) == 0
    capsys.readouterr()

    class FakeChat:
        model = "fake-chat"

        def complete(self, messages):
            return "检索器负责取回段落 [1]。"

    monkeypatch.setattr(
        "rag_multidocs.cli.commands.ask.build_chat_model",
        lambda *args, **kwargs: FakeChat(),
    )
    code = main(
        ["ask", "检索器", "--provider", "deepseek", "--config", str(config)]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "答案" in out and "[1]" in out
    assert "notes.txt" in out


def test_cli_models_list_and_use(tmp_path, project_root, capsys):
    config = _config(tmp_path, project_root)

    assert main(["models", "list", "--config", str(config)]) == 0
    out = capsys.readouterr().out
    assert "deepseek" in out and "local" in out

    code = main(
        [
            "models",
            "use",
            "--kind",
            "chat",
            "--provider",
            "deepseek",
            "--model",
            "deepseek-chat",
            "--config",
            str(config),
        ]
    )
    assert code == 0
    settings = Settings.load(config)
    assert settings.chat_provider == "deepseek"
    assert settings.chat_model == "deepseek-chat"


def test_models_use_can_disable_chat(tmp_path, project_root, capsys):
    config = _config(tmp_path, project_root)
    settings = Settings.load(config)
    settings.chat_provider = "deepseek"
    settings.chat_model = "deepseek-chat"
    settings.save(config)

    code = main(
        [
            "models", "use", "--kind", "chat", "--provider", "none",
            "--config", str(config),
        ]
    )
    assert code == 0
    settings = Settings.load(config)
    assert settings.chat_provider == ""
    assert settings.chat_model == ""


def test_cli_ask_requires_chat_provider(tmp_path, project_root, capsys):
    config = _config(tmp_path, project_root)
    txt = tmp_path / "notes.txt"
    txt.write_text("检索增强生成需要检索器和生成器。", encoding="utf-8")
    assert main(["ingest", "--source", "txt", "--path", str(txt), "--config", str(config)]) == 0
    capsys.readouterr()

    code = main(["ask", "检索器", "--config", str(config)])
    captured = capsys.readouterr()
    assert code == 2
    assert "CONFIG_ERROR" in captured.err
    assert "notes.txt" in captured.out


def test_main_without_subcommand_non_tty_prints_help(capsys):
    code = main([])
    assert code == 2
    assert "usage" in capsys.readouterr().err.lower()


def test_models_use_embedding_resets_provider_defaults(tmp_path, project_root, capsys):
    config = _config(tmp_path, project_root)
    settings = Settings.load(config)
    settings.embed_backend = "local"
    settings.embed_model = "BAAI/bge-small-zh-v1.5"
    settings.embed_dim = 512
    settings.save(config)

    code = main(
        [
            "models", "use", "--kind", "embedding",
            "--provider", "dashscope", "--model", "text-embedding-v4",
            "--dim", "1024", "--config", str(config),
        ]
    )
    assert code == 0
    settings = Settings.load(config)
    assert settings.embed_backend == "dashscope"
    assert settings.embed_model == "text-embedding-v4"
    assert settings.embed_dim == 1024


def test_cli_ingest_directory_shows_progress(tmp_path, project_root, capsys):
    config = _config(tmp_path, project_root)
    directory = tmp_path / "docs"
    directory.mkdir()
    (directory / "a.txt").write_text("第一个文档。", encoding="utf-8")
    (directory / "b.txt").write_text("第二个文档。", encoding="utf-8")

    code = main(
        ["ingest", "--source", "txt", "--path", str(directory),
         "--config", str(config)]
    )
    err = capsys.readouterr().err
    assert code == 0
    assert "[1/2]" in err
    assert "[2/2]" in err


def test_cli_ingest_strict_fails_on_issue(tmp_path, project_root, capsys):
    config = _config(tmp_path, project_root)
    directory = tmp_path / "docs"
    directory.mkdir()
    (directory / "ok.txt").write_text("正常文档。", encoding="utf-8")
    (directory / "empty.txt").write_text("", encoding="utf-8")  # 空文件 -> EMPTY_FILE

    code = main(
        ["ingest", "--source", "txt", "--path", str(directory),
         "--strict", "--config", str(config)]
    )
    assert code == 1  # 有 issue，strict 模式返回非 0


def test_cli_ask_llm_error_keeps_evidence(tmp_path, project_root, capsys, monkeypatch):
    config = _config(tmp_path, project_root)
    txt = tmp_path / "notes.txt"
    txt.write_text("检索增强生成需要检索器和生成器。", encoding="utf-8")
    assert main(
        ["ingest", "--source", "txt", "--path", str(txt), "--config", str(config)]
    ) == 0
    capsys.readouterr()

    def boom(*args, **kwargs):
        raise RuntimeError("HTTP Error 401: Unauthorized")

    monkeypatch.setattr("rag_multidocs.cli.commands.ask.build_chat_model", boom)
    code = main(
        ["ask", "检索器", "--provider", "deepseek", "--config", str(config)]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert "[LLM_ERROR]" in captured.err
    assert "Key" in captured.err  # 401 -> 建议含"Key"
    assert "检索证据已保留" in captured.out
    assert "notes.txt" in captured.out  # 证据（hits）被保留


def test_cli_docs_show_and_chunks(tmp_path, project_root, capsys):
    from rag_multidocs.cli import build_runtime

    config = _config(tmp_path, project_root)
    txt = tmp_path / "notes.txt"
    txt.write_text("检索增强生成需要检索器和生成器。", encoding="utf-8")
    assert main(
        ["ingest", "--source", "txt", "--path", str(txt), "--config", str(config)]
    ) == 0
    capsys.readouterr()

    runtime = build_runtime(Settings.load(config))
    doc_id = runtime.index.list_docs()[0]["doc_id"]
    chunk_id = runtime.index.list_chunks(doc_id=doc_id)[0]["id"]

    assert main(["docs", "show", doc_id, "--config", str(config)]) == 0
    out = capsys.readouterr().out
    assert "notes.txt" in out and "chunks:" in out

    assert main(
        ["chunks", "list", "--doc-id", doc_id, "--config", str(config)]
    ) == 0
    out = capsys.readouterr().out
    assert chunk_id in out

    assert main(["chunks", "show", chunk_id, "--config", str(config)]) == 0
    out = capsys.readouterr().out
    assert "检索增强生成" in out
