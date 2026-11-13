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
        "rag_multidocs.cli.build_chat_model",
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
