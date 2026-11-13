from rag_multidocs.cli import main
from rag_multidocs.config import Settings


def _config(tmp_path, project_root):
    config = Settings(
        data_dir=str(tmp_path / "data"),
        connectors_dir=str(project_root / "connectors"),
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
