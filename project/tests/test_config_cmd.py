"""``rag-cli config``（show / validate / profiles）的单元测试。"""

from rag_multidocs.cli import main
from rag_multidocs.cli.config_cmd import validate_settings
from rag_multidocs.config import Settings


def _settings(tmp_path, project_root, **overrides):
    values = dict(
        data_dir=str(tmp_path / "data"),
        connectors_dir=str(project_root / "connectors"),
        credentials_file=str(tmp_path / ".rag" / "credentials.json"),
        index_backend="keyword",
    )
    values.update(overrides)
    return Settings(**values)


def _config_file(tmp_path, project_root, **overrides):
    settings = _settings(tmp_path, project_root, **overrides)
    path = tmp_path / "config.json"
    settings.save(path)
    return path


def test_validate_ok(tmp_path, project_root):
    settings = _settings(tmp_path, project_root)
    assert validate_settings(settings) == []


def test_validate_bad_provider(tmp_path, project_root):
    settings = _settings(tmp_path, project_root, embed_backend="nope")
    issues = validate_settings(settings)
    assert any(i.level == "fail" and "embed_backend" in i.message for i in issues)


def test_validate_missing_docparse_path(tmp_path, project_root):
    settings = _settings(
        tmp_path, project_root,
        docparse_repo=str(tmp_path / "missing"),
    )
    issues = validate_settings(settings)
    assert any(i.level == "fail" and "docparse_repo" in i.message for i in issues)


def test_validate_legacy_env_var(tmp_path, project_root, monkeypatch):
    monkeypatch.setenv("PLATFORM_INDEX_BACKEND", "keyword")
    settings = _settings(tmp_path, project_root)
    issues = validate_settings(settings)
    assert any("PLATFORM_INDEX_BACKEND" in i.message for i in issues)


def test_config_show_cli(tmp_path, project_root, capsys):
    config = _config_file(tmp_path, project_root)
    code = main(["config", "show", "--config", str(config)])
    out = capsys.readouterr().out
    assert code == 0
    assert "index_backend" in out
    assert "keyword" in out


def test_config_validate_cli_passes(tmp_path, project_root, capsys):
    config = _config_file(tmp_path, project_root)
    code = main(["config", "validate", "--config", str(config)])
    out = capsys.readouterr().out
    assert code == 0
    assert "[PASS] 配置有效" in out


def test_config_validate_cli_fails(tmp_path, project_root, capsys):
    config = _config_file(tmp_path, project_root, embed_backend="nope")
    code = main(["config", "validate", "--config", str(config)])
    out = capsys.readouterr().out
    assert code == 1
    assert "[FAIL]" in out


def test_profiles_list_cli(tmp_path, project_root, capsys):
    config = _config_file(tmp_path, project_root)
    code = main(["config", "profiles", "list", "--config", str(config)])
    out = capsys.readouterr().out
    assert code == 0
    assert "offline" in out and "local" in out and "cloud" in out


def test_profiles_apply_cli(tmp_path, project_root, capsys):
    config = _config_file(tmp_path, project_root)
    code = main(["config", "profiles", "apply", "offline", "--config", str(config)])
    assert code == 0
    settings = Settings.load(config)
    assert settings.index_backend == "keyword"
    assert settings.embed_backend == "hashing"
    assert settings.chat_provider == ""


def test_profiles_apply_unknown(tmp_path, project_root, capsys):
    config = _config_file(tmp_path, project_root)
    code = main(["config", "profiles", "apply", "nope", "--config", str(config)])
    assert code == 2
    assert "未知 profile" in capsys.readouterr().err
