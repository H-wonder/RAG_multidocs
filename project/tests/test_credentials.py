import json
import os

import pytest

from rag_multidocs.cli import main
from rag_multidocs.config import Settings
from rag_multidocs.credentials import (
    apply_keys,
    credentials_path,
    load_keys,
    mask,
    permission_warning,
    save_keys,
    set_key,
    source_of,
    unset_key,
)


PROVIDER_ENV = ("DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY", "OPENAI_API_KEY")


@pytest.fixture(autouse=True)
def clean_provider_env(monkeypatch):
    for name in PROVIDER_ENV:
        monkeypatch.delenv(name, raising=False)


def _config(tmp_path):
    settings = Settings(
        data_dir=str(tmp_path / "data"),
        credentials_file=str(tmp_path / "credentials.json"),
    )
    path = tmp_path / "config.json"
    settings.save(path)
    return path


def test_set_key_writes_owner_only_file(tmp_path):
    path = tmp_path / "nested" / "credentials.json"
    set_key("DEEPSEEK_API_KEY", "sk-test-1234567890", path)

    assert load_keys(path) == {"DEEPSEEK_API_KEY": "sk-test-1234567890"}
    if os.name == "posix":
        assert (path.stat().st_mode & 0o777) == 0o600
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] == 1


def test_apply_keys_does_not_override_existing_env(tmp_path, monkeypatch):
    path = tmp_path / "credentials.json"
    save_keys({"DEEPSEEK_API_KEY": "from-file"}, path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "from-env")

    applied = apply_keys(path)

    assert "DEEPSEEK_API_KEY" not in applied
    assert os.environ["DEEPSEEK_API_KEY"] == "from-env"
    assert source_of("DEEPSEEK_API_KEY") == "env"


def test_apply_keys_loads_file_and_reports_source(tmp_path, monkeypatch):
    path = tmp_path / "credentials.json"
    save_keys({"OPENAI_API_KEY": "sk-openai-123456"}, path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    applied = apply_keys(path)

    assert applied == ["OPENAI_API_KEY"]
    assert os.environ["OPENAI_API_KEY"] == "sk-openai-123456"
    assert source_of("OPENAI_API_KEY") == "file"


def test_unset_key_removes_file_entry(tmp_path, monkeypatch):
    path = tmp_path / "credentials.json"
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    set_key("DEEPSEEK_API_KEY", "sk-test-1234567890", path)

    assert unset_key("DEEPSEEK_API_KEY", path) is True
    assert load_keys(path) == {}
    assert os.environ.get("DEEPSEEK_API_KEY") is None
    assert unset_key("DEEPSEEK_API_KEY", path) is False


def test_malformed_file_raises_value_error(tmp_path):
    path = tmp_path / "credentials.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        load_keys(path)


def test_permission_warning(tmp_path):
    path = tmp_path / "credentials.json"
    save_keys({"DEEPSEEK_API_KEY": "sk-test-1234567890"}, path)
    assert permission_warning(path) == ""
    if os.name == "posix":
        os.chmod(path, 0o644)
        assert "权限过宽" in permission_warning(path)


def test_mask_never_returns_full_key():
    assert mask("") == ""
    assert mask("short") == "****"
    masked = mask("sk-1234567890abcdef")
    assert masked == "sk-...cdef"
    assert "1234567890" not in masked


def test_credentials_path_env_override(tmp_path, monkeypatch):
    target = tmp_path / "custom.json"
    monkeypatch.setenv("RAG_CREDENTIALS", str(target))
    assert credentials_path(".rag/credentials.json") == target


def test_cli_keys_set_list_unset(tmp_path, capsys, monkeypatch):
    config = _config(tmp_path)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    code = main(
        [
            "keys", "set", "--provider", "deepseek",
            "--value", "sk-test-1234567890", "--config", str(config),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "DEEPSEEK_API_KEY" in out
    assert "sk-test-1234567890" not in out

    # 模拟新进程：清除进程内变量，让 list 从凭证文件加载
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    code = main(["keys", "list", "--config", str(config)])
    out = capsys.readouterr().out
    assert code == 0
    assert "DEEPSEEK_API_KEY" in out and "file" in out
    assert "sk-test-1234567890" not in out

    code = main(
        ["keys", "unset", "--provider", "deepseek", "--config", str(config)]
    )
    assert code == 0
    assert load_keys(tmp_path / "credentials.json") == {}


def test_cli_keys_set_hidden_input_path(tmp_path, capsys, monkeypatch):
    from argparse import Namespace

    from rag_multidocs.cli import cmd_keys_set

    class FakeTTY:
        def isatty(self):
            return True

    config = _config(tmp_path)
    monkeypatch.setattr(
        "rag_multidocs.cli.getpass.getpass", lambda prompt="": "sk-hidden-123456"
    )
    monkeypatch.setattr("rag_multidocs.cli.sys.stdin", FakeTTY())

    args = Namespace(config=str(config), provider="deepseek", env=None, value=None)
    assert cmd_keys_set(args) == 0
    assert load_keys(tmp_path / "credentials.json") == {
        "DEEPSEEK_API_KEY": "sk-hidden-123456"
    }
    assert "sk-hidden-123456" not in capsys.readouterr().out


def test_cli_keys_export_prints_shell_assignments(tmp_path, capsys, monkeypatch):
    config = _config(tmp_path)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    save_keys(
        {"DEEPSEEK_API_KEY": "sk-test-1234567890"},
        tmp_path / "credentials.json",
    )

    code = main(["keys", "export", "--config", str(config)])
    captured = capsys.readouterr()
    assert code == 0
    assert "export DEEPSEEK_API_KEY=" in captured.out
    assert "明文" in captured.err


def test_cli_keys_set_requires_provider(tmp_path, capsys):
    config = _config(tmp_path)
    code = main(["keys", "set", "--env", "bad-name", "--value", "x", "--config", str(config)])
    assert code == 2
    assert "CONFIG_ERROR" in capsys.readouterr().err
