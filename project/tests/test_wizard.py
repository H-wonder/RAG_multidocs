from rag_multidocs.config import Settings
from rag_multidocs.credentials import load_keys
from rag_multidocs.wizard import run_setup


class FakeTTY:
    def isatty(self):
        return True


def test_setup_wizard_saves_chat_and_key(tmp_path, monkeypatch):
    monkeypatch.setattr("rag_multidocs.wizard.sys.stdin", FakeTTY())
    config = tmp_path / "config.json"
    settings = Settings(
        data_dir=str(tmp_path / "data"),
        credentials_file=str(tmp_path / "credentials.json"),
    )

    answers = iter(["1", "", ""])  # chat=deepseek, 默认模型, 空回答给 Key
    outputs = []
    code = run_setup(
        settings,
        config,
        ask=lambda prompt="": next(answers),
        secret=lambda prompt="": "sk-test-1234567890",
        out=outputs.append,
    )

    assert code == 0
    saved = Settings.load(config)
    assert saved.chat_provider == "deepseek"
    assert saved.chat_model == ""
    assert load_keys(tmp_path / "credentials.json") == {
        "DEEPSEEK_API_KEY": "sk-test-1234567890"
    }
    assert any("已保存 DEEPSEEK_API_KEY" in line for line in outputs)


def test_setup_wizard_can_disable_chat(tmp_path, monkeypatch):
    monkeypatch.setattr("rag_multidocs.wizard.sys.stdin", FakeTTY())
    config = tmp_path / "config.json"
    settings = Settings(
        data_dir=str(tmp_path / "data"),
        credentials_file=str(tmp_path / "credentials.json"),
        chat_provider="deepseek",
        chat_model="deepseek-chat",
    )

    answers = iter(["0", ""])  # chat 关闭，embedding 保持
    code = run_setup(
        settings,
        config,
        ask=lambda prompt="": next(answers),
        secret=lambda prompt="": "",
        out=lambda *args: None,
    )

    assert code == 0
    assert Settings.load(config).chat_provider == ""
