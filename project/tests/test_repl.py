import rag_multidocs.cli.repl as repl
from rag_multidocs.config import Settings
from rag_multidocs.cli.repl import Session


def _config(tmp_path):
    settings = Settings(
        data_dir=str(tmp_path / "data"),
        index_backend="keyword",
        credentials_file=str(tmp_path / "credentials.json"),
    )
    path = tmp_path / "config.json"
    settings.save(path)
    return path


def test_repl_help_lists_commands(tmp_path, capsys):
    session = Session(str(_config(tmp_path)))
    session.dispatch("/help")
    out = capsys.readouterr().out
    assert "/keys set" in out
    assert "/ingest" in out
    assert "/setup" in out


def test_repl_unknown_command_is_reported(tmp_path, capsys):
    session = Session(str(_config(tmp_path)))
    session.dispatch("/nope")
    assert "未知命令" in capsys.readouterr().err


def test_repl_model_chat_persists_config(tmp_path, capsys):
    config = _config(tmp_path)
    session = Session(str(config))
    session.dispatch("/model chat deepseek deepseek-chat")
    capsys.readouterr()

    settings = Settings.load(config)
    assert settings.chat_provider == "deepseek"
    assert settings.chat_model == "deepseek-chat"


def test_repl_exit_stops_loop(tmp_path):
    session = Session(str(_config(tmp_path)))
    session.dispatch("/exit")
    assert session._running is False


def test_repl_bare_command_is_dispatched(tmp_path, capsys):
    session = Session(str(_config(tmp_path)))
    session.dispatch("config")
    captured = capsys.readouterr()
    assert "index:" in captured.out
    assert "可以写成 /config" in captured.err


def test_repl_bare_exit_quits_without_hint(tmp_path, capsys):
    session = Session(str(_config(tmp_path)))
    session.dispatch("exit")
    assert session._running is False
    assert "提示" not in capsys.readouterr().err


def test_repl_bare_q_alias_quits(tmp_path):
    session = Session(str(_config(tmp_path)))
    session.dispatch("q")
    assert session._running is False


def test_repl_prompt_marks_ansi_as_nonprinting(monkeypatch):
    monkeypatch.setattr(repl, "readline", object())
    monkeypatch.setattr(
        repl, "_color", lambda text, code: f"\033[{code}m{text}\033[0m"
    )

    prompt = repl._input_prompt("rag> ", "1;32")

    assert prompt.startswith("\001")
    assert prompt.endswith("\002")
    assert "rag> " in prompt


def test_model_picker_uses_local_default(monkeypatch, tmp_path):
    session = Session(str(_config(tmp_path)))
    answers = iter(["1", ""])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))

    model, dim = session._pick_model_name("embedding", "local")

    assert model == ""
    assert dim is None


def test_model_picker_accepts_custom_model_and_dim(monkeypatch, tmp_path):
    session = Session(str(_config(tmp_path)))
    answers = iter(["c", "my-embed-model", "1024"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))

    model, dim = session._pick_model_name("embedding", "local")

    assert model == "my-embed-model"
    assert dim == 1024


def test_repl_help_lists_new_commands(tmp_path, capsys):
    session = Session(str(_config(tmp_path)))
    session.dispatch("/help")
    out = capsys.readouterr().out
    assert "/doctor" in out
    assert "/models test" in out
    assert "/config validate" in out


def test_repl_doctor_outputs_diagnosis(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(
        "rag_multidocs.cli.doctor._tcp_reachable",
        lambda host, port, timeout=2.0: False,
    )
    session = Session(str(_config(tmp_path)))
    session.dispatch("/doctor")
    out = capsys.readouterr().out
    assert any(tag in out for tag in ("[PASS]", "[WARN]", "[FAIL]"))


def test_repl_config_show(tmp_path, capsys):
    session = Session(str(_config(tmp_path)))
    session.dispatch("/config show")
    out = capsys.readouterr().out
    assert "index_backend" in out
    assert "keyword" in out


def test_repl_config_validate(tmp_path, capsys):
    session = Session(str(_config(tmp_path)))
    session.dispatch("/config validate")
    assert "配置有效" in capsys.readouterr().out


def test_repl_config_profiles_list(tmp_path, capsys):
    session = Session(str(_config(tmp_path)))
    session.dispatch("/config profiles list")
    out = capsys.readouterr().out
    assert "offline" in out and "local" in out and "cloud" in out


def test_repl_models_test_embedding(tmp_path, capsys, monkeypatch):
    class FakeEmbedder:
        def embed(self, texts, query=False):
            return [[0.0] * 512]

    monkeypatch.setattr(
        "rag_multidocs.providers.embedding.build_embedder",
        lambda *args, **kwargs: FakeEmbedder(),
    )
    session = Session(str(_config(tmp_path)))
    session.dispatch("/models test embedding")
    out = capsys.readouterr().out
    assert "dim=512" in out
