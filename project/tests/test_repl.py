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
    from rag_multidocs.cli.repl import render as repl_render

    monkeypatch.setattr(repl_render, "readline", object())
    monkeypatch.setattr(
        repl_render, "color",
        lambda text, code: f"\033[{code}m{text}\033[0m",
    )

    prompt = repl_render.input_prompt("rag> ", "1;32")

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


def test_key_summary_distinguishes_available(tmp_path, monkeypatch):
    session = Session(str(_config(tmp_path)))
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert session._key_summary("DEEPSEEK_API_KEY") == "不可用"
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-abc123")
    assert session._key_summary("DEEPSEEK_API_KEY").startswith("可用")
    assert session._key_summary("") == "无需"


def test_model_embedding_hints_collection_keyword(tmp_path, capsys):
    session = Session(str(_config(tmp_path)))
    session.dispatch("/model embedding local")
    out = capsys.readouterr().out
    assert "keyword 模式" in out


def test_model_embedding_hints_collection_milvus(tmp_path, capsys):
    settings = Settings(
        data_dir=str(tmp_path / "data"),
        index_backend="milvus",
        credentials_file=str(tmp_path / "credentials.json"),
    )
    path = tmp_path / "config.json"
    settings.save(path)
    session = Session(str(path))
    session.dispatch("/model embedding local")
    out = capsys.readouterr().out
    assert "rag_chunks" in out
    assert "建议新集合" in out


def test_repl_last_and_open(tmp_path, capsys):
    config = _config(tmp_path)
    txt = tmp_path / "notes.txt"
    txt.write_text("检索增强生成需要检索器和生成器。", encoding="utf-8")

    session = Session(str(config))
    session.dispatch(f"/ingest txt {txt}")
    capsys.readouterr()
    session.dispatch("/search 检索器")
    capsys.readouterr()

    session.dispatch("/last")
    assert "notes.txt" in capsys.readouterr().out

    session.dispatch("/open 1")
    out = capsys.readouterr().out
    assert "source:" in out and "notes.txt" in out


def test_repl_docs_detail(tmp_path, capsys):
    config = _config(tmp_path)
    txt = tmp_path / "notes.txt"
    txt.write_text("检索增强生成。", encoding="utf-8")

    session = Session(str(config))
    session.dispatch(f"/ingest txt {txt}")
    capsys.readouterr()

    doc_id = session.runtime.index.list_docs()[0]["doc_id"]
    session.dispatch(f"/docs {doc_id}")
    out = capsys.readouterr().out
    assert "doc_id:" in out and "notes.txt" in out


def test_repl_last_empty_hint(tmp_path, capsys):
    session = Session(str(_config(tmp_path)))
    session.dispatch("/last")
    assert "还没有检索过" in capsys.readouterr().err
