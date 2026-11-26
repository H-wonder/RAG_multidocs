"""``rag-cli models test`` 的单元测试：真实调用成功、错误分类、脱敏。"""

from rag_multidocs.cli import main
from rag_multidocs.cli.model_test import (
    render,
    test_chat as _test_chat,
    test_embedding as _test_embedding,
)
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


def test_embedding_ok(tmp_path, project_root, monkeypatch):
    settings = _settings(tmp_path, project_root)

    class FakeEmbedder:
        def embed(self, texts, query=False):
            return [[0.1] * 512]

    monkeypatch.setattr(
        "rag_multidocs.providers.embedding.build_embedder",
        lambda *args, **kwargs: FakeEmbedder(),
    )
    result = _test_embedding(settings, "local", "", "", 0)
    assert result.ok is True
    assert result.dim == 512
    assert result.latency_ms >= 0


def test_chat_ok(tmp_path, project_root, monkeypatch):
    settings = _settings(tmp_path, project_root, chat_provider="deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    class FakeChat:
        model = "deepseek-chat"

        def complete(self, messages):
            return "你好！我是助手。"

    monkeypatch.setattr(
        "rag_multidocs.qa.llm.build_chat_model",
        lambda *args, **kwargs: FakeChat(),
    )
    result = _test_chat(settings, "deepseek", "", "")
    assert result.ok is True
    assert result.model == "deepseek-chat"
    assert "你好" in result.reply


def test_chat_missing_key(tmp_path, project_root, monkeypatch):
    settings = _settings(tmp_path, project_root, chat_provider="deepseek")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    result = _test_chat(settings, "deepseek", "", "")
    assert result.ok is False
    assert "未设置 DEEPSEEK_API_KEY" in result.error
    assert "keys set" in result.hint


def test_embedding_missing_key(tmp_path, project_root, monkeypatch):
    settings = _settings(tmp_path, project_root, embed_backend="dashscope")
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    result = _test_embedding(settings, "dashscope", "", "", 0)
    assert result.ok is False
    assert "未设置 DASHSCOPE_API_KEY" in result.error


def test_chat_timeout_classified(tmp_path, project_root, monkeypatch):
    settings = _settings(tmp_path, project_root, chat_provider="deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")

    class TimeoutChat:
        model = "deepseek-chat"

        def complete(self, messages):
            raise RuntimeError("<urlopen error timed out>")

    monkeypatch.setattr(
        "rag_multidocs.qa.llm.build_chat_model",
        lambda *args, **kwargs: TimeoutChat(),
    )
    result = _test_chat(settings, "deepseek", "", "")
    assert result.ok is False
    assert "超时" in result.hint


def test_chat_model_not_found_classified(tmp_path, project_root, monkeypatch):
    settings = _settings(tmp_path, project_root, chat_provider="deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")

    class NotFoundChat:
        model = "deepseek-chat"

        def complete(self, messages):
            raise RuntimeError("HTTP Error 404: Not Found")

    monkeypatch.setattr(
        "rag_multidocs.qa.llm.build_chat_model",
        lambda *args, **kwargs: NotFoundChat(),
    )
    result = _test_chat(settings, "deepseek", "", "")
    assert result.ok is False
    assert "404" in result.error
    assert "discover" in result.hint


def test_error_does_not_leak_key(tmp_path, project_root, monkeypatch):
    settings = _settings(tmp_path, project_root, chat_provider="deepseek")
    secret = "sk-secret-123456"
    monkeypatch.setenv("DEEPSEEK_API_KEY", secret)

    class LeakyChat:
        model = "deepseek-chat"

        def complete(self, messages):
            raise RuntimeError(f"auth failed for key {secret}")

    monkeypatch.setattr(
        "rag_multidocs.qa.llm.build_chat_model",
        lambda *args, **kwargs: LeakyChat(),
    )
    result = _test_chat(settings, "deepseek", "", "")
    assert result.ok is False
    assert secret not in result.error
    assert "****" in result.error


def test_render_pass_and_fail():
    from rag_multidocs.cli.model_test import ModelTestResult

    ok = ModelTestResult("embedding", "local", "m", True, 123, dim=512)
    assert "dim=512" in render(ok)
    assert "123ms" in render(ok)

    bad = ModelTestResult("chat", "deepseek", "m", False, 0, error="未设置 KEY", hint="set it")
    out = render(bad)
    assert "[FAIL]" in out
    assert "建议: set it" in out


def test_models_test_cli_returns_zero_on_ok(tmp_path, project_root, monkeypatch, capsys):
    config = _config_file(tmp_path, project_root)

    class FakeEmbedder:
        def embed(self, texts, query=False):
            return [[0.0] * 512]

    monkeypatch.setattr(
        "rag_multidocs.providers.embedding.build_embedder",
        lambda *args, **kwargs: FakeEmbedder(),
    )
    code = main(["models", "test", "--kind", "embedding", "--config", str(config)])
    out = capsys.readouterr().out
    assert code == 0
    assert "[PASS]" in out
    assert "dim=512" in out
