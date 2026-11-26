import json
import urllib.error

import pytest

from rag_multidocs.providers import discovery
from rag_multidocs.cli import main
from rag_multidocs.providers.discovery import DiscoveryError, discover_models


class _Response:
    def __init__(self, payload):
        self._data = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_discover_openai_compatible_models(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(
        discovery.urllib.request,
        "urlopen",
        lambda request, timeout: _Response(
            {"data": [{"id": "model-b"}, {"id": "model-a"}]}
        ),
    )

    assert discover_models("openai", kind="chat") == ["model-a", "model-b"]


def test_discover_local_returns_default_model(monkeypatch):
    monkeypatch.setattr(
        discovery.urllib.request,
        "urlopen",
        lambda request, timeout: (_ for _ in ()).throw(AssertionError("no HTTP")),
    )

    assert discover_models("local", kind="embedding") == [
        "BAAI/bge-small-zh-v1.5"
    ]


def test_discover_remote_without_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(DiscoveryError, match="OPENAI_API_KEY"):
        discover_models("openai", kind="chat")


def test_discover_ollama_falls_back_to_tags(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        if request.full_url.endswith("/v1/models"):
            raise urllib.error.HTTPError(
                request.full_url, 404, "not found", {}, None
            )
        return _Response({"models": [{"name": "qwen3.5:4b"}, {"name": "llama3"}]})

    monkeypatch.setattr(discovery.urllib.request, "urlopen", fake_urlopen)

    assert discover_models("ollama", kind="chat") == ["llama3", "qwen3.5:4b"]
    assert calls[0].endswith("/v1/models")
    assert calls[1].endswith("/api/tags")


def test_cli_models_discover(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(
        "rag_multidocs.cli.commands.models.discover_models",
        lambda provider, kind=None, timeout=10.0: ["m1", "m2"],
    )

    code = main(
        [
            "models",
            "discover",
            "--provider",
            "ollama",
            "--config",
            str(tmp_path / "config.json"),
        ]
    )

    assert code == 0
    assert capsys.readouterr().out.splitlines() == ["m1", "m2"]
