import pytest

from rag_multidocs.embedding import build_embedder
from rag_multidocs.llm import build_chat_model
from rag_multidocs.providers import (
    chat_provider,
    embedding_provider,
    key_available,
    list_chat_providers,
    list_embedding_providers,
)


def test_hashing_embedder_uses_requested_dim():
    embedder = build_embedder("hashing", dim=8)
    assert embedder.dim == 8
    assert len(embedder.embed(["x"])[0]) == 8


def test_provider_lists_include_expected_names():
    assert {item["name"] for item in list_embedding_providers()} >= {
        "local",
        "dashscope",
        "openai",
        "ollama",
        "hashing",
    }
    assert {item["name"] for item in list_chat_providers()} >= {
        "deepseek",
        "dashscope",
        "openai",
        "ollama",
    }
    assert embedding_provider("local").default_dim == 512
    assert chat_provider("deepseek").api_key_env == "DEEPSEEK_API_KEY"


def test_chat_provider_requires_env_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert not key_available(chat_provider("deepseek"))
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        build_chat_model("deepseek")

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    assert key_available(chat_provider("deepseek"))
    model = build_chat_model("deepseek", model="deepseek-chat")
    assert model.model == "deepseek-chat"
