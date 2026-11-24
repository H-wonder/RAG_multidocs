"""Model provider presets.

API keys come from environment variables.  The CLI additionally loads a
project-local, gitignored credentials file into the process environment at
startup (see ``credentials.py``); an already-set environment variable always
takes precedence.  Keys are never written to config.json or the repository.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingProvider:
    name: str
    kind: str                 # local | openai | hashing
    default_model: str
    default_dim: int
    base_url: str = ""
    api_key_env: str = ""
    description: str = ""

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "default_model": self.default_model,
            "default_dim": self.default_dim,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "description": self.description,
        }


@dataclass(frozen=True)
class ChatProvider:
    name: str
    default_model: str
    base_url: str
    api_key_env: str = ""
    description: str = ""

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "default_model": self.default_model,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "description": self.description,
        }


EMBEDDING_PROVIDERS: dict[str, EmbeddingProvider] = {
    "local": EmbeddingProvider(
        name="local",
        kind="local",
        default_model="BAAI/bge-small-zh-v1.5",
        default_dim=512,
        description="本地 sentence-transformers（离线，首次下载模型）",
    ),
    "hashing": EmbeddingProvider(
        name="hashing",
        kind="hashing",
        default_model="hashing",
        default_dim=256,
        description="确定性哈希，仅 CI/冒烟，无语义能力",
    ),
    "dashscope": EmbeddingProvider(
        name="dashscope",
        kind="openai",
        default_model="text-embedding-v4",
        default_dim=1024,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key_env="DASHSCOPE_API_KEY",
        description="阿里百炼 OpenAI 兼容 embeddings",
    ),
    "openai": EmbeddingProvider(
        name="openai",
        kind="openai",
        default_model="text-embedding-3-small",
        default_dim=1536,
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        description="OpenAI embeddings",
    ),
    "ollama": EmbeddingProvider(
        name="ollama",
        kind="openai",
        default_model="nomic-embed-text",
        default_dim=768,
        base_url="http://127.0.0.1:11434/v1",
        description="本地 Ollama OpenAI 兼容 embeddings（无需 Key）",
    ),
}


CHAT_PROVIDERS: dict[str, ChatProvider] = {
    "deepseek": ChatProvider(
        name="deepseek",
        default_model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
        description="DeepSeek Chat",
    ),
    "dashscope": ChatProvider(
        name="dashscope",
        default_model="qwen-plus",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key_env="DASHSCOPE_API_KEY",
        description="阿里百炼通义千问（OpenAI 兼容）",
    ),
    "openai": ChatProvider(
        name="openai",
        default_model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        description="OpenAI Chat",
    ),
    "ollama": ChatProvider(
        name="ollama",
        default_model="qwen3.5:4b",
        base_url="http://127.0.0.1:11434/v1",
        description="本地 Ollama Chat（无需 Key）",
    ),
}


def embedding_provider(name: str) -> EmbeddingProvider:
    try:
        return EMBEDDING_PROVIDERS[name]
    except KeyError as exc:
        raise ValueError(
            f"未知 embedding provider: {name}"
            f"（可选: {', '.join(EMBEDDING_PROVIDERS)}）"
        ) from exc


def chat_provider(name: str) -> ChatProvider:
    try:
        return CHAT_PROVIDERS[name]
    except KeyError as exc:
        raise ValueError(
            f"未知 chat provider: {name}"
            f"（可选: {', '.join(CHAT_PROVIDERS)}）"
        ) from exc


def api_key(provider: EmbeddingProvider | ChatProvider) -> str:
    """Read the key from env; local providers may not need one."""
    if not provider.api_key_env:
        return "local"
    value = os.environ.get(provider.api_key_env, "")
    if not value:
        raise ValueError(
            f"未设置 {provider.api_key_env}。可以执行 "
            f"`rag-cli keys set --provider <provider>` 保存到本项目，"
            f"或临时执行 export {provider.api_key_env}=..."
        )
    return value


def key_available(provider: EmbeddingProvider | ChatProvider) -> bool:
    if not provider.api_key_env:
        return True
    return bool(os.environ.get(provider.api_key_env))


def env_var_for_provider(provider_name: str) -> str:
    """Return the API-key environment variable for a provider name."""
    preset = CHAT_PROVIDERS.get(provider_name) or EMBEDDING_PROVIDERS.get(provider_name)
    if preset is None:
        raise ValueError(
            f"未知 provider: {provider_name}"
            f"（可选 chat: {', '.join(CHAT_PROVIDERS)}；"
            f"embedding: {', '.join(EMBEDDING_PROVIDERS)}）"
        )
    if not preset.api_key_env:
        raise ValueError(f"provider {provider_name} 不需要 API Key")
    return preset.api_key_env


def providers_for_env(env_name: str) -> list[str]:
    names: list[str] = []
    for preset in (*CHAT_PROVIDERS.values(), *EMBEDDING_PROVIDERS.values()):
        if preset.api_key_env == env_name and preset.name not in names:
            names.append(preset.name)
    return names


def list_embedding_providers() -> list[dict]:
    return [provider.as_dict() for provider in EMBEDDING_PROVIDERS.values()]


def list_chat_providers() -> list[dict]:
    return [provider.as_dict() for provider in CHAT_PROVIDERS.values()]


__all__ = [
    "CHAT_PROVIDERS",
    "EMBEDDING_PROVIDERS",
    "ChatProvider",
    "EmbeddingProvider",
    "api_key",
    "chat_provider",
    "embedding_provider",
    "env_var_for_provider",
    "key_available",
    "list_chat_providers",
    "list_embedding_providers",
    "providers_for_env",
]
