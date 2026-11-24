"""Discover model IDs from a provider's OpenAI-compatible API.

Model names change frequently, so they should not be hardcoded in the CLI.
This module asks the provider for its model list when possible:

- OpenAI-compatible: ``GET {base_url}/models``
- Ollama: ``GET {base_url}/models``; fallback to ``/api/tags``

Providers that cannot list models (local sentence-transformers, hashing) return
their configured default model. If a remote provider has no key, discovery
raises a clear error instead of guessing.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from .providers import (
    ChatProvider,
    EmbeddingProvider,
    api_key,
    chat_provider,
    embedding_provider,
    key_available,
)


class DiscoveryError(RuntimeError):
    """Provider is reachable in principle, but model discovery failed."""


def _resolve(
    name: str, kind: str | None
) -> ChatProvider | EmbeddingProvider:
    if kind == "chat":
        return chat_provider(name)
    if kind == "embedding":
        return embedding_provider(name)
    try:
        return embedding_provider(name)
    except ValueError:
        return chat_provider(name)


def _models_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/embeddings"):
        base = base[: -len("/embeddings")]
    return f"{base}/models"


def _tags_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    return f"{base}/api/tags"


def _extract_models(payload: dict[str, Any]) -> list[str]:
    rows = payload.get("data")
    if isinstance(rows, list):
        names = [
            str(row.get("id") or row.get("name") or "")
            for row in rows
            if isinstance(row, dict)
        ]
    else:
        rows = payload.get("models")
        names = [
            str(row.get("name") or row.get("id") or "")
            for row in rows
            if isinstance(row, dict)
        ] if isinstance(rows, list) else []
    return sorted({name for name in names if name})


def _fetch(url: str, headers: dict[str, str], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "replace")[:200]
        except Exception:
            pass
        raise DiscoveryError(f"HTTP {exc.code} {url} {detail}".strip()) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise DiscoveryError(f"无法访问 {url}: {exc}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DiscoveryError(f"{url} 返回的不是有效 JSON") from exc
    if not isinstance(data, dict):
        raise DiscoveryError(f"{url} 返回格式不是对象")
    return data


def discover_models(
    provider_name: str,
    *,
    kind: str | None = None,
    timeout: float = 10.0,
) -> list[str]:
    """Return model IDs for a provider, or its default for local providers."""
    provider = _resolve(provider_name, kind)
    provider_kind = getattr(provider, "kind", "openai")
    if provider_kind in {"local", "hashing"}:
        return [provider.default_model]

    env_name = provider.api_key_env
    if env_name and not key_available(provider):
        raise DiscoveryError(
            f"未设置 {env_name}，无法查询模型列表；"
            f"先运行 rag-cli keys set --provider {provider.name}"
        )

    headers = {"Accept": "application/json"}
    if env_name:
        headers["Authorization"] = f"Bearer {api_key(provider)}"

    try:
        payload = _fetch(_models_url(provider.base_url), headers, timeout)
        return _extract_models(payload)
    except DiscoveryError as first_error:
        # Ollama 的 OpenAI 兼容层不一定暴露 /v1/models 的全部内容；
        # 回退到原生 /api/tags。
        if provider.name != "ollama":
            raise
        try:
            payload = _fetch(_tags_url(provider.base_url), headers, timeout)
            return _extract_models(payload)
        except DiscoveryError as second_error:
            raise DiscoveryError(
                f"{first_error}；Ollama tags 也失败: {second_error}"
            ) from second_error


__all__ = ["DiscoveryError", "discover_models"]
