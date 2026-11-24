"""Provider presets, embedding backends, and model discovery."""

from .embedding import (
    Embedder,
    HashingEmbedder,
    OpenAICompatibleEmbedder,
    SentenceTransformerEmbedder,
    build_embedder,
)
from .presets import (
    CHAT_PROVIDERS,
    EMBEDDING_PROVIDERS,
    ChatProvider,
    EmbeddingProvider,
    api_key,
    chat_provider,
    embedding_provider,
    env_var_for_provider,
    key_available,
    list_chat_providers,
    list_embedding_providers,
    providers_for_env,
)
from .discovery import DiscoveryError, discover_models

__all__ = [
    "CHAT_PROVIDERS",
    "EMBEDDING_PROVIDERS",
    "ChatProvider",
    "DiscoveryError",
    "Embedder",
    "EmbeddingProvider",
    "HashingEmbedder",
    "OpenAICompatibleEmbedder",
    "SentenceTransformerEmbedder",
    "api_key",
    "build_embedder",
    "chat_provider",
    "discover_models",
    "embedding_provider",
    "env_var_for_provider",
    "key_available",
    "list_chat_providers",
    "list_embedding_providers",
    "providers_for_env",
]
