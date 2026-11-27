"""Explicit composition for teammate adapters; never imports CLI/REPL/Web."""

from __future__ import annotations

from pathlib import Path

from .connectors.registry import ConnectorRegistry
from .connectors.txt import TxtConnector
from .core import RAGCore
from .index import KeywordIndex
from .retrieval.profiles import recommended_policies


def build_connector_registry(settings, *, base_dir=None, data_dir=None):
    context = settings.resolved_connector_settings(base_dir)
    if data_dir is not None:
        context["data_dir"] = str(data_dir)
    registry = ConnectorRegistry(
        context["connectors_dir"], enabled=set(settings.enabled_connectors),
        context={"settings": context, "source_base": getattr(settings, "_config_dir", base_dir or Path.cwd())},
    )
    registry.register(TxtConnector())
    return registry


def configured_chat_budget(settings):
    from .qa.llm import resolve_chat_budget

    return resolve_chat_budget(
        settings.chat_provider, settings.chat_model, settings.chat_max_tokens,
        settings.chat_context_tokens, base_url=settings.chat_base_url,
    )


def configured_chat(settings, *, budget=None):
    from .qa.llm import build_chat_model

    return build_chat_model(
        settings.chat_provider, settings.chat_model, settings.chat_base_url,
        settings.chat_temperature, settings.chat_max_tokens,
        context_tokens=settings.chat_context_tokens, budget=budget,
    )


def create_core(
    settings,
    *,
    base_dir,
    data_dir=None,
    rerankers=None,
    policy=None,
    offline=False,
    index=None,
    registry=None,
    chat=None,
    rewriter=None,
    chunk_options=None,
):
    """Keep paths independent of CWD; optional models remain lazy.

    A caller may supply an existing Index and Registry to reuse its knowledge base.
    """
    base = Path(base_dir).resolve()
    data = Path(data_dir or settings.data_dir)
    data = (data if data.is_absolute() else base / data).resolve()
    if registry is None:
        registry = build_connector_registry(settings, base_dir=base, data_dir=data)
    if index is None:
        if settings.index_backend == "keyword":
            path = data / "index.json"
            index = KeywordIndex.load(path) if path.exists() else KeywordIndex(path=path)
        elif settings.index_backend == "milvus":
            from .index.milvus import MilvusIndex
            from .providers.embedding import build_embedder

            index = MilvusIndex(
                build_embedder(
                    settings.embed_backend,
                    settings.embed_model,
                    settings.embed_dim,
                    settings.embed_base_url,
                    local_files_only=offline,
                ),
                uri=settings.milvus_uri,
                db_name=settings.milvus_db,
                collection=settings.milvus_collection,
                dense_weight=settings.milvus_dense_weight,
                sparse_weight=settings.milvus_sparse_weight,
            )
            index.ensure_ready()
        else:
            raise ValueError("UNKNOWN_INDEX_BACKEND")
    chat_factory = None
    chat_budget_factory = None
    if chat is None and settings.chat_provider:
        def chat_budget_factory():
            return configured_chat_budget(settings)

        def chat_factory():
            return configured_chat(settings)
    # The normal Milvus path owns its document metadata; SQLite is legacy-only.
    return RAGCore(
        registry,
        index,
        catalog_path=None,
        rerankers=rerankers,
        policy=policy or recommended_policies()['stable'],
        chat=chat,
        chat_factory=chat_factory,
        chat_budget_factory=chat_budget_factory,
        rewriter=rewriter,
        chunk_options=chunk_options,
    )
