"""Public base Index contract and optional version/publication capabilities."""

from __future__ import annotations

import inspect
from typing import Protocol, runtime_checkable


class IndexBackend(Protocol):
    """Original base API. Version management is an optional separate capability."""

    name: str

    def ensure_ready(self): ...
    def add(self, chunks, document=None): ...
    def remove_doc(self, doc_id): ...
    def search(self, query, top_k=5, doc_ids=None): ...
    def list_docs(self): ...
    def stats(self): ...
    def persist(self): ...
    def close(self): ...


@runtime_checkable
class VersionedIndex(Protocol):
    def stage_version(self, document, chunks, version): ...
    def count_version(self, doc_id, version): ...
    def remove_version(self, doc_id, version): ...


def supports_versions(index):
    return (
        not getattr(index, "legacy_readonly", False)
        and getattr(index, "versioned_publication", True)
        and isinstance(index, VersionedIndex)
        and all(
            callable(getattr(index, n, None))
            for n in ("stage_version", "count_version", "remove_version")
        )
    )


def search_index(index, query, *, top_k=5, doc_ids=None, filters=None, **options):
    """Call legacy search without passing unsupported new keyword arguments.

    A requested filter/feature must never silently disappear.
    """
    parameters = inspect.signature(index.search).parameters
    variadic = any(p.kind == p.VAR_KEYWORD for p in parameters.values())
    supplied = {"top_k": top_k, "doc_ids": doc_ids, "filters": filters, **options}
    kwargs = {}
    for key, value in supplied.items():
        if key in parameters or variadic:
            kwargs[key] = value
        elif value is not None and key in {"filters", "snapshot"}:
            if value:
                raise ValueError("INDEX_CAPABILITY_UNAVAILABLE: " + key)
    return index.search(query, **kwargs)


def model_identity(embedder):
    if callable(getattr(embedder, "identity", None)):
        return embedder.identity()
    return {
        "provider": type(embedder).__name__,
        "model": getattr(embedder, "model_name", getattr(embedder, "model", "hashing")),
        "dimension": getattr(embedder, "dim", 0),
        "query_instruction": getattr(embedder, "query_instruction", ""),
        "base_url": getattr(embedder, "base_url", ""),
        "revision": getattr(embedder, "revision", None),
        "normalize_embeddings": getattr(embedder, "normalize_embeddings", True),
    }
