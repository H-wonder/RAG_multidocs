"""运行时构建：注册表、索引与检索/入库服务。"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass

from ..config import Settings
from ..connectors.registry import ConnectorRegistry
from ..connectors.txt import TxtConnector
from ..index import KeywordIndex
from ..ingestion import IngestService
from ..retrieval import SearchService


@dataclass
class Runtime:
    registry: ConnectorRegistry
    index: object
    ingest: IngestService
    search: SearchService
    backend: str
    warning: str = ""


def _keyword_index(settings: Settings) -> KeywordIndex:
    if settings.index_file.exists():
        return KeywordIndex.load(settings.index_file)
    return KeywordIndex(path=settings.index_file)


def _build_index(settings: Settings):
    if settings.index_backend == "keyword":
        return _keyword_index(settings), "keyword", ""

    if settings.index_backend != "milvus":
        raise ValueError(
            f"未知 index_backend: {settings.index_backend}（可选 milvus/keyword）"
        )

    try:
        from ..providers.embedding import build_embedder
        from ..index.milvus import MilvusIndex

        index = MilvusIndex(
            build_embedder(
                settings.embed_backend,
                model=settings.embed_model,
                dim=settings.embed_dim,
                base_url=settings.embed_base_url,
            ),
            uri=settings.milvus_uri,
            db_name=settings.milvus_db,
            collection=settings.milvus_collection,
            dense_weight=settings.milvus_dense_weight,
            sparse_weight=settings.milvus_sparse_weight,
        )
        index.ensure_ready()
        return index, "milvus", ""
    except Exception as exc:
        if not settings.fallback_to_keyword:
            raise
        warning = f"Milvus 后端不可用，已回退 keyword: {exc}"
        return _keyword_index(settings), "keyword", warning


def build_registry(settings: Settings) -> ConnectorRegistry:
    registry = ConnectorRegistry(
        plugins_dir=settings.connectors_dir,
        enabled=set(settings.enabled_connectors),
        context={"settings": asdict(settings)},
    )
    # 核心只内置 TXT；PDF/Markdown/HTML 都从 connectors_dir 以插件方式加载。
    registry.register(TxtConnector())
    return registry


def build_runtime(settings: Settings) -> Runtime:
    registry = build_registry(settings)
    index, backend, warning = _build_index(settings)
    return Runtime(
        registry=registry,
        index=index,
        ingest=IngestService(registry, index),
        search=SearchService(index),
        backend=backend,
        warning=warning,
    )


def print_issues(issues) -> None:
    for issue in issues:
        location = f" ({issue.source})" if issue.source else ""
        print(f"[{issue.code}] {issue.message}{location}", file=sys.stderr)


__all__ = [
    "Runtime",
    "build_registry",
    "build_runtime",
    "print_issues",
]
