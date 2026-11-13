"""Command line interface for RAG Multidocs."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass

from .config import Settings
from .connectors.pdf import DocparseOptions, PdfConnector
from .connectors.txt import TxtConnector
from .contracts import ConnectorError
from .index import KeywordIndex
from .ingest import IngestService
from .registry import ConnectorRegistry
from .search import SearchService


def add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default="config.json",
        help="配置文件路径（默认 config.json；不存在则使用默认值）",
    )


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
        from .embedding import build_embedder
        from .index_milvus import MilvusIndex

        index = MilvusIndex(
            build_embedder(
                settings.embed_backend,
                model=settings.embed_model,
                dim=settings.embed_dim,
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
    )
    registry.register(TxtConnector())
    registry.register(
        PdfConnector(
            DocparseOptions(
                repo=settings.docparse_repo,
                python=settings.docparse_python or sys.executable,
                mode=settings.docparse_mode,
                timeout_s=settings.docparse_timeout_s,
            )
        )
    )
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


def _print_issues(issues) -> None:
    for issue in issues:
        location = f" ({issue.source})" if issue.source else ""
        print(f"[{issue.code}] {issue.message}{location}", file=sys.stderr)


def cmd_ingest(args: argparse.Namespace) -> int:
    settings = Settings.load(args.config)
    runtime = build_runtime(settings)
    if runtime.warning:
        print(f"[WARN] {runtime.warning}", file=sys.stderr)
    options = {"mode": args.mode} if args.mode else {}
    report = runtime.ingest.ingest(args.source, args.path, options)

    if args.json:
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    else:
        print(f"connector: {report.connector}")
        print(f"index: {runtime.backend}")
        print(f"documents: {len(report.documents)}")
        print(f"chunks: {report.chunks}")
        print(f"issues: {len(report.issues)}")
        for document in report.documents:
            print(
                f"  + {document['source']}  {document['chunks']} chunks"
                f"  doc_id={document['doc_id']}"
            )
        _print_issues(report.issues)

    return 0 if report.documents else 2


def cmd_search(args: argparse.Namespace) -> int:
    settings = Settings.load(args.config)
    runtime = build_runtime(settings)
    if runtime.warning:
        print(f"[WARN] {runtime.warning}", file=sys.stderr)
    hits = runtime.search.search(args.query, top_k=args.top_k, doc_ids=args.doc_id)

    if args.json:
        print(json.dumps(hits, ensure_ascii=False, indent=2))
        return 0
    if not hits:
        print("no hits")
        return 0

    for rank, hit in enumerate(hits, start=1):
        section = f"  §{hit['section_path']}" if hit.get("section_path") else ""
        print(
            f"[{rank}] {hit['score']:.4f}  {hit['source']}"
            f" p{hit.get('page', 0)}{section}"
        )
        snippet = (hit.get("text") or "").replace("\n", " ")
        print(f"    {snippet[:120]}")
    return 0


def cmd_docs(args: argparse.Namespace) -> int:
    settings = Settings.load(args.config)
    runtime = build_runtime(settings)
    if runtime.warning:
        print(f"[WARN] {runtime.warning}", file=sys.stderr)
    docs = runtime.index.list_docs()
    if args.json:
        print(json.dumps(docs, ensure_ascii=False, indent=2))
        return 0
    if not docs:
        print("no documents")
        return 0
    print(f"{'source':40} {'chunks':>6} {'pages':>6}  doc_id")
    for doc in docs:
        print(
            f"{doc['source']:40} {doc['chunks']:6} {doc['pages']:6}  {doc['doc_id']}"
        )
    return 0


def cmd_connectors_list(args: argparse.Namespace) -> int:
    settings = Settings.load(args.config)
    registry = build_registry(settings)
    enabled = set(settings.enabled_connectors)
    rows = []
    for name in registry.builtin_names():
        rows.append({"name": name, "kind": "builtin", "enabled": name in enabled})
    for info in registry.plugin_infos():
        rows.append(
            {
                "name": info.name,
                "kind": "plugin",
                "enabled": info.enabled,
                "version": info.version,
                "description": info.description,
                "path": str(info.path),
            }
        )

    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    for row in rows:
        mark = "x" if row["enabled"] else " "
        suffix = f"  {row.get('description', '')}" if row.get("description") else ""
        print(f"[{mark}] {row['name']:24} {row['kind']:8}{suffix}")
    return 0


def cmd_connectors_toggle(args: argparse.Namespace) -> int:
    settings = Settings.load(args.config)
    registry = build_registry(settings)
    known = set(registry.all_names())
    if args.name not in known:
        print(
            f"[CONNECTOR_NOT_FOUND] 未知 Connector: {args.name}"
            f"（可用: {', '.join(sorted(known)) or '无'}）",
            file=sys.stderr,
        )
        return 2

    enabled = set(settings.enabled_connectors)
    if args.action == "enable":
        enabled.add(args.name)
    else:
        enabled.discard(args.name)
    settings.enabled_connectors = sorted(enabled)
    path = settings.save(args.config)
    print(f"{args.action}d {args.name}; config -> {path}")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    settings = Settings.load(args.config)
    runtime = build_runtime(settings)
    if runtime.warning:
        print(f"[WARN] {runtime.warning}", file=sys.stderr)
    data = {**runtime.index.stats(), "backend": runtime.backend}
    if runtime.backend == "keyword":
        data["index_file"] = str(settings.index_file)
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        for key, value in data.items():
            print(f"{key}: {value}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-cli",
        description="插件化多模态文档接入与检索（RAG Multidocs）",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="导入 TXT/PDF/Markdown 等数据源")
    ingest.add_argument("--source", required=True, help="Connector 名称，如 txt/pdf/connector.markdown")
    ingest.add_argument("--path", required=True, help="文件或目录")
    ingest.add_argument("--mode", choices=["auto", "page", "region"], default=None, help="docparse 解析模式")
    ingest.add_argument("--json", action="store_true")
    add_config(ingest)
    ingest.set_defaults(func=cmd_ingest)

    search = sub.add_parser("search", help="关键词检索")
    search.add_argument("query")
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--doc-id", action="append", default=None, help="限定文档，可重复")
    search.add_argument("--json", action="store_true")
    add_config(search)
    search.set_defaults(func=cmd_search)

    docs = sub.add_parser("docs", help="文档列表")
    docs_sub = docs.add_subparsers(dest="docs_command", required=True)
    docs_list = docs_sub.add_parser("list")
    docs_list.add_argument("--json", action="store_true")
    add_config(docs_list)
    docs_list.set_defaults(func=cmd_docs)

    connectors = sub.add_parser("connectors", help="Connector 列表与启停")
    connectors_sub = connectors.add_subparsers(dest="connectors_command", required=True)
    connectors_list = connectors_sub.add_parser("list")
    connectors_list.add_argument("--json", action="store_true")
    add_config(connectors_list)
    connectors_list.set_defaults(func=cmd_connectors_list)

    for action in ("enable", "disable"):
        toggle = connectors_sub.add_parser(action)
        toggle.add_argument("name")
        add_config(toggle)
        toggle.set_defaults(func=cmd_connectors_toggle, action=action)

    stats = sub.add_parser("stats", help="索引统计")
    stats.add_argument("--json", action="store_true")
    add_config(stats)
    stats.set_defaults(func=cmd_stats)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ConnectorError as exc:
        print(f"[{exc.code}] {exc.message}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"[FILE_NOT_FOUND] {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"[CONFIG_ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
