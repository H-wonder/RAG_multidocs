"""Command line interface for RAG Multidocs."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import shlex
import sys
from dataclasses import asdict, dataclass

from ..config import Settings
from ..connectors.txt import TxtConnector
from ..contracts import ConnectorError
from ..providers.discovery import DiscoveryError, discover_models
from ..credentials import (
    apply_keys,
    credentials_path,
    load_keys,
    mask,
    permission_warning,
    set_key,
    source_of,
    unset_key,
)
from ..index import KeywordIndex
from ..ingestion import IngestService
from ..qa.llm import answer_with_evidence, build_chat_model
from ..providers import (
    chat_provider,
    embedding_provider,
    env_var_for_provider,
    key_available,
    list_chat_providers,
    list_embedding_providers,
    providers_for_env,
)
from ..connectors.registry import ConnectorRegistry
from ..retrieval import SearchService
from .wizard import run_setup
from .doctor import cmd_doctor
from .model_test import cmd_models_test
from .config_cmd import cmd_config_profiles, cmd_config_show, cmd_config_validate


def add_config(parser: argparse.ArgumentParser) -> None:
    # SUPPRESS 让子命令未显式传 --config 时保留顶层解析出的值
    parser.add_argument(
        "--config",
        default=argparse.SUPPRESS,
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


def _print_hits(hits) -> None:
    if not hits:
        print("no hits")
        return
    for rank, hit in enumerate(hits, start=1):
        section = f"  §{hit['section_path']}" if hit.get("section_path") else ""
        print(
            f"[{rank}] {hit['score']:.4f}  {hit['source']}"
            f" p{hit.get('page', 0)}{section}"
        )
        snippet = (hit.get("text") or "").replace("\n", " ")
        print(f"    {snippet[:120]}")


def cmd_ask(args: argparse.Namespace) -> int:
    settings = Settings.load(args.config)
    if args.provider:
        settings.chat_provider = args.provider
    if args.model:
        settings.chat_model = args.model
    if args.base_url:
        settings.chat_base_url = args.base_url
    top_k = args.top_k or settings.rag_top_k

    runtime = build_runtime(settings)
    if runtime.warning:
        print(f"[WARN] {runtime.warning}", file=sys.stderr)
    hits = runtime.search.search(args.query, top_k=top_k, doc_ids=args.doc_id)

    if args.no_generate:
        if args.json:
            print(json.dumps(hits, ensure_ascii=False, indent=2))
        else:
            _print_hits(hits)
        return 0

    if not hits:
        if args.json:
            print(json.dumps(
                {"question": args.query, "answer": "没有检索到相关依据。",
                 "citations": [], "hits": []},
                ensure_ascii=False, indent=2,
            ))
        else:
            print("没有检索到相关依据。")
        return 0

    if not settings.chat_provider:
        print(
            "[CONFIG_ERROR] 未配置 chat provider。运行 `rag-cli setup` 选择模型，"
            "或用 --provider 指定；API Key 可用 `rag-cli keys set --provider "
            "deepseek` 保存到本项目，也支持 export DEEPSEEK_API_KEY=... 临时注入。",
            file=sys.stderr,
        )
        _print_hits(hits)
        return 2

    stream = bool(getattr(args, "stream", False)) and not args.json
    on_token = None
    if stream:
        print("答案：")
        sys.stdout.flush()
        on_token = lambda token: print(token, end="", flush=True)  # noqa: E731

    try:
        chat = build_chat_model(
            settings.chat_provider,
            model=settings.chat_model,
            base_url=settings.chat_base_url,
            temperature=settings.chat_temperature,
            max_tokens=settings.chat_max_tokens,
        )
        result = answer_with_evidence(args.query, hits, chat, on_token=on_token)
    except ValueError as exc:
        print(f"[CONFIG_ERROR] {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"\n[LLM_ERROR] {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(
            {
                "question": args.query,
                "provider": settings.chat_provider,
                "model": chat.model,
                "index_backend": runtime.backend,
                **result,
                "hits": hits,
            },
            ensure_ascii=False,
            indent=2,
        ))
        return 0

    if stream:
        print()
    else:
        print("答案：")
        print(result["answer"])
    print("\n来源：")
    for citation in result["citations"]:
        section = f" §{citation['section_path']}" if citation["section_path"] else ""
        print(
            f"[{citation['index']}] {citation['source']}"
            f" p{citation['page']}{section}"
        )
    return 0


def cmd_models_list(args: argparse.Namespace) -> int:
    settings = Settings.load(args.config)
    rows = []
    for item in list_embedding_providers():
        provider = item["name"]
        rows.append(
            {
                "category": "embedding",
                "provider": provider,
                "selected": provider == settings.embed_backend,
                "available": key_available(embedding_provider(provider)),
                **item,
            }
        )
    for item in list_chat_providers():
        provider = item["name"]
        rows.append(
            {
                "category": "chat",
                "provider": provider,
                "selected": provider == settings.chat_provider,
                "available": key_available(chat_provider(provider)),
                **item,
            }
        )

    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    for row in rows:
        mark = "*" if row["selected"] else " "
        model = row.get("default_model", "")
        dim = f" dim={row['default_dim']}" if row["category"] == "embedding" else ""
        key = _key_label(row.get("api_key_env", ""))
        print(
            f"[{mark}] {row['category']:9} {row['provider']:10} {model}{dim}"
            f"  key={key}  available={row['available']}"
        )
    return 0


def _key_label(env_name: str) -> str:
    if not env_name:
        return "无需 Key"
    source = source_of(env_name)
    value = mask(os.environ.get(env_name, ""))
    if not source:
        return f"{env_name}[未设置]"
    return f"{env_name}[{source}] {value}"


def cmd_models_discover(args: argparse.Namespace) -> int:
    try:
        models = discover_models(
            args.provider,
            kind=args.kind,
            timeout=args.timeout,
        )
    except (ValueError, DiscoveryError) as exc:
        print(f"[MODEL_DISCOVERY_ERROR] {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(
            {
                "provider": args.provider,
                "kind": args.kind or "",
                "models": models,
            },
            ensure_ascii=False,
            indent=2,
        ))
        return 0
    if not models:
        print("no models")
        return 0
    for model in models:
        print(model)
    return 0


def cmd_models_use(args: argparse.Namespace) -> int:
    settings = Settings.load(args.config)
    if args.kind == "chat" and args.provider.lower() in ("none", "off", "clear"):
        settings.chat_provider = ""
        settings.chat_model = ""
        settings.chat_base_url = ""
        path = settings.save(args.config)
        print("chat -> 已关闭（仅检索，不调用 LLM）")
        print(f"config -> {path}")
        return 0
    try:
        if args.kind == "embedding":
            from ..providers import embedding_provider

            preset = embedding_provider(args.provider)
            settings.embed_backend = args.provider
            # 切换 provider 时重置为 provider 默认，避免沿用旧模型的维度
            settings.embed_model = args.model or ""
            settings.embed_dim = args.dim if args.dim is not None else 0
            settings.embed_base_url = args.base_url or ""
            print(
                f"embedding -> {settings.embed_backend}"
                f" model={settings.embed_model or preset.default_model}"
                f" dim={settings.embed_dim or preset.default_dim}"
            )
            print("注意：换嵌入模型/维度后需要新的 Milvus 集合，旧向量不能复用。")
        else:
            preset = chat_provider(args.provider)
            settings.chat_provider = args.provider
            settings.chat_model = args.model or ""
            settings.chat_base_url = args.base_url or ""
            print(
                f"chat -> {settings.chat_provider}"
                f" model={settings.chat_model or preset.default_model}"
            )
    except ValueError as exc:
        print(f"[CONFIG_ERROR] {exc}", file=sys.stderr)
        return 2

    path = settings.save(args.config)
    preset_key = preset.api_key_env
    if preset_key:
        print(
            f"填入 API Key：rag-cli keys set --provider {preset.name}"
            f"（或 export {preset_key}=...，环境变量优先）"
        )
    else:
        print("该 provider 不需要 API Key。")
    print(f"config -> {path}")
    return 0


def _key_target(args: argparse.Namespace) -> str:
    if getattr(args, "env", None):
        return args.env.strip().upper()
    if getattr(args, "provider", None):
        return env_var_for_provider(args.provider)
    raise ValueError("请用 --provider 或 --env 指定要操作的 Key")


def cmd_keys_list(args: argparse.Namespace) -> int:
    settings = Settings.load(args.config)
    path = credentials_path(settings.credentials_file)
    stored = load_keys(settings.credentials_file)
    env_names = sorted(
        set(stored)
        | {
            item["api_key_env"]
            for item in (*list_chat_providers(), *list_embedding_providers())
            if item.get("api_key_env")
        }
    )
    rows = [
        {
            "env": name,
            "source": source_of(name) or "",
            "masked": mask(os.environ.get(name, "")),
            "providers": providers_for_env(name),
        }
        for name in env_names
    ]
    if args.json:
        print(json.dumps(
            {"credentials_file": str(path), "keys": rows},
            ensure_ascii=False,
            indent=2,
        ))
        return 0

    print(f"credentials: {path}")
    warning = permission_warning(settings.credentials_file)
    if warning:
        print(f"[WARN] {warning}", file=sys.stderr)
    if not rows:
        print("未保存任何 API Key。")
        return 0
    for row in rows:
        providers = ", ".join(row["providers"]) or "-"
        print(
            f"{row['env']:22} {row['source'] or '-':6} {row['masked'] or '-':16}"
            f" ({providers})"
        )
    if not any(row["source"] for row in rows):
        print("提示：rag-cli keys set --provider deepseek 可保存 Key 到本项目。")
    return 0


def cmd_keys_set(args: argparse.Namespace) -> int:
    settings = Settings.load(args.config)
    try:
        env_name = _key_target(args)
    except ValueError as exc:
        print(f"[CONFIG_ERROR] {exc}", file=sys.stderr)
        return 2
    value = args.value
    if value is None:
        if not sys.stdin.isatty():
            print(
                "[CONFIG_ERROR] 非交互环境请用 --value 传入 Key",
                file=sys.stderr,
            )
            return 2
        try:
            value = getpass.getpass(f"{env_name}（输入不回显）: ")
        except (EOFError, KeyboardInterrupt):
            print("\n未输入，已取消。", file=sys.stderr)
            return 2
    try:
        path = set_key(env_name, value, settings.credentials_file)
    except ValueError as exc:
        print(f"[CONFIG_ERROR] {exc}", file=sys.stderr)
        return 2
    print(f"已保存 {env_name}={mask(value.strip())} -> {path}（权限 600）")
    print(
        "提示：环境变量优先于本地凭证；如需当前 shell 也生效，"
        "可运行 eval \"$(rag-cli keys export)\"。"
    )
    return 0


def cmd_keys_unset(args: argparse.Namespace) -> int:
    settings = Settings.load(args.config)
    try:
        env_name = _key_target(args)
    except ValueError as exc:
        print(f"[CONFIG_ERROR] {exc}", file=sys.stderr)
        return 2
    if unset_key(env_name, settings.credentials_file):
        print(f"已删除 {env_name}")
    else:
        print(f"{env_name} 未保存，无需删除")
    if source_of(env_name) == "env":
        print(
            f"注意：当前 shell 仍设置了 {env_name}，环境变量优先于本地凭证。"
        )
    return 0


def cmd_keys_export(args: argparse.Namespace) -> int:
    settings = Settings.load(args.config)
    keys = load_keys(settings.credentials_file)
    if not keys:
        print("# 没有保存的 API Key", file=sys.stderr)
        return 0
    print(
        "# 警告：以下内容是明文 API Key，只应通过 "
        "eval \"$(rag-cli keys export)\" 使用",
        file=sys.stderr,
    )
    for name in sorted(keys):
        print(f"export {name}={shlex.quote(keys[name])}")
    return 0


def cmd_keys_path(args: argparse.Namespace) -> int:
    settings = Settings.load(args.config)
    print(credentials_path(settings.credentials_file))
    return 0


def cmd_setup(args: argparse.Namespace) -> int:
    settings = Settings.load(args.config)
    return run_setup(settings, args.config)


def cmd_repl(args: argparse.Namespace) -> int:
    from .repl import run_repl

    return run_repl(args.config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-cli",
        description=(
            "插件化多模态文档接入与检索（RAG Multidocs）。"
            "直接运行 rag-cli 进入交互模式。"
        ),
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="配置文件路径（默认 config.json）",
    )
    parser.add_argument(
        "-p", "--print", dest="print_query", metavar="QUERY",
        help="一次性提问并输出答案（等同 ask，可配合 --provider/--model）",
    )
    parser.add_argument("--provider", help="一次性提问的 chat provider")
    parser.add_argument("--model", help="一次性提问的模型名")
    parser.add_argument(
        "--no-stream", action="store_true", help="一次性提问时关闭流式输出"
    )
    sub = parser.add_subparsers(dest="command")

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

    ask = sub.add_parser("ask", help="检索增强问答（检索 + LLM 生成 + 引用）")
    ask.add_argument("query")
    ask.add_argument("--top-k", type=int, default=None)
    ask.add_argument("--doc-id", action="append", default=None, help="限定文档，可重复")
    ask.add_argument("--provider", help="chat provider，如 deepseek/dashscope/openai/ollama")
    ask.add_argument("--model", help="chat 模型名")
    ask.add_argument("--base-url", dest="base_url", help="覆盖 chat base URL")
    ask.add_argument("--no-generate", action="store_true", help="只返回检索证据，不调用 LLM")
    ask.add_argument("--stream", action="store_true", help="流式打印答案")
    ask.add_argument("--json", action="store_true")
    add_config(ask)
    ask.set_defaults(func=cmd_ask)

    models = sub.add_parser("models", help="模型 provider 列表与选择")
    models_sub = models.add_subparsers(dest="models_command", required=True)
    models_list = models_sub.add_parser("list", help="列出 embedding/chat provider")
    models_list.add_argument("--json", action="store_true")
    add_config(models_list)
    models_list.set_defaults(func=cmd_models_list)

    models_use = models_sub.add_parser("use", help="选择 provider/模型并写入 config")
    models_use.add_argument("--kind", choices=["embedding", "chat"], required=True)
    models_use.add_argument("--provider", required=True)
    models_use.add_argument("--model", default="")
    models_use.add_argument("--dim", type=int, default=None, help="embedding 维度（覆盖默认）")
    models_use.add_argument("--base-url", dest="base_url", default="", help="覆盖 base URL")
    add_config(models_use)
    models_use.set_defaults(func=cmd_models_use)

    models_discover = models_sub.add_parser(
        "discover", help="从 provider 动态查询可用模型"
    )
    models_discover.add_argument("--provider", required=True)
    models_discover.add_argument(
        "--kind", choices=["embedding", "chat"], default=None,
        help="同名 provider 区分 chat/embedding；可省略",
    )
    models_discover.add_argument("--timeout", type=float, default=10.0)
    models_discover.add_argument("--json", action="store_true")
    add_config(models_discover)
    models_discover.set_defaults(func=cmd_models_discover)

    models_test = models_sub.add_parser(
        "test", help="真实调用一次 provider，验证配置可用"
    )
    models_test.add_argument(
        "--kind", choices=["embedding", "chat"], default=None,
        help="测哪种；不传则测当前配置的 embedding（+ chat 若已配置）",
    )
    models_test.add_argument("--provider", help="覆盖 provider")
    models_test.add_argument("--model", help="覆盖模型名")
    models_test.add_argument("--base-url", dest="base_url", help="覆盖 base URL")
    models_test.add_argument("--dim", type=int, default=None, help="embedding 维度覆盖")
    models_test.add_argument("--json", action="store_true")
    add_config(models_test)
    models_test.set_defaults(func=cmd_models_test)

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

    keys = sub.add_parser("keys", help="本地 API Key 管理（0600，gitignored）")
    keys_sub = keys.add_subparsers(dest="keys_command", required=True)

    keys_list = keys_sub.add_parser("list", help="查看已保存的 Key（脱敏）")
    keys_list.add_argument("--json", action="store_true")
    add_config(keys_list)
    keys_list.set_defaults(func=cmd_keys_list)

    key_help = {
        "set": "保存某个 provider 的 API Key",
        "unset": "删除已保存的 API Key",
    }
    for action in ("set", "unset"):
        key_action = keys_sub.add_parser(action, help=key_help[action])
        group = key_action.add_mutually_exclusive_group(required=True)
        group.add_argument("--provider", help="provider 名，如 deepseek/dashscope/openai")
        group.add_argument("--env", help="直接指定环境变量名，如 DEEPSEEK_API_KEY")
        if action == "set":
            key_action.add_argument(
                "--value",
                help="（不推荐，会进入 shell 历史）直接传 Key；省略则隐藏输入",
            )
        add_config(key_action)
        key_action.set_defaults(
            func=cmd_keys_set if action == "set" else cmd_keys_unset
        )

    keys_export = keys_sub.add_parser(
        "export", help="输出 export 语句，供 eval 到当前 shell（含明文）"
    )
    add_config(keys_export)
    keys_export.set_defaults(func=cmd_keys_export)

    keys_path = keys_sub.add_parser("path", help="显示凭证文件路径")
    add_config(keys_path)
    keys_path.set_defaults(func=cmd_keys_path)

    doctor = sub.add_parser("doctor", help="诊断配置、依赖服务与索引")
    doctor.add_argument(
        "--full", action="store_true",
        help="额外检查 Milvus 集合维度（需连接 Milvus）",
    )
    doctor.add_argument("--json", action="store_true")
    add_config(doctor)
    doctor.set_defaults(func=cmd_doctor)

    config = sub.add_parser("config", help="查看/校验配置与预设方案")
    config_sub = config.add_subparsers(dest="config_command", required=True)

    config_show = config_sub.add_parser("show", help="查看当前配置")
    config_show.add_argument("--json", action="store_true")
    add_config(config_show)
    config_show.set_defaults(func=cmd_config_show)

    config_validate = config_sub.add_parser("validate", help="校验配置合法性")
    config_validate.add_argument("--json", action="store_true")
    add_config(config_validate)
    config_validate.set_defaults(func=cmd_config_validate)

    profiles = config_sub.add_parser(
        "profiles", help="预设方案（offline/local/cloud）"
    )
    profiles_sub = profiles.add_subparsers(dest="profiles_command", required=True)

    profiles_list = profiles_sub.add_parser("list", help="列出预设方案")
    profiles_list.add_argument("--json", action="store_true")
    add_config(profiles_list)
    profiles_list.set_defaults(func=cmd_config_profiles)

    profiles_apply = profiles_sub.add_parser("apply", help="应用预设方案")
    profiles_apply.add_argument("name", help="offline / local / cloud")
    add_config(profiles_apply)
    profiles_apply.set_defaults(func=cmd_config_profiles)

    setup = sub.add_parser(
        "setup", help="交互式配置 chat + embedding 模型并填入 API Key"
    )
    add_config(setup)
    setup.set_defaults(func=cmd_setup)

    repl = sub.add_parser("repl", help="进入交互模式（等同直接运行 rag-cli）")
    add_config(repl)
    repl.set_defaults(func=cmd_repl)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config_path = getattr(args, "config", None) or "config.json"

    try:
        apply_keys(Settings.load(config_path).credentials_file)
    except ValueError as exc:
        print(f"[WARN] {exc}", file=sys.stderr)

    if getattr(args, "print_query", None) is not None:
        ask_args = argparse.Namespace(
            config=config_path,
            query=args.print_query,
            top_k=None,
            doc_id=None,
            provider=args.provider,
            model=args.model,
            base_url=None,
            no_generate=False,
            json=False,
            stream=not args.no_stream and sys.stdout.isatty(),
        )
        return cmd_ask(ask_args)

    if args.command is None:
        if not sys.stdin.isatty():
            parser.print_help(sys.stderr)
            return 2
        from .repl import run_repl

        return run_repl(config_path)

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
