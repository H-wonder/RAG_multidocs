"""rag-cli 入口：argparse 定义与命令分发。"""

from __future__ import annotations

import argparse
import sys

from ..config import Settings
from ..contracts import ConnectorError
from ..credentials import apply_keys
from .commands import (
    cmd_ask,
    cmd_connectors_list,
    cmd_connectors_toggle,
    cmd_docs,
    cmd_ingest,
    cmd_keys_export,
    cmd_keys_list,
    cmd_keys_path,
    cmd_keys_set,
    cmd_keys_unset,
    cmd_models_discover,
    cmd_models_list,
    cmd_models_use,
    cmd_search,
    cmd_stats,
)
from .config_cmd import cmd_config_profiles, cmd_config_show, cmd_config_validate
from .doctor import cmd_doctor
from .model_test import cmd_models_test
from .wizard import run_setup


def add_config(parser: argparse.ArgumentParser) -> None:
    # SUPPRESS 让子命令未显式传 --config 时保留顶层解析出的值
    parser.add_argument(
        "--config",
        default=argparse.SUPPRESS,
        help="配置文件路径（默认 config.json；不存在则使用默认值）",
    )


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
