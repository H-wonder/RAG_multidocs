"""Interactive REPL for RAG Multidocs (``rag-cli`` with no subcommand)."""

from __future__ import annotations

import argparse
import getpass
import os
import shlex
import sys
from pathlib import Path

from ..config import Settings
from ..credentials import (
    credentials_path,
    mask,
    permission_warning,
    set_key,
    source_of,
    unset_key,
)
from ..providers.discovery import DiscoveryError, discover_models
from ..retrieval.profiles import recommended_policies
from .asking import print_answer, print_trace, trace_summary
from ..providers import (
    chat_provider,
    embedding_provider,
    env_var_for_provider,
    list_chat_providers,
    list_embedding_providers,
    providers_for_env,
)
from .wizard import run_setup

ALIASES = {"?": "help", "q": "quit"}


def _prepare_terminal() -> None:
    """Some terminals advertise TERM=dumb, which disables readline editing."""
    if sys.stdout.isatty() and os.environ.get("TERM", "") in ("", "dumb"):
        os.environ["TERM"] = "xterm-256color"


_prepare_terminal()
try:  # readline is stdlib on Linux/macOS; Windows users just lose history
    import readline
except ImportError:  # pragma: no cover - platform dependent
    readline = None


HELP = """\
直接输入问题 -> 检索 + 生成答案（配置了 chat 时）
命令也可以省略开头的 /（例如 setup 等同 /setup）
/help                         显示帮助
/search <问题>                只检索，不调用 LLM
/ask <问题>                   检索并生成经证据校验的答案
/policy [名称]                查看或切换 stable/lightweight/quality/experimental
/trace                        显示上次问答的诊断摘要
/ingest auto <路径> [...]      自动识别文件，目录递归导入，含空格请加引号
/ingest <source> <路径> [...]  指定 Connector 导入
/docs                         文档列表
/stats                        索引统计
/connectors                   连接器列表
/models                       模型 provider 列表
/model                        打开 chat/embedding 交互选择菜单
/model <chat|embedding> <provider> [模型名] [--dim N]
                              直接设置模型并写入 config.json；
                              chat 用 none 关闭生成式问答
/keys                         查看已保存的 Key（脱敏）
/keys set <provider>          输入并保存 API Key（不回显）
/keys unset <provider>        删除已保存的 Key
/setup                        两步配置 chat/embedding 模型 + 填 Key
/config                       查看当前配置（不显示 Key）
/clear                        清屏
/exit, /quit, /q              退出（Ctrl-D 也可以）
"""


def _color(text: str, code: str) -> str:
    if (
        not sys.stdout.isatty()
        or os.environ.get("NO_COLOR")
        or os.environ.get("TERM") == "dumb"
    ):
        return text
    return f"\033[{code}m{text}\033[0m"


def _input_prompt(text: str, code: str) -> str:
    """Colored prompt with readline non-printing markers.

    readline 需要 ``\\001``/``\\002`` 包住 ANSI 转义，否则会把颜色码算进
    光标列数；在长命令、粘贴和退格时会出现错位或粘贴内容异常。
    """
    prompt = _color(text, code)
    if readline is not None and prompt != text:
        return f"\001{prompt}\002"
    return prompt


def _print_hits(hits: list[dict], snippet_chars: int = 120) -> None:
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
        print(f"    {snippet[:snippet_chars]}")


class Session:
    """Holds runtime state for one interactive session."""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.settings = Settings.load(config_path)
        self._runtime = None
        self._running = True
        self.last_trace = None

    # -- runtime ---------------------------------------------------------
    @property
    def runtime(self):
        if self._runtime is None:
            if sys.stderr.isatty():
                print(
                    "正在加载索引与嵌入模型（首次可能需要几秒）...",
                    file=sys.stderr,
                )
            from .app import build_runtime

            runtime = build_runtime(self.settings)
            if runtime.warning:
                print(f"[WARN] {runtime.warning}", file=sys.stderr)
            self._runtime = runtime
        return self._runtime

    def close(self) -> None:
        if self._runtime is not None:
            runtime, self._runtime = self._runtime, None
            runtime.close()

    def reload_settings(self, *, rebuild: bool) -> None:
        updated = Settings.load(self.config_path)
        keys = (
            "data_dir", "connectors_dir", "enabled_connectors", "docparse_repo",
            "docparse_python", "docparse_mode", "docparse_timeout_s", "index_backend",
            "fallback_to_keyword", "milvus_uri", "milvus_db", "milvus_collection",
            "milvus_dense_weight", "milvus_sparse_weight", "embed_backend",
            "embed_model", "embed_dim", "embed_base_url",
        )
        if rebuild and any(getattr(self.settings, k) != getattr(updated, k) for k in keys):
            self.close()
        elif self._runtime is not None:
            self._runtime.invalidate_chat()
        self.settings = updated

    # -- dispatch --------------------------------------------------------
    def _command_names(self) -> set[str]:
        return {name[4:] for name in dir(self) if name.startswith("cmd_")}

    def dispatch(self, line: str) -> None:
        if line.startswith("/"):
            self._run_command(line[1:])
            return

        first = line.split(None, 1)[0]
        if first in self._command_names() or first in ALIASES:
            name = ALIASES.get(first, first)
            if name not in ("exit", "quit", "help"):
                print(
                    f"提示：交互命令可以写成 /{name}；本次已按命令执行。",
                    file=sys.stderr,
                )
            rest = line[len(first):]
            self._run_command(f"{name}{rest}")
            return
        self.do_ask(line)

    def _run_command(self, text: str) -> None:
        try:
            parts = shlex.split(text)
        except ValueError as exc:
            print(f"[ERROR] 命令解析失败: {exc}", file=sys.stderr)
            return
        if not parts:
            return
        name = ALIASES.get(parts[0], parts[0])
        handler = getattr(self, f"cmd_{name}", None)
        if handler is None:
            print(f"未知命令: /{parts[0]}（输入 /help 查看）", file=sys.stderr)
            return
        handler(parts[1:])

    # -- commands --------------------------------------------------------
    def cmd_help(self, args: list[str]) -> None:
        print(HELP)

    def cmd_exit(self, args: list[str]) -> None:
        self._running = False

    def cmd_quit(self, args: list[str]) -> None:
        self._running = False

    def cmd_clear(self, args: list[str]) -> None:
        if sys.stdout.isatty():
            print("\033[2J\033[H", end="")

    def cmd_search(self, args: list[str]) -> None:
        query = " ".join(args).strip()
        if not query:
            print("用法: /search <问题>", file=sys.stderr)
            return
        hits = self.runtime.search.search(query, top_k=self.settings.rag_top_k)
        if not hits:
            print("没有检索到相关依据。")
            return
        _print_hits(hits)

    def cmd_ask(self, args: list[str]) -> None:
        query = " ".join(args).strip()
        if not query:
            print("用法: /ask <问题>", file=sys.stderr)
            return
        self.do_ask(query)

    def cmd_policy(self, args: list[str]) -> None:
        choices = recommended_policies()
        if not args:
            print(f"当前策略 {self.settings.rag_policy}。可选 {', '.join(choices)}")
        elif len(args) == 1 and args[0].lower() in choices:
            self.settings.rag_policy = args[0].lower()
            print(f"当前会话策略 {self.settings.rag_policy}")
        else:
            print(f"用法 /policy [{'|'.join(choices)}]", file=sys.stderr)

    def cmd_trace(self, args: list[str]) -> None:
        if args:
            print("用法 /trace", file=sys.stderr)
        elif self.last_trace is None:
            print("当前会话尚无问答记录。")
        else:
            print_trace(self.last_trace)

    def cmd_ingest(self, args: list[str]) -> None:
        from .app import print_ingest_report

        parser = argparse.ArgumentParser(prog="/ingest", description="auto 自动匹配 Connector，或指定名称")
        parser.add_argument("source")
        parser.add_argument("paths", nargs="+", metavar="PATH", help="文件或目录，支持多个路径")
        parser.add_argument("--mode", choices=["auto", "page", "region"])
        try:
            values = parser.parse_intermixed_args(args)
        except SystemExit:
            return
        options = {"mode": values.mode} if values.mode else {}
        report = self.runtime.ingest.ingest(values.source, values.paths, options)
        print_ingest_report(report, self.runtime.backend)

    def cmd_docs(self, args: list[str]) -> None:
        from .app import print_documents

        print_documents(self.runtime.index.list_docs())

    def cmd_stats(self, args: list[str]) -> None:
        data = {**self.runtime.index.stats(), "backend": self.runtime.backend}
        for key, value in data.items():
            print(f"{key}: {value}")

    def cmd_connectors(self, args: list[str]) -> None:
        settings = self.settings
        enabled = set(settings.enabled_connectors)
        for name in self.runtime.registry.builtin_names():
            mark = "x" if name in enabled else " "
            print(f"[{mark}] {name:24} builtin")
        for info in self.runtime.registry.plugin_infos():
            mark = "x" if info.enabled else " "
            suffix = f"  {info.description}" if info.description else ""
            print(f"[{mark}] {info.name:24} plugin{suffix}")

    def cmd_models(self, args: list[str]) -> None:
        settings = self.settings
        for item in list_embedding_providers():
            name = item["name"]
            mark = "*" if name == settings.embed_backend else " "
            key = self._key_summary(item.get("api_key_env", ""))
            dim = f" dim={item['default_dim']}"
            print(
                f"[{mark}] embedding {name:10} {item['default_model']}{dim}"
                f"  key={key}"
            )
        for item in list_chat_providers():
            name = item["name"]
            mark = "*" if name == settings.chat_provider else " "
            key = self._key_summary(item.get("api_key_env", ""))
            print(
                f"[{mark}] chat      {name:10} {item['default_model']}"
                f"  key={key}"
            )

    def _ask_line(self, prompt: str) -> str | None:
        try:
            return input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None

    def _pick_model_kind(self) -> str | None:
        print("选择模型类型：")
        print("  [1] chat       生成答案")
        print("  [2] embedding  向量化入库/检索")
        choice = self._ask_line("编号 [1]: ")
        if choice is None:
            return None
        if choice in ("", "1"):
            return "chat"
        if choice == "2":
            return "embedding"
        print("无效编号", file=sys.stderr)
        return None

    def _pick_model_provider(self, kind: str) -> str | None:
        if kind == "chat":
            rows = list_chat_providers()
            current = self.settings.chat_provider
        else:
            rows = list_embedding_providers()
            current = self.settings.embed_backend
        print(f"选择 {kind} provider：")
        if kind == "chat":
            print("  [0] none       关闭生成式问答")
        for index, item in enumerate(rows, start=1):
            mark = "*" if item["name"] == current else " "
            key = self._key_summary(item.get("api_key_env", ""))
            print(
                f"  [{index}]{mark} {item['name']:10} "
                f"{item['default_model']:28} key={key}"
            )
        choice = self._ask_line(
            f"编号 [回车保持 {current or '未选择'}]: "
        )
        if choice is None:
            return None
        if kind == "chat" and choice == "0":
            return "none"
        if not choice:
            if current:
                return current
            return "none" if kind == "chat" else "local"
        if choice.isdigit():
            index = int(choice)
            if 1 <= index <= len(rows):
                return rows[index - 1]["name"]
            print("编号超出范围", file=sys.stderr)
            return None
        try:
            if kind == "chat":
                chat_provider(choice)
            else:
                embedding_provider(choice)
        except ValueError as exc:
            print(f"[CONFIG_ERROR] {exc}", file=sys.stderr)
            return None
        return choice

    def _pick_model_name(
        self, kind: str, provider: str
    ) -> tuple[str, int | None] | None:
        preset = (
            chat_provider(provider)
            if kind == "chat"
            else embedding_provider(provider)
        )
        try:
            models = discover_models(provider, kind=kind)
        except (DiscoveryError, ValueError) as exc:
            print(f"[WARN] 无法动态获取模型列表: {exc}", file=sys.stderr)
            models = []
        if not models:
            models = [preset.default_model]

        print("可用模型：")
        for index, model in enumerate(models, start=1):
            suffix = "  (默认)" if model == preset.default_model else ""
            print(f"  [{index}] {model}{suffix}")
        print("  [c] 手动输入模型名")
        choice = self._ask_line(
            f"编号或模型名 [默认 {preset.default_model}]: "
        )
        if choice is None:
            return None
        if not choice:
            model = ""
        elif choice.lower() == "c":
            model = self._ask_line("模型名: ") or ""
        elif choice.isdigit() and 1 <= int(choice) <= len(models):
            model = models[int(choice) - 1]
        else:
            model = choice
        if model == preset.default_model:
            model = ""

        dim: int | None = None
        if kind == "embedding":
            dim_text = self._ask_line(f"向量维度 [{preset.default_dim}]: ")
            if dim_text is None:
                return None
            if dim_text:
                if not dim_text.isdigit() or int(dim_text) <= 0:
                    print("维度必须是正整数", file=sys.stderr)
                    return None
                dim = int(dim_text)
        return model, dim

    def _model_picker(self) -> None:
        if not sys.stdin.isatty():
            print(
                "用法: /model <chat|embedding> <provider> [模型名] [--dim N]",
                file=sys.stderr,
            )
            return
        kind = self._pick_model_kind()
        if kind is None:
            return
        provider = self._pick_model_provider(kind)
        if provider is None:
            return
        if kind == "chat" and provider == "none":
            self.cmd_model(["chat", "none"])
            return
        picked = self._pick_model_name(kind, provider)
        if picked is None:
            return
        model, dim = picked
        args = [kind, provider]
        if model:
            args.append(model)
        if kind == "embedding" and dim is not None:
            args.extend(["--dim", str(dim)])
        self.cmd_model(args)

    def cmd_model(self, args: list[str]) -> None:
        if not args:
            self._model_picker()
            return
        if len(args) < 2 or args[0] not in ("chat", "embedding"):
            print("用法: /model <chat|embedding> <provider> [模型名] [--dim N]", file=sys.stderr)
            return
        kind, provider = args[0], args[1]
        if kind == "chat" and provider.lower() in ("none", "off", "clear"):
            self.settings.chat_provider = ""
            self.settings.chat_model = ""
            self.settings.chat_base_url = ""
            path = self.settings.save(self.config_path)
            if self._runtime is not None:
                self._runtime.invalidate_chat()
            print(f"chat -> 已关闭（仅检索）; config -> {path}")
            return
        rest = args[2:]
        model = ""
        dim = None
        index = 0
        while index < len(rest):
            item = rest[index]
            if item == "--dim" and index + 1 < len(rest):
                dim = int(rest[index + 1])
                index += 2
                continue
            if item.startswith("--dim="):
                dim = int(item.split("=", 1)[1])
            elif not item.startswith("-"):
                model = item
            index += 1
        try:
            if kind == "chat":
                preset = chat_provider(provider)
                if self.settings.chat_provider != provider:
                    self.settings.chat_base_url = ""
                self.settings.chat_provider = provider
                self.settings.chat_model = model
                print(
                    f"chat -> {provider} model={model or preset.default_model}"
                )
                rebuild = False
            else:
                preset = embedding_provider(provider)
                self.settings.embed_backend = provider
                self.settings.embed_model = model
                self.settings.embed_dim = dim or 0
                print(
                    f"embedding -> {provider}"
                    f" model={model or preset.default_model}"
                    f" dim={dim or preset.default_dim}"
                )
                print("注意：换 embedding 模型/维度后需要新的 Milvus 集合。")
                rebuild = True
        except ValueError as exc:
            print(f"[CONFIG_ERROR] {exc}", file=sys.stderr)
            return
        path = self.settings.save(self.config_path)
        print(f"config -> {path}")
        if rebuild:
            self.close()
        elif self._runtime is not None:
            self._runtime.invalidate_chat()
        if getattr(preset, "api_key_env", ""):
            print(f"如需填 Key: /keys set {provider}")

    def cmd_keys(self, args: list[str]) -> None:
        action = args[0] if args else "list"
        if action == "list":
            self._keys_list()
        elif action == "set":
            self._keys_set(args[1:])
        elif action in ("unset", "delete", "remove"):
            self._keys_unset(args[1:])
        elif action == "path":
            print(credentials_path(self.settings.credentials_file))
        else:
            print("用法: /keys [list|set <provider>|unset <provider>|path]", file=sys.stderr)

    def _keys_list(self) -> None:
        path = credentials_path(self.settings.credentials_file)
        print(f"凭证文件: {path}")
        warning = permission_warning(self.settings.credentials_file)
        if warning:
            print(f"[WARN] {warning}", file=sys.stderr)
        env_names = sorted(
            {
                item["api_key_env"]
                for item in (*list_chat_providers(), *list_embedding_providers())
                if item.get("api_key_env")
            }
        )
        if not env_names:
            print("没有需要 API Key 的 provider。")
            return
        for name in env_names:
            source = source_of(name)
            value = mask(os.environ.get(name, ""))
            providers = ", ".join(providers_for_env(name))
            state = source or "-"
            shown = value or "-"
            print(f"{name:22} {state:5} {shown:16} ({providers})")

    def _keys_set(self, args: list[str]) -> None:
        if not args:
            print("用法: /keys set <provider>", file=sys.stderr)
            return
        try:
            env_name = env_var_for_provider(args[0])
        except ValueError as exc:
            print(f"[CONFIG_ERROR] {exc}", file=sys.stderr)
            return
        try:
            value = getpass.getpass(f"{env_name}（输入不回显）: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not value:
            print("未输入，已取消。", file=sys.stderr)
            return
        path = set_key(env_name, value, self.settings.credentials_file)
        if self._runtime is not None:
            self._runtime.invalidate_chat()
        print(f"已保存 {env_name}={mask(value)} -> {path}")

    def _keys_unset(self, args: list[str]) -> None:
        if not args:
            print("用法: /keys unset <provider>", file=sys.stderr)
            return
        try:
            env_name = env_var_for_provider(args[0])
        except ValueError as exc:
            print(f"[CONFIG_ERROR] {exc}", file=sys.stderr)
            return
        if unset_key(env_name, self.settings.credentials_file):
            if self._runtime is not None:
                self._runtime.invalidate_chat()
            print(f"已删除 {env_name}")
        else:
            print(f"{env_name} 未保存，无需删除")
        if source_of(env_name) == "env":
            print(f"注意：当前进程仍设置了 {env_name}，环境变量优先。")

    def cmd_setup(self, args: list[str]) -> None:
        code = run_setup(self.settings, self.config_path)
        if code == 0:
            self.reload_settings(rebuild=True)

    def cmd_config(self, args: list[str]) -> None:
        settings = self.settings
        print(f"config: {Path(self.config_path)}")
        paths = settings.resolved_connector_settings()
        for name in ("connectors_dir", "docparse_repo", "docparse_python"):
            print(f"{name}: {paths[name] or '未找到'}")
        print(f"index: {settings.index_backend} collection={settings.milvus_collection}")
        print(f"policy: {settings.rag_policy}")
        model = settings.embed_model or embedding_provider(
            settings.embed_backend
        ).default_model
        dim = settings.embed_dim or embedding_provider(settings.embed_backend).default_dim
        print(f"embedding: {settings.embed_backend} model={model} dim={dim}")
        if settings.chat_provider:
            preset = chat_provider(settings.chat_provider)
            print(
                f"chat: {settings.chat_provider}"
                f" model={settings.chat_model or preset.default_model}"
            )
        else:
            print("chat: 未启用（只检索）")
        print(f"credentials: {credentials_path(settings.credentials_file)}")

    # -- RAG ask ---------------------------------------------------------
    def do_ask(self, question: str) -> None:
        result = self.runtime.ask(self.settings, question)
        self.last_trace = trace_summary(result, self.settings.rag_policy)
        print_answer(result, _print_hits)
        if result["status"] == "retrieval_only":
            print("未配置聊天模型。输入 /model 选择模型，或继续仅检索。")

    def _key_summary(self, env_name: str) -> str:
        if not env_name:
            return "无需"
        source = source_of(env_name)
        if not source:
            return "未设置"
        return f"{mask(os.environ.get(env_name, ''))} ({source})"


def _setup_readline(history_path: Path) -> None:
    if readline is None:
        return
    try:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        readline.read_history_file(history_path)
    except OSError:
        pass
    readline.set_history_length(1000)


def _save_readline(history_path: Path) -> None:
    if readline is None:
        return
    try:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        readline.write_history_file(history_path)
    except OSError:
        pass


def run_repl(config_path: str = "config.json") -> int:
    session = Session(config_path)
    settings = session.settings
    creds = credentials_path(settings.credentials_file)
    _setup_readline(creds.parent / "history")

    chat = settings.chat_provider or "未配置"
    print(_color("RAG Multidocs CLI", "1"))
    print(
        f"config={Path(config_path)}  index={settings.index_backend}"
        f"  chat={chat}  credentials={creds}"
    )
    warning = permission_warning(settings.credentials_file)
    if warning:
        print(f"[WARN] {warning}", file=sys.stderr)
    print("直接输入问题开始；/help 查看命令，输入 exit 或 Ctrl-D 退出。")

    prompt = _input_prompt("rag> ", "1;32")
    try:
        while session._running:
            try:
                line = input(prompt).strip()
            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                print()
                continue
            if not line:
                continue
            try:
                session.dispatch(line)
            except KeyboardInterrupt:
                print("\n[已取消]")
            except SystemExit:
                break
            except Exception as exc:  # 单条命令失败不应退出会话
                print(f"[ERROR] {exc}", file=sys.stderr)
    finally:
        try:
            session.close()
        finally:
            _save_readline(creds.parent / "history")
    print("bye")
    return 0


__all__ = ["Session", "run_repl"]
