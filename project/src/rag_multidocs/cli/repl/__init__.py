"""REPL 入口（``rag-cli`` 无子命令时进入）。"""

from __future__ import annotations

import sys
from pathlib import Path

from ...credentials import credentials_path, permission_warning
from .readline import save_readline, setup_readline
from .render import color, input_prompt
from .session import Session

__all__ = ["Session", "run_repl"]


def run_repl(config_path: str = "config.json") -> int:
    session = Session(config_path)
    settings = session.settings
    creds = credentials_path(settings.credentials_file)
    setup_readline(creds.parent / "history")

    chat = settings.chat_provider or "未配置"
    print(color("RAG Multidocs CLI", "1"))
    print(
        f"config={Path(config_path)}  index={settings.index_backend}"
        f"  chat={chat}  credentials={creds}"
    )
    warning = permission_warning(settings.credentials_file)
    if warning:
        print(f"[WARN] {warning}", file=sys.stderr)
    print("直接输入问题开始；/help 查看命令，输入 exit 或 Ctrl-D 退出。")

    prompt = input_prompt("rag> ", "1;32")
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
        save_readline(creds.parent / "history")
    print("bye")
    return 0
