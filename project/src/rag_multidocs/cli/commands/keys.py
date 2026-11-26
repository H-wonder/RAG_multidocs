"""``rag-cli keys`` 子命令。"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import shlex
import sys

from ...config import Settings
from ...credentials import (
    credentials_path,
    load_keys,
    mask,
    permission_warning,
    set_key,
    source_of,
    unset_key,
)
from ...providers import (
    env_var_for_provider,
    list_chat_providers,
    list_embedding_providers,
    providers_for_env,
)


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


__all__ = [
    "cmd_keys_export",
    "cmd_keys_list",
    "cmd_keys_path",
    "cmd_keys_set",
    "cmd_keys_unset",
]
