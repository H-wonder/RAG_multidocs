"""Interactive setup wizard: pick providers/models and store API keys."""

from __future__ import annotations

import getpass
import os
import sys
from pathlib import Path

from ..config import Settings
from ..credentials import set_key
from ..providers import (
    CHAT_PROVIDERS,
    EMBEDDING_PROVIDERS,
    chat_provider,
    embedding_provider,
)


def _pick(provider, current: str) -> bool:
    return provider.name == current


def _show_options(
    title: str,
    options: list,
    current: str,
    *,
    allow_skip: bool,
    out=print,
) -> None:
    out(title)
    if allow_skip:
        keep = f"当前 {current}" if current else "当前未启用"
        out(f"  [0] 不启用问答（{keep}）")
    for index, preset in enumerate(options, start=1):
        mark = "*" if _pick(preset, current) else " "
        out(f"  [{index}]{mark} {preset.name:10} {preset.default_model}")


def _select(preset_type, options: list, answer: str):
    answer = answer.strip()
    if not answer:
        return None
    if answer.isdigit():
        index = int(answer)
        if 1 <= index <= len(options):
            return options[index - 1]
        raise ValueError(f"编号超出范围: {answer}")
    try:
        return preset_type(answer)
    except ValueError as exc:
        raise ValueError(f"{exc}") from exc


def _configure_chat(
    settings: Settings,
    credentials_file: str,
    *,
    ask,
    secret,
    out,
) -> None:
    options = list(CHAT_PROVIDERS.values())
    _show_options(
        "\n== 第 1/2 步：Chat 模型（生成答案；可用 /model 切换）==",
        options,
        settings.chat_provider,
        allow_skip=True,
        out=out,
    )
    choice = ask(
        "选择编号或 provider 名 [回车保持当前，输入 0 关闭生成式问答]: "
    )
    if choice.strip() == "0":
        settings.chat_provider = ""
        settings.chat_model = ""
        out("  已关闭生成式问答（仍可用检索）。")
        return
    preset = _select(chat_provider, options, choice)
    if preset is None:
        return
    settings.chat_provider = preset.name
    model = ask(f"  模型名 [{preset.default_model}]: ").strip()
    settings.chat_model = model
    if preset.api_key_env:
        env_name = preset.api_key_env
        existing = bool(_env_value(env_name))
        hint = "回车保留已保存的 Key" if existing else "留空则稍后再设置"
        value = secret(f"  {env_name}（输入不回显，{hint}）: ").strip()
        if value:
            path = set_key(env_name, value, credentials_file)
            out(f"  已保存 {env_name} -> {path}（权限 600）")
        elif not existing:
            out(
                f"  未设置 {env_name}；稍后可用 "
                f"rag-cli keys set --provider {preset.name} 补上。"
            )


def _configure_embedding(
    settings: Settings,
    *,
    ask,
    secret,
    out,
) -> None:
    options = list(EMBEDDING_PROVIDERS.values())
    _show_options(
        "\n== 第 2/2 步：Embedding 模型（向量化入库与检索）==",
        options,
        settings.embed_backend,
        allow_skip=False,
        out=out,
    )
    choice = ask("选择编号或 provider 名 [回车保持当前]: ")
    preset = _select(embedding_provider, options, choice)
    if preset is None:
        return
    old = (settings.embed_backend, settings.embed_model, settings.embed_dim)
    settings.embed_backend = preset.name
    model = ask(f"  模型名 [{preset.default_model}]: ").strip()
    settings.embed_model = model
    dim = ask(f"  向量维度 [{preset.default_dim}]: ").strip()
    if dim:
        if not dim.isdigit() or int(dim) <= 0:
            raise ValueError(f"维度必须是正整数: {dim}")
        settings.embed_dim = int(dim)
    else:
        settings.embed_dim = 0
    if preset.api_key_env:
        env_name = preset.api_key_env
        existing = bool(_env_value(env_name))
        hint = "回车保留已保存的 Key" if existing else "留空则稍后再设置"
        value = secret(f"  {env_name}（输入不回显，{hint}）: ").strip()
        if value:
            path = set_key(env_name, value, settings.credentials_file)
            out(f"  已保存 {env_name} -> {path}（权限 600）")
        elif not existing:
            out(
                f"  未设置 {env_name}；稍后可用 "
                f"rag-cli keys set --provider {preset.name} 补上。"
            )
    if (settings.embed_backend, settings.embed_model, settings.embed_dim) != old:
        out("  注意：换 embedding provider/模型/维度后需要新的 Milvus 集合，旧向量不能复用。")


def run_setup(
    settings: Settings,
    config_path: str | Path,
    *,
    ask=input,
    secret=getpass.getpass,
    out=print,
) -> int:
    """Walk through chat and embedding settings, then save config + keys."""
    if not sys.stdin.isatty():
        print("[CONFIG_ERROR] setup 需要交互式终端", file=sys.stderr)
        return 2
    out(
        "setup 依次配置两步："
        "1) Chat 模型（生成答案，可选） 2) Embedding 模型（向量化入库与检索）。"
    )
    try:
        _configure_chat(
            settings,
            settings.credentials_file,
            ask=ask,
            secret=secret,
            out=out,
        )
        _configure_embedding(
            settings,
            ask=ask,
            secret=secret,
            out=out,
        )
    except ValueError as exc:
        print(f"[CONFIG_ERROR] {exc}", file=sys.stderr)
        return 2
    except (EOFError, KeyboardInterrupt):
        print("\n已取消，未保存配置。", file=sys.stderr)
        return 2

    path = settings.save(config_path)
    out(f"\n配置已保存 -> {path}")
    out(_summary(settings))
    out(
        f"凭证文件 -> {settings.credentials_file}"
        "（保存 Key 时创建，0600，已 gitignore）"
    )
    out("环境变量优先级高于本地凭证；Key 也可以用 rag-cli keys set 单独管理。")
    return 0


def _summary(settings: Settings) -> str:
    if settings.chat_provider:
        chat_preset = chat_provider(settings.chat_provider)
        chat = (
            f"{settings.chat_provider}/"
            f"{settings.chat_model or chat_preset.default_model}"
        )
    else:
        chat = "未启用（只检索）"
    embed_preset = embedding_provider(settings.embed_backend)
    embed = (
        f"{settings.embed_backend}/"
        f"{settings.embed_model or embed_preset.default_model}"
        f" dim={settings.embed_dim or embed_preset.default_dim}"
    )
    return f"当前选择：chat={chat}  embedding={embed}"


def _env_value(name: str) -> str:
    return os.environ.get(name, "")


__all__ = ["run_setup"]
