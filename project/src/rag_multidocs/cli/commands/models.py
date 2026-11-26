"""``rag-cli models`` 子命令（list / discover / use）。"""

from __future__ import annotations

import argparse
import json
import sys

from ...config import Settings
from ...providers import (
    chat_provider,
    embedding_provider,
    key_available,
    list_chat_providers,
    list_embedding_providers,
)
from ...providers.discovery import DiscoveryError, discover_models
from ..render import key_label


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
        key = key_label(row.get("api_key_env", ""))
        print(
            f"[{mark}] {row['category']:9} {row['provider']:10} {model}{dim}"
            f"  key={key}  available={row['available']}"
        )
    return 0


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
            preset = embedding_provider(args.provider)
            settings.embed_backend = args.provider
            # 切换 provider 时重置为 provider 默认，避免沿用旧模型的维度
            settings.embed_model = args.model or ""
            settings.embed_dim = args.dim if args.dim is not None else 0
            settings.embed_base_url = args.base_url or ""
            dim = settings.embed_dim or preset.default_dim
            print(
                f"embedding -> {settings.embed_backend}"
                f" model={settings.embed_model or preset.default_model}"
                f" dim={dim}"
            )
            if settings.index_backend == "milvus":
                print(
                    f"将使用 Milvus 集合 {settings.milvus_collection}（dim={dim}）；"
                    f"换模型/维度建议新集合 {settings.milvus_collection}_d{dim}"
                )
            else:
                print("当前 keyword 模式，暂不涉及 Milvus 集合。")
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


__all__ = ["cmd_models_discover", "cmd_models_list", "cmd_models_use"]
