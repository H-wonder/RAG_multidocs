"""``rag-cli config``：查看、校验配置，以及一键切换预设方案（profiles）。

- ``config show``：友好展示当前生效配置（Settings 不含 API Key，天然不泄露 Key）；
- ``config validate``：校验 provider 名、数值范围、路径，并检测历史环境变量；
- ``config profiles list/apply``：offline / local / cloud 三套预设一键切换。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass

from ..config import Settings
from ..providers import CHAT_PROVIDERS, EMBEDDING_PROVIDERS

# 历史环境变量名 -> 新名（迁移提示用；代码里已不再读取旧名）
LEGACY_ENV_VARS = {
    "PLATFORM_INDEX_BACKEND": "RAG_INDEX_BACKEND",
    "CONNECTORS_DIR": "RAG_CONNECTORS_DIR",
}

# 三套预设：切换 provider 时会重置 model/dim，避免沿用旧模型的参数。
PROFILES: dict[str, dict] = {
    "offline": {
        "description": "离线关键词检索（无 Milvus / 网络）",
        "index_backend": "keyword",
        "embed_backend": "hashing",
        "chat_provider": "",
    },
    "local": {
        "description": "本地 bge 嵌入 + Ollama 问答",
        "index_backend": "milvus",
        "embed_backend": "local",
        "chat_provider": "ollama",
    },
    "cloud": {
        "description": "云端 DashScope 嵌入 + DeepSeek 问答",
        "index_backend": "milvus",
        "embed_backend": "dashscope",
        "chat_provider": "deepseek",
    },
}

_SHOW_GROUPS: list[tuple[str, list[str]]] = [
    (
        "索引与检索",
        [
            "index_backend",
            "fallback_to_keyword",
            "milvus_uri",
            "milvus_db",
            "milvus_collection",
            "milvus_dense_weight",
            "milvus_sparse_weight",
            "rag_top_k",
        ],
    ),
    (
        "嵌入",
        ["embed_backend", "embed_model", "embed_dim", "embed_base_url"],
    ),
    (
        "问答",
        [
            "chat_provider",
            "chat_model",
            "chat_base_url",
            "chat_temperature",
            "chat_max_tokens",
        ],
    ),
    (
        "docparse",
        ["docparse_repo", "docparse_python", "docparse_mode", "docparse_timeout_s"],
    ),
    (
        "其他",
        ["data_dir", "connectors_dir", "enabled_connectors", "credentials_file"],
    ),
]


@dataclass
class ConfigIssue:
    level: str          # warn | fail
    message: str
    suggestion: str = ""


def _format_value(value) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) or "(空)"
    if value == "" or value is None:
        return "(未设置)"
    return str(value)


def cmd_config_show(args: argparse.Namespace) -> int:
    settings = Settings.load(args.config)
    if args.json:
        print(json.dumps(settings.to_dict(), ensure_ascii=False, indent=2))
        return 0
    print(f"config: {args.config}")
    for title, keys in _SHOW_GROUPS:
        print(f"[{title}]")
        for key in keys:
            print(f"  {key}: {_format_value(getattr(settings, key))}")
    return 0


def validate_settings(settings: Settings) -> list[ConfigIssue]:
    issues: list[ConfigIssue] = []

    if settings.index_backend not in ("milvus", "keyword"):
        issues.append(
            ConfigIssue(
                "fail", f"未知 index_backend: {settings.index_backend}",
                "可选 milvus / keyword",
            )
        )
    if settings.embed_backend not in EMBEDDING_PROVIDERS:
        issues.append(
            ConfigIssue(
                "fail", f"未知 embed_backend: {settings.embed_backend}",
                "可选 " + ", ".join(EMBEDDING_PROVIDERS),
            )
        )
    if settings.chat_provider and settings.chat_provider not in CHAT_PROVIDERS:
        issues.append(
            ConfigIssue(
                "fail", f"未知 chat_provider: {settings.chat_provider}",
                "可选 " + ", ".join(CHAT_PROVIDERS),
            )
        )

    if settings.rag_top_k <= 0:
        issues.append(
            ConfigIssue("fail", f"rag_top_k={settings.rag_top_k} 应为正整数")
        )
    if not (0.0 <= settings.chat_temperature <= 2.0):
        issues.append(
            ConfigIssue(
                "warn",
                f"chat_temperature={settings.chat_temperature} 超出常见范围 [0, 2]",
            )
        )
    if settings.docparse_timeout_s <= 0:
        issues.append(
            ConfigIssue(
                "warn", f"docparse_timeout_s={settings.docparse_timeout_s} 应为正数"
            )
        )
    total = settings.milvus_dense_weight + settings.milvus_sparse_weight
    if abs(total - 1.0) > 1e-6:
        issues.append(
            ConfigIssue(
                "warn",
                f"milvus_dense_weight + sparse_weight = {total}，通常应为 1.0",
            )
        )

    if settings.docparse_repo and not os.path.isdir(settings.docparse_repo):
        issues.append(
            ConfigIssue("fail", f"docparse_repo 不存在: {settings.docparse_repo}")
        )
    if settings.docparse_python and not os.path.isfile(settings.docparse_python):
        issues.append(
            ConfigIssue("fail", f"docparse_python 不存在: {settings.docparse_python}")
        )
    if settings.connectors_dir and not os.path.isdir(settings.connectors_dir):
        issues.append(
            ConfigIssue(
                "warn", f"connectors_dir 不存在: {settings.connectors_dir}"
            )
        )

    for old, new in LEGACY_ENV_VARS.items():
        if os.environ.get(old):
            issues.append(
                ConfigIssue(
                    "warn",
                    f"检测到旧环境变量 {old}，已改名 {new}",
                    f"export {new}=... 并 unset {old}",
                )
            )
    return issues


def cmd_config_validate(args: argparse.Namespace) -> int:
    settings = Settings.load(args.config)
    issues = validate_settings(settings)
    if args.json:
        print(json.dumps([asdict(i) for i in issues], ensure_ascii=False, indent=2))
    elif not issues:
        print("[PASS] 配置有效")
    else:
        for issue in issues:
            tag = "FAIL" if issue.level == "fail" else "WARN"
            print(f"[{tag}] {issue.message}")
            if issue.suggestion:
                print(f"      建议: {issue.suggestion}")
    return 1 if any(i.level == "fail" for i in issues) else 0


def _apply_profile(settings: Settings, name: str) -> None:
    profile = PROFILES[name]
    settings.index_backend = profile["index_backend"]
    settings.embed_backend = profile["embed_backend"]
    settings.embed_model = ""
    settings.embed_dim = 0
    settings.embed_base_url = ""
    settings.chat_provider = profile["chat_provider"]
    settings.chat_model = ""
    settings.chat_base_url = ""


def cmd_config_profiles(args: argparse.Namespace) -> int:
    settings = Settings.load(args.config)
    if args.profiles_command == "list":
        if args.json:
            print(json.dumps(PROFILES, ensure_ascii=False, indent=2))
        else:
            for name, profile in PROFILES.items():
                print(f"{name:8} {profile['description']}")
        return 0

    name = args.name
    if name not in PROFILES:
        print(
            f"[CONFIG_ERROR] 未知 profile: {name}（可选: {', '.join(PROFILES)}）",
            file=sys.stderr,
        )
        return 2
    _apply_profile(settings, name)
    path = settings.save(args.config)
    print(f"已应用 profile {name}：{PROFILES[name]['description']}")
    print(f"config -> {path}")
    return 0


__all__ = [
    "ConfigIssue",
    "LEGACY_ENV_VARS",
    "PROFILES",
    "cmd_config_profiles",
    "cmd_config_show",
    "cmd_config_validate",
    "validate_settings",
]
