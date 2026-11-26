"""``rag-cli doctor``：诊断配置、依赖服务与索引，输出可执行建议。

只做检查与渲染，不做修复。检查项尽量复用现有能力（``Settings``、
``providers``、``credentials``），网络探测用短超时，避免拖慢诊断。
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from ..config import Settings
from ..credentials import load_keys, permission_warning
from ..providers import chat_provider, embedding_provider, key_available

_STATUS_TAG = {"pass": "PASS", "warn": "WARN", "fail": "FAIL", "info": "INFO"}

# Ollama 默认监听地址（embedding/chat 的 ollama provider 与 docparse OCR 都用它）
OLLAMA_HOST = "127.0.0.1"
OLLAMA_PORT = 11434


@dataclass
class CheckResult:
    """单条诊断结果。status: pass | warn | fail | info。"""

    key: str
    label: str
    status: str
    message: str
    suggestion: str = ""

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "status": self.status,
            "message": self.message,
            "suggestion": self.suggestion,
        }


def _tcp_reachable(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _parse_host_port(uri: str) -> tuple[str, int]:
    parsed = urlparse(uri if "://" in uri else f"http://{uri}")
    host = parsed.hostname or "localhost"
    port = parsed.port or 19530
    return host, port


def _check_env() -> CheckResult:
    py = f"python {sys.version_info.major}.{sys.version_info.minor}"
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if conda_env == "rag":
        return CheckResult("env", "环境", "pass", f"conda env rag（{py}）")
    if conda_env:
        return CheckResult(
            "env", "环境", "warn",
            f"当前 conda 环境为 {conda_env}，非 rag（{py}）",
            "conda activate rag",
        )
    if Path(sys.prefix).name == "rag":
        return CheckResult("env", "环境", "pass", f"rag 环境（{py}）")
    return CheckResult(
        "env", "环境", "warn",
        f"未检测到 rag conda 环境（当前 {sys.prefix}）",
        "conda activate rag",
    )


def _check_config(settings: Settings, config_path: str) -> CheckResult:
    path = Path(config_path)
    if not path.exists():
        return CheckResult(
            "config", "配置", "info",
            f"{path} 不存在，使用默认配置",
            "rag-cli setup 可生成配置",
        )
    return CheckResult(
        "config", "配置", "pass",
        f"{path}（index_backend={settings.index_backend}）",
    )


def _check_milvus(settings: Settings) -> CheckResult:
    host, port = _parse_host_port(settings.milvus_uri)
    ok = _tcp_reachable(host, port)
    target = f"{settings.milvus_uri}（{host}:{port}）"
    if ok:
        return CheckResult("milvus", "Milvus", "pass", target)
    if settings.index_backend == "milvus":
        return CheckResult(
            "milvus", "Milvus", "fail",
            f"{target} 不可达，但 index_backend=milvus",
            "启动 Milvus 容器，或改 index_backend=keyword 回退",
        )
    return CheckResult(
        "milvus", "Milvus", "warn",
        f"{target} 不可达（当前 keyword 模式，混合检索不可用）",
        "启动 Milvus 后切换回 milvus",
    )


def _check_ollama(settings: Settings) -> CheckResult:
    ok = _tcp_reachable(OLLAMA_HOST, OLLAMA_PORT)
    if ok:
        return CheckResult("ollama", "Ollama", "pass", "127.0.0.1:11434 可达")
    needs = (
        settings.embed_backend == "ollama"
        or settings.chat_provider == "ollama"
        or bool(settings.docparse_repo)
    )
    if needs:
        return CheckResult(
            "ollama", "Ollama", "warn",
            "127.0.0.1:11434 不可达（扫描件 OCR / ollama provider 不可用）",
            "systemctl start ollama",
        )
    return CheckResult(
        "ollama", "Ollama", "info",
        "未运行（当前配置未使用 Ollama）",
        "systemctl start ollama",
    )


def _check_docparse(settings: Settings) -> CheckResult:
    repo = settings.docparse_repo
    python = settings.docparse_python
    if not repo and not python:
        return CheckResult(
            "docparse", "docparse", "warn",
            "未配置 docparse（扫描件 PDF 解析不可用）",
            "在 config.json 配置 docparse_repo / docparse_python",
        )
    problems = []
    if repo and not Path(repo).is_dir():
        problems.append(f"repo 不存在: {repo}")
    if python and not Path(python).is_file():
        problems.append(f"python 不存在: {python}")
    if problems:
        return CheckResult(
            "docparse", "docparse", "fail",
            "；".join(problems),
            "检查 config.json 中的 docparse 路径",
        )
    return CheckResult(
        "docparse", "docparse", "pass",
        f"python={python or '(未设置)'} repo={repo or '(未设置)'}",
    )


def _check_embedding(settings: Settings) -> CheckResult:
    try:
        preset = embedding_provider(settings.embed_backend)
    except ValueError as exc:
        return CheckResult(
            "embedding", "embedding", "fail", str(exc),
            "rag-cli models use --kind embedding --provider ...",
        )
    model = settings.embed_model or preset.default_model
    dim = settings.embed_dim or preset.default_dim
    if not key_available(preset):
        return CheckResult(
            "embedding", "embedding", "warn",
            f"{preset.name} {model} dim={dim}，未设置 {preset.api_key_env}",
            f"rag-cli keys set --provider {preset.name}",
        )
    return CheckResult(
        "embedding", "embedding", "pass",
        f"{preset.name} {model} dim={dim}",
    )


def _check_chat(settings: Settings) -> CheckResult:
    if not settings.chat_provider:
        return CheckResult(
            "chat", "chat", "info",
            "未配置（ask 仅返回检索证据，不生成答案）",
            "rag-cli setup 选择 chat provider",
        )
    try:
        preset = chat_provider(settings.chat_provider)
    except ValueError as exc:
        return CheckResult(
            "chat", "chat", "fail", str(exc),
            "rag-cli models use --kind chat --provider ...",
        )
    model = settings.chat_model or preset.default_model
    if not key_available(preset):
        return CheckResult(
            "chat", "chat", "warn",
            f"{preset.name} {model}，未设置 {preset.api_key_env}",
            f"rag-cli keys set --provider {preset.name}",
        )
    return CheckResult("chat", "chat", "pass", f"{preset.name} {model}")


def _on_windows_mount(path: Path) -> bool:
    """WSL 的 /mnt/ 下是 Windows 文件系统，不区分 Unix 权限，chmod 无效。"""
    return str(path.resolve()).startswith("/mnt/")


def _check_credentials(settings: Settings) -> CheckResult:
    path = Path(settings.credentials_file)
    if not path.exists():
        return CheckResult(
            "credentials", "凭证", "info",
            f"{path} 不存在（尚未保存 Key）",
            "rag-cli keys set --provider ...",
        )
    names = sorted(load_keys(settings.credentials_file))
    detail = (
        f"{path}（已存 {len(names)} 个 Key: {', '.join(names)}）"
        if names
        else f"{path}（空）"
    )
    warning = permission_warning(settings.credentials_file)
    if warning:
        if _on_windows_mount(path):
            return CheckResult(
                "credentials", "凭证", "info",
                f"{detail}；文件在 Windows 挂载下，权限由宿主机管理（chmod 无效）",
            )
        return CheckResult(
            "credentials", "凭证", "warn", f"{detail}；{warning}",
            f"chmod 600 {path}",
        )
    return CheckResult("credentials", "凭证", "pass", detail)


def _check_connectors(settings: Settings) -> CheckResult:
    from ..connectors.registry import ConnectorRegistry

    plugins_dir = Path(settings.connectors_dir)
    if not plugins_dir.is_dir():
        return CheckResult(
            "connectors", "connectors", "warn",
            f"connectors_dir 不存在: {plugins_dir}",
            "检查 config.json 的 connectors_dir",
        )
    registry = ConnectorRegistry(
        plugins_dir=settings.connectors_dir,
        enabled=set(settings.enabled_connectors),
    )
    enabled_infos = [info for info in registry.plugin_infos() if info.enabled]
    missing = []
    for info in enabled_infos:
        module_name = info.entry.partition(":")[0]
        adapter = info.path / f"{module_name}.py"
        if not adapter.exists():
            missing.append(f"{info.name}（缺 {adapter.name}）")
    if missing:
        return CheckResult(
            "connectors", "connectors", "fail",
            "；".join(missing),
            "检查插件目录的 adapter.py",
        )
    enabled_names = ", ".join(info.name for info in enabled_infos) or "（无插件）"
    return CheckResult(
        "connectors", "connectors", "pass",
        f"{len(enabled_infos)} 个插件已启用: {enabled_names}",
    )


def _check_milvus_dim(settings: Settings) -> CheckResult | None:
    """Milvus 可达时对比集合维度与 embedder 维度（仅 --full 时调用）。"""
    host, port = _parse_host_port(settings.milvus_uri)
    if not _tcp_reachable(host, port):
        return None
    try:
        from pymilvus import MilvusClient
    except ImportError:
        return None

    embedder_dim = (
        settings.embed_dim
        or embedding_provider(settings.embed_backend).default_dim
    )
    try:
        client = MilvusClient(uri=settings.milvus_uri, db_name=settings.milvus_db)
        try:
            if not client.has_collection(settings.milvus_collection):
                return CheckResult(
                    "milvus_dim", "Milvus 集合", "info",
                    f"集合 {settings.milvus_collection} 尚未创建（首次 ingest 自动建）",
                )
            desc = client.describe_collection(settings.milvus_collection)
            actual_dim = None
            for field in desc.get("fields", []):
                if field.get("name") == "dense":
                    actual_dim = field.get("params", {}).get("dim")
        finally:
            client.close()
    except Exception as exc:  # 连接/权限/版本差异都不应拖垮 doctor
        return CheckResult(
            "milvus_dim", "Milvus 集合", "warn",
            f"读取集合维度失败: {exc}",
        )

    if actual_dim is not None and int(actual_dim) != int(embedder_dim):
        return CheckResult(
            "milvus_dim", "Milvus 集合", "warn",
            f"{settings.milvus_collection} dim={actual_dim} != embedder dim={embedder_dim}",
            f"设置 MILVUS_COLLECTION={settings.milvus_collection}_d{embedder_dim}",
        )
    return CheckResult(
        "milvus_dim", "Milvus 集合", "pass",
        f"{settings.milvus_collection} dim={actual_dim}",
    )


def run_checks(
    settings: Settings,
    config_path: str,
    *,
    full: bool = False,
) -> list[CheckResult]:
    results = [
        _check_env(),
        _check_config(settings, config_path),
        _check_milvus(settings),
        _check_ollama(settings),
        _check_docparse(settings),
        _check_embedding(settings),
        _check_chat(settings),
        _check_credentials(settings),
        _check_connectors(settings),
    ]
    if full:
        dim = _check_milvus_dim(settings)
        if dim is not None:
            results.append(dim)
    return results


def render(results: list[CheckResult], *, json_output: bool = False) -> str:
    if json_output:
        return json.dumps(
            [result.as_dict() for result in results],
            ensure_ascii=False,
            indent=2,
        )
    lines = []
    for result in results:
        tag = _STATUS_TAG[result.status]
        lines.append(f"[{tag}] {result.label} {result.message}")
        if result.suggestion:
            lines.append(f"      建议: {result.suggestion}")
    return "\n".join(lines)


def cmd_doctor(args: argparse.Namespace) -> int:
    settings = Settings.load(args.config)
    results = run_checks(settings, args.config, full=args.full)
    print(render(results, json_output=args.json))
    return 1 if any(r.status == "fail" for r in results) else 0


__all__ = ["CheckResult", "cmd_doctor", "render", "run_checks"]
