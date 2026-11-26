"""``rag-cli models test``：真实调用一次 provider，验证配置是否真的可用。

与 ``models list`` 的区别：list 只检查 Key 是否存在（静态），test 会真的发一次
embedding / chat 请求（动态），输出模型名、维度与耗时，并对无 Key、模型不存在、
超时给出明确错误（错误信息不包含 Key）。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass

from ..config import Settings
from ..providers import chat_provider, embedding_provider, key_available


@dataclass
class ModelTestResult:
    kind: str           # embedding | chat
    provider: str
    model: str
    ok: bool
    latency_ms: int
    dim: int = 0        # embedding 实际维度
    reply: str = ""     # chat 回复摘要
    error: str = ""     # 失败原因（不含 Key）
    hint: str = ""      # 可执行建议


def _classify(exc: Exception) -> tuple[str, str]:
    """把异常归到「无 Key / 模型不存在 / 超时 / 限流」等明确类别。"""
    text = str(exc)
    lowered = text.lower()
    if "timed out" in lowered or "timeout" in lowered:
        return text, "网络超时，检查网络/代理后重试"
    if "404" in text:
        return text, "模型名可能不存在，用 rag-cli models discover 查询可用模型"
    if "401" in text or "403" in text:
        return text, "API Key 无效或无权限，检查 Key 是否正确"
    if "429" in text:
        return text, "触发限流，稍后重试"
    return text, ""


def _sanitize(text: str, secret: str) -> str:
    """防御性清洗：万一 provider 报错时把 Key 带进消息，替换为 ****。"""
    if secret and secret in text:
        return text.replace(secret, "****")
    return text


def test_embedding(
    settings: Settings,
    provider: str,
    model: str,
    base_url: str,
    dim: int,
) -> ModelTestResult:
    try:
        preset = embedding_provider(provider)
    except ValueError as exc:
        return ModelTestResult(
            "embedding", provider, model or "", False, 0,
            error=str(exc),
            hint="rag-cli models use --kind embedding --provider ...",
        )
    resolved_model = model or preset.default_model
    if not key_available(preset):
        return ModelTestResult(
            "embedding", provider, resolved_model, False, 0,
            error=f"未设置 {preset.api_key_env}",
            hint=f"rag-cli keys set --provider {preset.name}",
        )
    try:
        from ..providers.embedding import build_embedder

        start = time.monotonic()
        embedder = build_embedder(provider, model=model, dim=dim, base_url=base_url)
        vectors = embedder.embed(["测试"], query=False)
        latency_ms = int((time.monotonic() - start) * 1000)
        actual_dim = len(vectors[0]) if vectors else 0
        return ModelTestResult(
            "embedding", provider, resolved_model, True, latency_ms, dim=actual_dim,
        )
    except Exception as exc:
        error, hint = _classify(exc)
        error = _sanitize(error, os.environ.get(preset.api_key_env, ""))
        return ModelTestResult(
            "embedding", provider, resolved_model, False, 0, error=error, hint=hint,
        )


def test_chat(
    settings: Settings,
    provider: str,
    model: str,
    base_url: str,
) -> ModelTestResult:
    try:
        preset = chat_provider(provider)
    except ValueError as exc:
        return ModelTestResult(
            "chat", provider, model or "", False, 0,
            error=str(exc),
            hint="rag-cli models use --kind chat --provider ...",
        )
    resolved_model = model or preset.default_model
    if not key_available(preset):
        return ModelTestResult(
            "chat", provider, resolved_model, False, 0,
            error=f"未设置 {preset.api_key_env}",
            hint=f"rag-cli keys set --provider {preset.name}",
        )
    try:
        from ..qa.llm import build_chat_model

        start = time.monotonic()
        chat = build_chat_model(provider, model=model, base_url=base_url)
        reply = chat.complete([{"role": "user", "content": "你好"}])
        latency_ms = int((time.monotonic() - start) * 1000)
        snippet = (reply or "").replace("\n", " ").strip()[:80]
        return ModelTestResult(
            "chat", provider, resolved_model, True, latency_ms, reply=snippet,
        )
    except Exception as exc:
        error, hint = _classify(exc)
        error = _sanitize(error, os.environ.get(preset.api_key_env, ""))
        return ModelTestResult(
            "chat", provider, resolved_model, False, 0, error=error, hint=hint,
        )


def render(result: ModelTestResult, *, json_output: bool = False) -> str:
    if json_output:
        return json.dumps(asdict(result), ensure_ascii=False, indent=2)
    if result.ok:
        if result.kind == "embedding":
            return (
                f"[PASS] {result.provider} {result.model}"
                f" dim={result.dim} {result.latency_ms}ms"
            )
        return (
            f"[PASS] {result.provider} {result.model}"
            f" {result.latency_ms}ms -> {result.reply}"
        )
    lines = [f"[FAIL] {result.provider} {result.model}: {result.error}"]
    if result.hint:
        lines.append(f"      建议: {result.hint}")
    return "\n".join(lines)


def run_model_tests(
    settings: Settings,
    kind: str | None,
    provider: str,
    model: str,
    base_url: str,
    dim: int,
) -> list[ModelTestResult]:
    """按 kind 跑 embedding / chat 测试；kind 为空则测当前配置的两个。"""
    kinds = [kind] if kind else ["embedding"]
    if not kind and settings.chat_provider:
        kinds.append("chat")

    results: list[ModelTestResult] = []
    for kind in kinds:
        if kind == "embedding":
            results.append(
                test_embedding(
                    settings,
                    provider or settings.embed_backend,
                    model or "",
                    base_url or "",
                    dim or 0,
                )
            )
        else:
            if not settings.chat_provider and not provider:
                results.append(
                    ModelTestResult(
                        "chat", "", "", False, 0,
                        error="未配置 chat provider",
                        hint="rag-cli setup 选择 chat provider",
                    )
                )
                continue
            results.append(
                test_chat(
                    settings,
                    provider or settings.chat_provider,
                    model or "",
                    base_url or "",
                )
            )
    return results


def cmd_models_test(args: argparse.Namespace) -> int:
    settings = Settings.load(args.config)
    results = run_model_tests(
        settings,
        args.kind,
        args.provider or "",
        args.model or "",
        args.base_url or "",
        args.dim or 0,
    )
    for result in results:
        print(render(result, json_output=args.json))
    return 0 if all(result.ok for result in results) else 1


__all__ = [
    "ModelTestResult",
    "cmd_models_test",
    "render",
    "run_model_tests",
    "test_chat",
    "test_embedding",
]
