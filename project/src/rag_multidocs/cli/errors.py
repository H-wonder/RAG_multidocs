"""统一的错误分类与脱敏，供 CLI 命令复用。"""

from __future__ import annotations


def classify_error(exc: Exception) -> tuple[str, str]:
    """把异常归到「无 Key / 模型不存在 / 超时 / 限流」等明确类别。

    返回 ``(message, hint)``，hint 是可执行的建议；识别不了的返回空建议。
    """
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


def sanitize(text: str, secret: str) -> str:
    """防御性清洗：万一报错把 Key 带进消息，替换为 ****。"""
    if secret and secret in text:
        return text.replace(secret, "****")
    return text


__all__ = ["classify_error", "sanitize"]
