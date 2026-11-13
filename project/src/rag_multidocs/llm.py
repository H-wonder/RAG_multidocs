"""OpenAI-compatible chat client and retrieval-augmented answer construction.

API keys are read from environment variables only (see ``providers.py``).
The generated answer is always paired with the retrieved evidence; callers must
not present model output as if it were a retrieved fact.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Protocol

from .providers import api_key, chat_provider

SYSTEM_PROMPT = """你是课程资料检索助手。只能依据用户提供的“证据”回答问题。
要求：
1. 每个结论后用 [编号] 标注证据，例如 [1]、[2]；
2. 证据不足或互相矛盾时，明确说“根据现有资料无法确定”，不要编造；
3. 不要引用证据之外的事实；
4. 用简洁中文回答。"""


class ChatModel(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> str:
        ...


class OpenAICompatibleChat:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key_value: str = "",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout: int = 60,
        retries: int = 2,
    ):
        self.base_url = base_url.rstrip("/")
        self.url = (
            self.base_url
            if self.base_url.endswith("/chat/completions")
            else f"{self.base_url}/chat/completions"
        )
        self.model = model
        self.api_key = api_key_value
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries

    def _post(self, messages: list[dict[str, str]]) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.url, data=payload, headers=headers, method="POST"
        )
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                return str(data["choices"][0]["message"]["content"]).strip()
            except Exception as exc:  # 网络/限流/畸形响应都按可重试处理
                last_error = exc
                if attempt + 1 < self.retries:
                    time.sleep(1 + attempt)
        raise RuntimeError(f"LLM 调用失败: {last_error}") from last_error

    def complete(self, messages: list[dict[str, str]]) -> str:
        return self._post(messages)


def build_chat_model(
    provider: str,
    model: str = "",
    base_url: str = "",
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> OpenAICompatibleChat:
    preset = chat_provider(provider)
    return OpenAICompatibleChat(
        base_url=base_url or preset.base_url,
        model=model or preset.default_model,
        api_key_value=api_key(preset),
        temperature=temperature,
        max_tokens=max_tokens,
    )


def build_evidence(hits: list[dict[str, Any]], max_chars: int = 8000) -> str:
    """Render retrieved chunks as numbered evidence with source metadata."""
    parts: list[str] = []
    used = 0
    for index, hit in enumerate(hits, start=1):
        section = f" §{hit['section_path']}" if hit.get("section_path") else ""
        header = (
            f"[{index}] {hit.get('source', '')}"
            f" p{hit.get('page', 0)}{section}"
        )
        text = str(hit.get("text", ""))[:1200]
        block = f"{header}\n{text}"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)


def build_messages(question: str, hits: list[dict[str, Any]]) -> list[dict[str, str]]:
    evidence = build_evidence(hits)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"问题：{question}\n\n"
                f"证据：\n{evidence}\n\n"
                "请依据证据回答，并用 [编号] 标注引用。"
            ),
        },
    ]


def answer_with_evidence(
    question: str,
    hits: list[dict[str, Any]],
    chat: ChatModel,
    max_evidence_chars: int = 8000,
) -> dict[str, Any]:
    """Retrieve-backed answer + citations; never call the model without evidence."""
    citations = [
        {
            "index": index,
            "chunk_id": hit.get("chunk_id"),
            "doc_id": hit.get("doc_id"),
            "source": hit.get("source", ""),
            "page": hit.get("page", 0),
            "section_path": hit.get("section_path", ""),
        }
        for index, hit in enumerate(hits, start=1)
    ]
    if not hits:
        return {
            "answer": "没有检索到相关依据。",
            "citations": [],
            "used_llm": False,
        }

    messages = build_messages(question, hits)
    # build_messages uses default max evidence; rebuild if a custom budget is set
    if max_evidence_chars != 8000:
        evidence = build_evidence(hits, max_chars=max_evidence_chars)
        messages[1]["content"] = (
            f"问题：{question}\n\n"
            f"证据：\n{evidence}\n\n"
            "请依据证据回答，并用 [编号] 标注引用。"
        )
    answer = chat.complete(messages)
    return {"answer": answer, "citations": citations, "used_llm": True}


__all__ = [
    "ChatModel",
    "OpenAICompatibleChat",
    "SYSTEM_PROMPT",
    "answer_with_evidence",
    "build_chat_model",
    "build_evidence",
    "build_messages",
]
