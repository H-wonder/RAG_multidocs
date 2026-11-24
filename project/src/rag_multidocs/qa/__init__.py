"""Retrieval-augmented QA helpers."""

from .llm import (
    ChatModel,
    OpenAICompatibleChat,
    SYSTEM_PROMPT,
    answer_with_evidence,
    build_chat_model,
    build_evidence,
    build_messages,
)

__all__ = [
    "ChatModel",
    "OpenAICompatibleChat",
    "SYSTEM_PROMPT",
    "answer_with_evidence",
    "build_chat_model",
    "build_evidence",
    "build_messages",
]
