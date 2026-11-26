"""REPL 渲染：颜色、提示、引用、流式输出。"""

from __future__ import annotations

import os
import sys

from .readline import readline


def color(text: str, code: str) -> str:
    if (
        not sys.stdout.isatty()
        or os.environ.get("NO_COLOR")
        or os.environ.get("TERM") == "dumb"
    ):
        return text
    return f"\033[{code}m{text}\033[0m"


def input_prompt(text: str, code: str) -> str:
    """Colored prompt with readline non-printing markers.

    readline 需要 ``\\001``/``\\002`` 包住 ANSI 转义，否则会把颜色码算进
    光标列数；在长命令、粘贴和退格时会出现错位或粘贴内容异常。
    """
    prompt = color(text, code)
    if readline is not None and prompt != text:
        return f"\001{prompt}\002"
    return prompt


def print_citations(citations: list[dict]) -> None:
    if not citations:
        return
    print("来源：")
    for citation in citations:
        section = f" §{citation['section_path']}" if citation["section_path"] else ""
        print(
            f"[{citation['index']}] {citation['source']}"
            f" p{citation['page']}{section}"
        )


def stream_token(token: str) -> None:
    print(token, end="", flush=True)


__all__ = ["color", "input_prompt", "print_citations", "stream_token"]
