"""共享渲染：检索命中与 Key 标签。"""

from __future__ import annotations

import os

from ..credentials import mask, source_of


def print_hits(hits, snippet_chars: int = 120) -> None:
    if not hits:
        print("no hits")
        return
    for rank, hit in enumerate(hits, start=1):
        section = f"  §{hit['section_path']}" if hit.get("section_path") else ""
        print(
            f"[{rank}] {hit['score']:.4f}  {hit['source']}"
            f" p{hit.get('page', 0)}{section}"
        )
        snippet = (hit.get("text") or "").replace("\n", " ")
        print(f"    {snippet[:snippet_chars]}")


def key_label(env_name: str) -> str:
    if not env_name:
        return "无需 Key"
    source = source_of(env_name)
    value = mask(os.environ.get(env_name, ""))
    if not source:
        return f"{env_name}[未设置]"
    return f"{env_name}[{source}] {value}"


__all__ = ["key_label", "print_hits"]
