"""``rag-cli chunks`` 子命令：浏览 chunk 内容。"""

from __future__ import annotations

import argparse
import json
import sys

from ...config import Settings
from ..runtime import build_runtime


def cmd_chunks_list(args: argparse.Namespace) -> int:
    settings = Settings.load(args.config)
    runtime = build_runtime(settings)
    if runtime.warning:
        print(f"[WARN] {runtime.warning}", file=sys.stderr)
    chunks = runtime.index.list_chunks(doc_id=args.doc_id, page=args.page)

    if args.json:
        print(json.dumps(chunks, ensure_ascii=False, indent=2))
        return 0
    if not chunks:
        print("no chunks")
        return 0
    for chunk in chunks:
        section = f" §{chunk.get('section_path')}" if chunk.get("section_path") else ""
        snippet = (chunk.get("text") or "").replace("\n", " ")[:80]
        print(
            f"{chunk['id']}  p{chunk.get('page', 0)}"
            f"  {chunk.get('content_type', 'text')}{section}"
        )
        print(f"    {snippet}")
    return 0


def cmd_chunks_show(args: argparse.Namespace) -> int:
    settings = Settings.load(args.config)
    runtime = build_runtime(settings)
    if runtime.warning:
        print(f"[WARN] {runtime.warning}", file=sys.stderr)
    chunk = runtime.index.get_chunk(args.chunk_id)
    if chunk is None:
        print(f"[CHUNK_NOT_FOUND] 未找到 chunk: {args.chunk_id}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(chunk, ensure_ascii=False, indent=2))
        return 0
    for key in (
        "id", "doc_id", "source", "index", "content_type",
        "section_path", "page", "page_end", "bbox",
    ):
        value = chunk.get(key)
        if value not in ("", None, 0):
            print(f"{key}: {value}")
    print("text:")
    print(chunk.get("text", ""))
    return 0


__all__ = ["cmd_chunks_list", "cmd_chunks_show"]
