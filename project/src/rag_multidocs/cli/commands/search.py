"""``rag-cli search`` 子命令。"""

from __future__ import annotations

import argparse
import json
import sys

from ...config import Settings
from ..runtime import build_runtime


def cmd_search(args: argparse.Namespace) -> int:
    settings = Settings.load(args.config)
    runtime = build_runtime(settings)
    if runtime.warning:
        print(f"[WARN] {runtime.warning}", file=sys.stderr)
    hits = runtime.search.search(args.query, top_k=args.top_k, doc_ids=args.doc_id)

    if args.json:
        print(json.dumps(hits, ensure_ascii=False, indent=2))
        return 0
    if not hits:
        print("no hits")
        return 0

    for rank, hit in enumerate(hits, start=1):
        section = f"  §{hit['section_path']}" if hit.get("section_path") else ""
        print(
            f"[{rank}] {hit['score']:.4f}  {hit['source']}"
            f" p{hit.get('page', 0)}{section}"
        )
        snippet = (hit.get("text") or "").replace("\n", " ")
        print(f"    {snippet[:120]}")
    return 0


__all__ = ["cmd_search"]
