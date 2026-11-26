"""``rag-cli docs`` 子命令。"""

from __future__ import annotations

import argparse
import json
import sys

from ...config import Settings
from ..runtime import build_runtime


def cmd_docs(args: argparse.Namespace) -> int:
    settings = Settings.load(args.config)
    runtime = build_runtime(settings)
    if runtime.warning:
        print(f"[WARN] {runtime.warning}", file=sys.stderr)
    docs = runtime.index.list_docs()
    if args.json:
        print(json.dumps(docs, ensure_ascii=False, indent=2))
        return 0
    if not docs:
        print("no documents")
        return 0
    print(f"{'source':40} {'chunks':>6} {'pages':>6}  doc_id")
    for doc in docs:
        print(
            f"{doc['source']:40} {doc['chunks']:6} {doc['pages']:6}  {doc['doc_id']}"
        )
    return 0


__all__ = ["cmd_docs"]
