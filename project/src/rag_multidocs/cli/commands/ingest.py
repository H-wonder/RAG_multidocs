"""``rag-cli ingest`` 子命令。"""

from __future__ import annotations

import argparse
import json
import sys

from ...config import Settings
from ..runtime import build_runtime, print_issues


def cmd_ingest(args: argparse.Namespace) -> int:
    settings = Settings.load(args.config)
    runtime = build_runtime(settings)
    if runtime.warning:
        print(f"[WARN] {runtime.warning}", file=sys.stderr)
    options = {"mode": args.mode} if args.mode else {}
    report = runtime.ingest.ingest(args.source, args.path, options)

    if args.json:
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    else:
        print(f"connector: {report.connector}")
        print(f"index: {runtime.backend}")
        print(f"documents: {len(report.documents)}")
        print(f"chunks: {report.chunks}")
        print(f"issues: {len(report.issues)}")
        for document in report.documents:
            print(
                f"  + {document['source']}  {document['chunks']} chunks"
                f"  doc_id={document['doc_id']}"
            )
        print_issues(report.issues)

    return 0 if report.documents else 2


__all__ = ["cmd_ingest"]
