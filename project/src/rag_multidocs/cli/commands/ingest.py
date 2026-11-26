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

    def _progress(index: int, total: int, document) -> None:
        # 单文件不刷进度；目录导入显示 [i/N]
        if total > 1:
            print(f"[{index}/{total}] {document.source}", file=sys.stderr)

    report = runtime.ingest.ingest(
        args.source, args.path, options, progress=_progress
    )

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
        if report.issues:
            print(
                f"汇总: 成功 {len(report.documents)} 个文档，"
                f"失败/告警 {len(report.issues)} 个"
            )

    if args.strict and report.issues:
        return 1
    return 0 if report.documents else 2


__all__ = ["cmd_ingest"]
