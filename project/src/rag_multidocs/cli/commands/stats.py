"""``rag-cli stats`` 子命令。"""

from __future__ import annotations

import argparse
import json
import sys

from ...config import Settings
from ..runtime import build_runtime


def cmd_stats(args: argparse.Namespace) -> int:
    settings = Settings.load(args.config)
    runtime = build_runtime(settings)
    if runtime.warning:
        print(f"[WARN] {runtime.warning}", file=sys.stderr)
    data = {**runtime.index.stats(), "backend": runtime.backend}
    if runtime.backend == "keyword":
        data["index_file"] = str(settings.index_file)
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        for key, value in data.items():
            print(f"{key}: {value}")
    return 0


__all__ = ["cmd_stats"]
