"""``rag-cli connectors`` 子命令。"""

from __future__ import annotations

import argparse
import json
import sys

from ...config import Settings
from ..runtime import build_registry


def cmd_connectors_list(args: argparse.Namespace) -> int:
    settings = Settings.load(args.config)
    registry = build_registry(settings)
    enabled = set(settings.enabled_connectors)
    rows = []
    for name in registry.builtin_names():
        rows.append({"name": name, "kind": "builtin", "enabled": name in enabled})
    for info in registry.plugin_infos():
        rows.append(
            {
                "name": info.name,
                "kind": "plugin",
                "enabled": info.enabled,
                "version": info.version,
                "description": info.description,
                "path": str(info.path),
            }
        )

    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    for row in rows:
        mark = "x" if row["enabled"] else " "
        suffix = f"  {row.get('description', '')}" if row.get("description") else ""
        print(f"[{mark}] {row['name']:24} {row['kind']:8}{suffix}")
    return 0


def cmd_connectors_toggle(args: argparse.Namespace) -> int:
    settings = Settings.load(args.config)
    registry = build_registry(settings)
    known = set(registry.all_names())
    if args.name not in known:
        print(
            f"[CONNECTOR_NOT_FOUND] 未知 Connector: {args.name}"
            f"（可用: {', '.join(sorted(known)) or '无'}）",
            file=sys.stderr,
        )
        return 2

    enabled = set(settings.enabled_connectors)
    if args.action == "enable":
        enabled.add(args.name)
    else:
        enabled.discard(args.name)
    settings.enabled_connectors = sorted(enabled)
    path = settings.save(args.config)
    print(f"{args.action}d {args.name}; config -> {path}")
    return 0


__all__ = ["cmd_connectors_list", "cmd_connectors_toggle"]
