"""Shared manifest discovery and loading for the two A+ extension points.

Manifest discovery is not a sandbox. Plugins are trusted local Python code.
Explicit allowlists override manifest defaults, preserving the original Connector policy.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import re
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PluginInfo:
    name: str
    version: str
    description: str
    path: Path
    enabled: bool
    entry: str = "adapter:load"
    kind: str = "connector"
    contract_version: str = "1"
    requires: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    execution: str = "process"
    file_extensions: tuple[str, ...] = ()
    auto_priority: int = 0

    def as_dict(self):
        return {**vars(self), "path": str(self.path)}


def discover(directory, enabled=None, *, kind="connector", issues=None):
    diagnostics = issues if issues is not None else []
    if not directory or not Path(directory).is_dir():
        return []
    infos = []
    for path in sorted(Path(directory).resolve().iterdir()):
        manifest = path / "manifest.json"
        if not path.is_dir() or not manifest.exists():
            continue
        try:
            raw = json.loads(manifest.read_text(encoding="utf8"))
            if not isinstance(raw, dict):
                raise ValueError("manifest object required")
            if raw.get("kind", "connector") != kind:
                continue
            if not isinstance(raw.get("name"), str) or not raw["name"].strip():
                raise ValueError("name required")
            if (
                not isinstance(raw.get("requires", []), list)
                or any(not isinstance(r, str) for r in raw.get("requires", []))
                or not isinstance(raw.get("config", {}), dict)
            ):
                raise ValueError("invalid requires/config")
            active = (
                raw["name"] in enabled if enabled is not None else bool(raw.get("enabled", True))
            )
            extensions = raw.get("file_extensions", []) if kind == "connector" else []
            priority = raw.get("auto_priority", 0) if kind == "connector" else 0
            if (not isinstance(extensions, list)
                or any(not isinstance(ext, str) or not re.fullmatch(r"\.[\w+-]+", ext) for ext in extensions)
                or type(priority) is not int):
                raise ValueError("invalid file_extensions/auto_priority")
            infos.append(
                PluginInfo(
                    raw["name"],
                    str(raw.get("version", "")),
                    str(raw.get("description", "")),
                    path,
                    active,
                    str(raw.get("entry", "adapter:load")),
                    kind,
                    str(raw.get("contract_version", "1")),
                    raw.get("requires", []),
                    raw.get("config", {}),
                    str(raw.get("execution", "process")),
                    tuple(sorted({ext.casefold() for ext in extensions})),
                    priority,
                )
            )
        except (OSError, ValueError, TypeError):
            diagnostics.append(
                {
                    "code": "MANIFEST_INVALID",
                    "message": "Invalid plugin manifest",
                    "source": str(manifest),
                    "severity": "error",
                }
            )
    for name in sorted({i.name for i in infos}):
        if sum(i.name == name for i in infos) > 1:
            diagnostics.append(
                {
                    "code": "PLUGIN_DUPLICATE",
                    "message": "Duplicate plugin name",
                    "plugin": name,
                    "severity": "error",
                }
            )
    return infos


def dependency_error(info):
    """Check the deliberately small v1 requirement grammar: name[, comparators].

    Complex environment markers/extras require explicit deployment preflight rather
    than installing dependencies from a plugin during a request.
    """
    for requirement in info.requires:
        match = re.fullmatch(
            r"([A-Za-z0-9_.-]+)\s*((?:(?:>=|<=|==|!=|>|<)\s*[0-9.]+\s*,?\s*)*)", requirement
        )
        if not match:
            return "DEPENDENCY_SPEC_ERROR"
        try:
            installed = importlib.metadata.version(match[1])
        except importlib.metadata.PackageNotFoundError:
            return "DEPENDENCY_MISSING"
        actual = tuple(int(n) for n in re.findall(r"\d+", installed.split("+")[0])[:3])
        actual += (0,) * (3 - len(actual))
        for op, version in re.findall(r"(>=|<=|==|!=|>|<)\s*([0-9.]+)", match[2]):
            target = tuple(int(n) for n in version.split("."))
            target += (0,) * max(0, 3 - len(target))
            valid = {
                ">=": actual >= target,
                "<=": actual <= target,
                "==": actual == target,
                "!=": actual != target,
                ">": actual > target,
                "<": actual < target,
            }[op]
            if not valid:
                return "DEPENDENCY_VERSION_ERROR"
    return None


def load_entry(info):
    """Load a function in a private package, supporting relative helper imports."""
    module, sep, function = info.entry.partition(":")
    if not sep or not module.isidentifier() or not function.isidentifier():
        raise ValueError("invalid plugin entry")
    prefix = "_rag_plugin_" + hashlib.sha256(str(info.path).encode()).hexdigest()[:16]
    full_name = prefix + "." + module
    if full_name not in sys.modules:
        package = types.ModuleType(prefix)
        package.__path__ = [str(info.path)]
        sys.modules[prefix] = package
        spec = importlib.util.spec_from_file_location(full_name, info.path / (module + ".py"))
        if spec is None or spec.loader is None:
            raise ImportError("plugin module missing")
        plugin = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = plugin
        try:
            spec.loader.exec_module(plugin)
        except BaseException:
            sys.modules.pop(full_name, None)
            raise
    entry = getattr(sys.modules[full_name], function)
    if not callable(entry):
        raise TypeError("plugin entry must be callable")
    return entry
