"""Connector registry: built-ins plus manifest/adapter plugins.

A plugin directory contains:

    <plugins_dir>/<plugin_name>/
    ├── manifest.json
    └── adapter.py          # def load(payload, context=None)

manifest.json:

    {
      "name": "connector.markdown",
      "version": "0.1.0",
      "description": "Markdown directory connector",
      "entry": "adapter:load",
      "enabled": true,
      "input_schema": {"type": "object", "required": ["path"]}
    }

Core loads the adapter by file path; the adapter may import ``rag_multidocs``
public modules. It must not import core internals beyond ``contracts``/``model``.
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import (
    Connector,
    ConnectorConfig,
    ConnectorError,
    ConnectorIssue,
    ConnectorResult,
)


@dataclass
class PluginInfo:
    name: str
    version: str
    description: str
    path: Path
    enabled: bool
    entry: str = "adapter:load"

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "path": str(self.path),
            "enabled": self.enabled,
            "entry": self.entry,
        }


class ConnectorRegistry:
    def __init__(
        self,
        plugins_dir: str | Path | None = None,
        enabled: set[str] | None = None,
    ):
        self.plugins_dir = Path(plugins_dir) if plugins_dir else None
        self.enabled = set(enabled) if enabled is not None else None
        self._builtins: dict[str, Connector] = {}

    # ---------------------------------------------------------------- built-ins

    def register(self, connector: Connector) -> None:
        self._builtins[connector.name] = connector

    def builtin_names(self) -> list[str]:
        return sorted(self._builtins)

    # ---------------------------------------------------------------- plugins

    def plugin_infos(self) -> list[PluginInfo]:
        if not self.plugins_dir or not self.plugins_dir.is_dir():
            return []
        infos: list[PluginInfo] = []
        for directory in sorted(self.plugins_dir.iterdir()):
            manifest_path = directory / "manifest.json"
            if not directory.is_dir() or not manifest_path.exists():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            name = manifest.get("name")
            if not name:
                continue
            enabled = (
                name in self.enabled
                if self.enabled is not None
                else bool(manifest.get("enabled", True))
            )
            infos.append(
                PluginInfo(
                    name=name,
                    version=str(manifest.get("version", "")),
                    description=str(manifest.get("description", "")),
                    path=directory,
                    enabled=enabled,
                    entry=str(manifest.get("entry", "adapter:load")),
                )
            )
        return infos

    def all_names(self) -> list[str]:
        return sorted(set(self._builtins) | {info.name for info in self.plugin_infos()})

    def _find_plugin(self, name: str) -> PluginInfo | None:
        for info in self.plugin_infos():
            if info.name == name:
                return info
        return None

    def _load_plugin(self, info: PluginInfo, config: ConnectorConfig) -> ConnectorResult:
        if not info.enabled:
            return ConnectorResult(
                issues=[ConnectorIssue("CONNECTOR_DISABLED", f"{info.name} 已禁用")]
            )
        module_name, _, func_name = info.entry.partition(":")
        adapter_path = info.path / f"{module_name}.py"
        if not adapter_path.exists():
            return ConnectorResult(
                issues=[
                    ConnectorIssue(
                        "CONNECTOR_PROTOCOL_ERROR",
                        f"插件缺少 adapter: {adapter_path}",
                    )
                ]
            )
        try:
            spec = importlib.util.spec_from_file_location(
                f"rag_multidocs_plugin_{info.name.replace('.', '_')}", adapter_path
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"无法加载插件 {info.name}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            func = getattr(module, func_name)
            raw = func(config.as_dict(), {})
            return self._normalize(raw)
        except ConnectorError as exc:
            return ConnectorResult(
                issues=[ConnectorIssue(exc.code, exc.message, exc.source)]
            )
        except Exception as exc:  # 扩展失败不能拖垮核心
            return ConnectorResult(
                issues=[
                    ConnectorIssue(
                        "CONNECTOR_ERROR",
                        f"插件 {info.name} 执行失败: {exc}",
                        config.path,
                    )
                ]
            )

    @staticmethod
    def _normalize(raw: Any) -> ConnectorResult:
        if isinstance(raw, ConnectorResult):
            return raw
        if isinstance(raw, list):
            return ConnectorResult(documents=raw)
        if isinstance(raw, dict) and "documents" in raw:
            issues = [
                item if isinstance(item, ConnectorIssue) else ConnectorIssue(**item)
                for item in raw.get("issues", [])
            ]
            return ConnectorResult(documents=list(raw["documents"]), issues=issues)
        return ConnectorResult(
            issues=[
                ConnectorIssue(
                    "CONNECTOR_PROTOCOL_ERROR",
                    "插件必须返回 ConnectorResult 或 Document 列表",
                )
            ]
        )

    # ---------------------------------------------------------------- load

    def load(self, name: str, config: ConnectorConfig) -> ConnectorResult:
        connector = self._builtins.get(name)
        if connector is not None:
            if self.enabled is not None and name not in self.enabled:
                return ConnectorResult(
                    issues=[ConnectorIssue("CONNECTOR_DISABLED", f"{name} 已禁用")]
                )
            try:
                return connector.load(config)
            except ConnectorError as exc:
                return ConnectorResult(
                    issues=[ConnectorIssue(exc.code, exc.message, exc.source)]
                )
            except Exception as exc:
                return ConnectorResult(
                    issues=[
                        ConnectorIssue(
                            "CONNECTOR_ERROR", f"{name} 执行失败: {exc}", config.path
                        )
                    ]
                )

        info = self._find_plugin(name)
        if info is None:
            return ConnectorResult(
                issues=[
                    ConnectorIssue(
                        "CONNECTOR_NOT_FOUND",
                        f"未知 Connector: {name}（可用: {', '.join(self.all_names()) or '无'}）",
                    )
                ]
            )
        return self._load_plugin(info, config)


__all__ = ["ConnectorRegistry", "PluginInfo"]
