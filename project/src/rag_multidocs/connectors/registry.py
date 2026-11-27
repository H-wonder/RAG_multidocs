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

import copy
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..contracts import (
    Connector,
    ConnectorConfig,
    ConnectorError,
    ConnectorIssue,
    ConnectorResult,
)


from ..extensions import PluginInfo, dependency_error, discover, load_entry
from ..model import Document


class ConnectorRegistry:
    def __init__(
        self,
        plugins_dir: str | Path | None = None,
        enabled: set[str] | None = None,
        context: dict[str, Any] | None = None,
    ):
        self.plugins_dir = Path(plugins_dir).resolve() if plugins_dir else None
        self.enabled = set(enabled) if enabled is not None else None
        # 公共只读上下文：核心把配置以普通 dict 传给插件，插件不需要 import Settings。
        self.context = dict(context or {})
        self._builtins: dict[str, Connector] = {}
        self.discovery_issues = []

    # ---------------------------------------------------------------- built-ins

    def register(self, connector: Connector) -> None:
        self._builtins[connector.name] = connector

    def builtin_names(self) -> list[str]:
        return sorted(self._builtins)

    # ---------------------------------------------------------------- plugins

    def plugin_infos(self) -> list[PluginInfo]:
        self.discovery_issues = []
        return discover(self.plugins_dir, self.enabled, kind="connector", issues=self.discovery_issues)

    def all_names(self) -> list[str]:
        return sorted(set(self._builtins) | {info.name for info in self.plugin_infos()})

    def file_routes(self) -> dict[str, tuple[str | None, str]]:
        """Resolve one route per suffix. Return diagnostics for declared but unavailable routes."""
        plugins = self.plugin_infos()
        counts = Counter([*self._builtins, *(p.name for p in plugins)])
        candidates = defaultdict(list)
        for plugin in plugins:
            for extension in plugin.file_extensions:
                candidates[extension].append((plugin.auto_priority, plugin.name, plugin.enabled))
        for name, connector in self._builtins.items():
            for extension in getattr(connector, "file_extensions", ()):
                candidates[extension.casefold()].append((
                    getattr(connector, "auto_priority", 0), name,
                    self.enabled is None or name in self.enabled,
                ))
        routes = {}
        for extension, choices in candidates.items():
            valid = [(priority, name) for priority, name, enabled in choices
                     if enabled and counts[name] == 1]
            if valid:
                _, name = min(valid, key=lambda choice: (-choice[0], choice[1]))
                routes[extension] = (name, "")
            elif any(enabled for _, _, enabled in choices):
                routes[extension] = (None, "CONNECTOR_DUPLICATE")
            else:
                routes[extension] = (None, "CONNECTOR_DISABLED")
        return routes

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
        if info.contract_version != "1":
            return ConnectorResult(
                issues=[ConnectorIssue("CONNECTOR_VERSION_ERROR", "仅支持契约版本1")]
            )
        error = dependency_error(info)
        if error:
            return ConnectorResult(issues=[ConnectorIssue(
                "CONNECTOR_" + error, f"{info.name} 需要主程序环境依赖: {', '.join(info.requires)}",
                config.path,
            )])
        try:
            raw = load_entry(info)(config.as_dict(), copy.deepcopy(self.context))
            return self._normalize(raw)
        except ConnectorError as exc:
            return ConnectorResult(issues=[ConnectorIssue(exc.code, exc.message, exc.source)])
        except Exception as exc:  # 扩展失败不能拖垮核心
            return ConnectorResult(
                issues=[
                    ConnectorIssue(
                        "CONNECTOR_ERROR",
                        f"插件 {info.name} 执行失败: {type(exc).__name__}",
                        config.path,
                    )
                ]
            )

    @staticmethod
    def _normalize(raw: Any) -> ConnectorResult:
        if isinstance(raw, ConnectorResult):
            if any(not isinstance(d, Document) for d in raw.documents):
                raise ValueError("Connector documents必须是公开Document")
            return raw
        if isinstance(raw, list):
            return ConnectorRegistry._normalize(ConnectorResult(documents=raw))
        if isinstance(raw, dict) and "documents" in raw:
            issues = [
                item if isinstance(item, ConnectorIssue) else ConnectorIssue(**item)
                for item in raw.get("issues", [])
            ]
            return ConnectorRegistry._normalize(
                ConnectorResult(documents=list(raw["documents"]), issues=issues)
            )
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
        all_plugins = self.plugin_infos()
        plugin_matches = [i for i in all_plugins if i.name == name]
        if len(plugin_matches) > 1 or (plugin_matches and name in self._builtins):
            return ConnectorResult(issues=[ConnectorIssue("CONNECTOR_DUPLICATE", "Plugin name is not unique", config.path)])
        connector = self._builtins.get(name)
        if connector is not None:
            if self.enabled is not None and name not in self.enabled:
                return ConnectorResult(
                    issues=[ConnectorIssue("CONNECTOR_DISABLED", f"{name} 已禁用")]
                )
            try:
                return connector.load(config)
            except ConnectorError as exc:
                return ConnectorResult(issues=[ConnectorIssue(exc.code, exc.message, exc.source)])
            except Exception as exc:
                return ConnectorResult(
                    issues=[
                        ConnectorIssue("CONNECTOR_ERROR", f"{name} 执行失败: {type(exc).__name__}", config.path)
                    ]
                )

        info = plugin_matches[0] if plugin_matches else None
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
