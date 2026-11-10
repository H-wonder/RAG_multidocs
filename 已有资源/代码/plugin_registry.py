"""通用插件注册表：manifest.json + adapter.py 的发现、校验与调用。

来源：``skills/registry.py``（任务一/二共用的技能层）。原实现服务于 Airflow
技能调度；这里去掉 Airflow 语义，抽象成课程项目可用的 Connector / Exporter
注册机制。

目录约定::

    插件根目录/
    ├── registry.json          # 可选；不写则自动发现含 manifest.json 的子目录
    └── <plugin_dir>/
        ├── manifest.json      # name/version/input_schema/entry/...
        └── adapter.py         # def run(payload: dict, context: dict | None) -> dict

manifest.json 示例::

    {
      "name": "connector.markdown",
      "version": "0.1.0",
      "description": "把 Markdown 目录解析为统一文档",
      "entry": "adapter:run",
      "input_schema": {
        "type": "object",
        "required": ["root"],
        "properties": {"root": {"type": "string"}}
      },
      "enabled": true
    }

核心只依赖这个公开契约：新增插件不需要改核心源码，只需增加目录并注册。
异常直接抛出，由调用方统一转换为“扩展失败”响应；不要在插件内吞掉错误。
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
from typing import Any


class PluginRegistry:
    def __init__(self, root: str | pathlib.Path):
        self.root = pathlib.Path(root)
        self.registry_file = self.root / "registry.json"

    # ---------------------------------------------------------------- 发现

    def _read_json(self, path: pathlib.Path) -> dict:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def _entries(self) -> list[dict]:
        if self.registry_file.exists():
            data = self._read_json(self.registry_file)
            return [entry for entry in data.get("plugins", []) if entry.get("enabled", True)]
        return [
            {"name": directory.name, "dir": directory.name}
            for directory in sorted(self.root.iterdir())
            if directory.is_dir() and (directory / "manifest.json").exists()
        ]

    def _find(self, name: str) -> tuple[pathlib.Path, dict]:
        for entry in self._entries():
            plugin_dir = self.root / entry.get("dir", entry["name"])
            manifest_path = plugin_dir / "manifest.json"
            if not manifest_path.exists():
                continue
            manifest = self._read_json(manifest_path)
            if manifest.get("name") == name:
                return plugin_dir, manifest
        available = ", ".join(item["name"] for item in self.list_plugins())
        raise KeyError(f"未知插件: {name}（可用: {available or '无'}）")

    def list_plugins(self) -> list[dict]:
        """返回插件目录摘要；不加载 adapter 实现。"""
        result: list[dict] = []
        for entry in self._entries():
            plugin_dir = self.root / entry.get("dir", entry["name"])
            manifest_path = plugin_dir / "manifest.json"
            if not manifest_path.exists():
                continue
            manifest = self._read_json(manifest_path)
            result.append(
                {
                    "name": manifest["name"],
                    "version": manifest.get("version"),
                    "description": manifest.get("description", ""),
                    "input_schema": manifest.get("input_schema", {}),
                    "output_schema": manifest.get("output_schema", {}),
                    "side_effects": manifest.get("side_effects", []),
                    "path": str(plugin_dir),
                }
            )
        return result

    # ---------------------------------------------------------------- 校验

    @staticmethod
    def _validate(payload: dict, schema: dict) -> None:
        if not schema:
            return
        try:
            import jsonschema
        except ImportError:
            # 极简兜底：至少检查必填字段，不因为没装 jsonschema 就不可用
            for key in schema.get("required", []):
                if key not in payload:
                    raise ValueError(f"缺少必填参数: {key}")
            return
        try:
            jsonschema.validate(payload, schema)
        except jsonschema.ValidationError as exc:
            raise ValueError(f"插件输入不符合 Schema: {exc.message}") from exc

    # ---------------------------------------------------------------- 调用

    def run(self, name: str, payload: dict | None = None,
            context: dict | None = None) -> Any:
        payload = payload or {}
        context = context or {}
        plugin_dir, manifest = self._find(name)
        self._validate(payload, manifest.get("input_schema", {}))

        entry = manifest.get("entry", "adapter:run")
        module_name, func_name = entry.split(":", 1)
        adapter_path = plugin_dir / f"{module_name}.py"
        if not adapter_path.exists():
            raise FileNotFoundError(f"插件 {name} 缺少 adapter: {adapter_path}")

        spec = importlib.util.spec_from_file_location(
            f"plugin_{name.replace('.', '_')}", adapter_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载插件 {name}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        func = getattr(module, func_name)
        return func(payload, context)


__all__ = ["PluginRegistry"]
