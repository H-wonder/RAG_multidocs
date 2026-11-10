import json

import pytest

from plugin_registry import PluginRegistry


def _write_plugin(root, name="echo", payload_schema=None):
    plugin_dir = root / name
    plugin_dir.mkdir(parents=True)
    manifest = {
        "name": f"test.{name}",
        "version": "0.1.0",
        "description": "echo payload",
        "entry": "adapter:run",
        "input_schema": payload_schema or {"type": "object", "required": ["text"]},
    }
    (plugin_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    (plugin_dir / "adapter.py").write_text(
        "def run(payload, context=None):\n"
        "    return {'echo': payload['text'], 'context': context or {}}\n",
        encoding="utf-8",
    )
    return manifest["name"]


def test_auto_discovery_and_run(tmp_path):
    name = _write_plugin(tmp_path)
    registry = PluginRegistry(tmp_path)

    assert [item["name"] for item in registry.list_plugins()] == [name]
    assert registry.run(name, {"text": "hello"}, {"request": "r1"}) == {
        "echo": "hello",
        "context": {"request": "r1"},
    }


def test_missing_required_field_is_rejected(tmp_path):
    name = _write_plugin(tmp_path)
    registry = PluginRegistry(tmp_path)
    with pytest.raises(ValueError, match="缺少必填参数|不符合 Schema"):
        registry.run(name, {})


def test_unknown_plugin_lists_available(tmp_path):
    name = _write_plugin(tmp_path)
    registry = PluginRegistry(tmp_path)
    with pytest.raises(KeyError, match=name):
        registry.run("test.missing", {})


def test_registry_json_can_disable_plugin(tmp_path):
    name = _write_plugin(tmp_path)
    (tmp_path / "registry.json").write_text(
        json.dumps({"plugins": [{"name": "echo", "dir": "echo", "enabled": False}]}),
        encoding="utf-8",
    )
    registry = PluginRegistry(tmp_path)
    assert registry.list_plugins() == []
    with pytest.raises(KeyError):
        registry.run(name, {"text": "x"})
