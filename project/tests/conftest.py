import importlib.util
import sys
from pathlib import Path

import pytest

from rag_multidocs.cli import build_runtime
from rag_multidocs.config import Settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def settings(tmp_path, project_root) -> Settings:
    return Settings(
        data_dir=str(tmp_path / "data"),
        connectors_dir=str(project_root / "connectors"),
        index_backend="keyword",
    )


@pytest.fixture
def runtime(settings):
    return build_runtime(settings)


@pytest.fixture
def load_plugin(project_root):
    """Load a connector adapter by path, for adapter-level contract tests."""

    def _load(plugin: str, filename: str = "adapter.py"):
        path = project_root / "connectors" / plugin / filename
        spec = importlib.util.spec_from_file_location(
            f"rag_multidocs_test_plugin_{plugin.replace('.', '_')}", path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load plugin: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(spec.name, None)
        return module

    return _load
