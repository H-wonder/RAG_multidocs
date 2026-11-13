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
