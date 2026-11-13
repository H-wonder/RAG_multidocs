"""Project settings loaded from JSON + environment variables."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

DEFAULT_CONNECTORS = ["txt", "pdf", "connector.markdown"]


@dataclass
class Settings:
    data_dir: str = "data"
    connectors_dir: str = "connectors"
    enabled_connectors: list[str] = field(
        default_factory=lambda: list(DEFAULT_CONNECTORS)
    )
    docparse_repo: str = ""
    docparse_python: str = ""
    docparse_mode: str = "auto"
    docparse_timeout_s: int = 1800

    @property
    def index_file(self) -> Path:
        return Path(self.data_dir) / "index.json"

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Settings":
        config_path = Path(path or os.environ.get("RAG_CONFIG", "config.json"))
        data: dict = {}
        if config_path.exists():
            data = json.loads(config_path.read_text(encoding="utf-8"))

        allowed = {item.name for item in fields(cls)}
        values = {key: value for key, value in data.items() if key in allowed}
        settings = cls(**values)

        settings.data_dir = os.environ.get("RAG_DATA_DIR", settings.data_dir)
        settings.connectors_dir = os.environ.get(
            "RAG_CONNECTORS_DIR", settings.connectors_dir
        )
        settings.docparse_repo = os.environ.get(
            "DOCPARSE_REPO", settings.docparse_repo
        )
        settings.docparse_python = os.environ.get(
            "DOCPARSE_PYTHON", settings.docparse_python
        )
        settings.docparse_mode = os.environ.get(
            "DOCPARSE_MODE", settings.docparse_mode
        )
        timeout = os.environ.get("DOCPARSE_TIMEOUT_S")
        if timeout:
            settings.docparse_timeout_s = int(timeout)
        enabled = os.environ.get("RAG_ENABLED_CONNECTORS")
        if enabled:
            settings.enabled_connectors = [
                item.strip() for item in enabled.split(",") if item.strip()
            ]
        return settings

    def save(self, path: str | Path | None = None) -> Path:
        config_path = Path(path or os.environ.get("RAG_CONFIG", "config.json"))
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return config_path

    def to_dict(self) -> dict:
        return asdict(self)


__all__ = ["DEFAULT_CONNECTORS", "Settings"]
