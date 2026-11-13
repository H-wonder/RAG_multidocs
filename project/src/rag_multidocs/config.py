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

    # index backend: milvus (target) | keyword (offline fallback)
    index_backend: str = "milvus"
    fallback_to_keyword: bool = True

    # Milvus
    milvus_uri: str = "http://localhost:19530"
    milvus_db: str = "rag"
    milvus_collection: str = "rag_chunks"
    milvus_dense_weight: float = 0.7
    milvus_sparse_weight: float = 0.3

    # embedding: hashing (CI/smoke) | local (bge) | dashscope
    embed_backend: str = "local"
    embed_model: str = "BAAI/bge-small-zh-v1.5"
    embed_dim: int = 512

    # docparse
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
        settings.index_backend = os.environ.get(
            "RAG_INDEX_BACKEND", settings.index_backend
        )
        fallback = os.environ.get("RAG_FALLBACK_KEYWORD")
        if fallback is not None:
            settings.fallback_to_keyword = fallback.lower() not in ("0", "false", "no")
        settings.milvus_uri = os.environ.get("MILVUS_URI", settings.milvus_uri)
        settings.milvus_db = os.environ.get("MILVUS_DB", settings.milvus_db)
        settings.milvus_collection = os.environ.get(
            "MILVUS_COLLECTION", settings.milvus_collection
        )
        if os.environ.get("MILVUS_DENSE_WEIGHT"):
            settings.milvus_dense_weight = float(os.environ["MILVUS_DENSE_WEIGHT"])
        if os.environ.get("MILVUS_SPARSE_WEIGHT"):
            settings.milvus_sparse_weight = float(os.environ["MILVUS_SPARSE_WEIGHT"])
        settings.embed_backend = os.environ.get(
            "EMBED_BACKEND", settings.embed_backend
        )
        settings.embed_model = os.environ.get("EMBED_MODEL", settings.embed_model)
        if os.environ.get("EMBED_DIM"):
            settings.embed_dim = int(os.environ["EMBED_DIM"])
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
