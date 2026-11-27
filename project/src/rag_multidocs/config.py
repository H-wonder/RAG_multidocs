"""Project settings loaded from JSON + environment variables."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

DEFAULT_CONNECTORS = [
    "txt", "pdf", "connector.markdown", "connector.html",
    "connector.jsonl", "connector.image", "connector.word", "connector.excel",
]


def _docparse_interpreter() -> str:
    """Find the existing docparse environment beside the active interpreter."""
    prefixes = [Path(sys.prefix), Path(os.environ.get("CONDA_PREFIX") or sys.prefix)]
    env_dirs = [p.parent for p in prefixes] + [p / "envs" for p in prefixes]
    if conda := os.environ.get("CONDA_EXE"):
        env_dirs.append(Path(conda).parent.parent / "envs")
    env_dirs.extend(Path(p).expanduser() for p in os.environ.get("CONDA_ENVS_PATH", "").split(os.pathsep) if p)
    env_dirs.append(Path.home() / ".conda" / "envs")
    for env_dir in dict.fromkeys(env_dirs):
        for relative in ("bin/python", "python.exe"):
            candidate = env_dir / "docparse" / relative
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate.absolute())
    return sys.executable


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

    # embedding provider: local | hashing | dashscope | openai | ollama
    embed_backend: str = "local"
    embed_model: str = ""          # 空 -> provider 默认模型
    embed_dim: int = 0             # 0 -> provider 默认维度
    embed_base_url: str = ""       # 空 -> provider 默认地址

    # chat provider: 空表示不启用生成式问答
    chat_provider: str = ""
    chat_model: str = ""           # 空 -> provider 默认模型
    chat_base_url: str = ""        # 空 -> provider 默认地址
    chat_temperature: float = 0.2
    chat_max_tokens: int = 0       # 0 -> provider/model default
    chat_context_tokens: int = 0   # 0 -> provider/model window
    rag_policy: str = "stable"
    rag_top_k: int = 5

    # docparse
    docparse_repo: str = ""
    docparse_python: str = ""
    docparse_mode: str = "auto"
    docparse_timeout_s: int = 1800

    # local API-key store (0600, gitignored); env vars take precedence
    credentials_file: str = ".rag/credentials.json"

    @property
    def index_file(self) -> Path:
        return Path(self.data_dir) / "index.json"

    def resolved_connector_settings(self, base_dir: str | Path | None = None) -> dict:
        """Resolve only connector paths. Index and credential location semantics stay intact."""
        base = Path(getattr(self, "_config_dir", base_dir or Path.cwd())).resolve()
        source_project = Path(__file__).resolve().parents[2]
        roots = (base, *base.parents, source_project, source_project.parent)
        resolved = asdict(self)
        plugins = base / Path(self.connectors_dir).expanduser()
        if self.connectors_dir == "connectors" and not plugins.is_dir() and "RAG_CONNECTORS_DIR" not in os.environ:
            plugins = next(
                (path for root in roots for path in (root / "connectors", root / "project/connectors")
                 if path.is_dir() and any(path.glob("*/manifest.json"))),
                plugins,
            )
        resolved["connectors_dir"] = str(plugins.resolve())
        if self.docparse_repo:
            resolved["docparse_repo"] = str((base / Path(self.docparse_repo).expanduser()).resolve())
        elif hasattr(self, "_config_dir"):
            repo = next(
                (root / "已有资源/可选组件/docparse" for root in roots
                 if (root / "已有资源/可选组件/docparse/docparse/__main__.py").is_file()),
                None,
            )
            resolved["docparse_repo"] = str(repo) if repo else ""
        python = self.docparse_python
        if python and ("/" in python or "\\" in python or python.startswith("~")):
            python = str((base / Path(python).expanduser()).absolute())
        resolved["docparse_python"] = python or (
            _docparse_interpreter() if hasattr(self, "_config_dir") else ""
        )
        return resolved

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Settings":
        config_path = Path(path or os.environ.get("RAG_CONFIG", "config.json")).expanduser()
        data: dict = {}
        if config_path.exists():
            data = json.loads(config_path.read_text(encoding="utf-8"))

        allowed = {item.name for item in fields(cls)}
        values = {key: value for key, value in data.items() if key in allowed}
        if not values.get("docparse_python") and data.get("parser_python"):
            values["docparse_python"] = data["parser_python"]
        settings = cls(**values)
        settings._config_dir = config_path.absolute().parent

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
        env_embed_backend = os.environ.get("EMBED_BACKEND")
        if env_embed_backend and env_embed_backend != settings.embed_backend:
            # 换 provider 时不要沿用旧模型的维度和模型名
            settings.embed_backend = env_embed_backend
            settings.embed_model = ""
            settings.embed_dim = 0
        elif env_embed_backend:
            settings.embed_backend = env_embed_backend
        settings.embed_model = os.environ.get("EMBED_MODEL", settings.embed_model)
        if os.environ.get("EMBED_DIM"):
            settings.embed_dim = int(os.environ["EMBED_DIM"])
        settings.embed_base_url = os.environ.get(
            "EMBED_BASE_URL", settings.embed_base_url
        )
        settings.chat_provider = os.environ.get(
            "CHAT_PROVIDER", settings.chat_provider
        )
        settings.chat_model = os.environ.get("CHAT_MODEL", settings.chat_model)
        settings.chat_base_url = os.environ.get(
            "CHAT_BASE_URL", settings.chat_base_url
        )
        if os.environ.get("CHAT_TEMPERATURE"):
            settings.chat_temperature = float(os.environ["CHAT_TEMPERATURE"])
        if os.environ.get("CHAT_MAX_TOKENS"):
            settings.chat_max_tokens = int(os.environ["CHAT_MAX_TOKENS"])
        if os.environ.get("CHAT_CONTEXT_TOKENS"):
            settings.chat_context_tokens = int(os.environ["CHAT_CONTEXT_TOKENS"])
        settings.rag_policy = os.environ.get("RAG_POLICY", settings.rag_policy)
        if os.environ.get("RAG_TOP_K"):
            settings.rag_top_k = int(os.environ["RAG_TOP_K"])
        settings.docparse_repo = os.environ.get(
            "DOCPARSE_REPO", settings.docparse_repo
        )
        settings.docparse_python = os.environ.get(
            "DOCPARSE_PYTHON", settings.docparse_python
        )
        settings.docparse_mode = os.environ.get(
            "DOCPARSE_MODE", settings.docparse_mode
        )
        settings.credentials_file = os.environ.get(
            "RAG_CREDENTIALS", settings.credentials_file
        )
        timeout = os.environ.get("DOCPARSE_TIMEOUT_S")
        if timeout:
            settings.docparse_timeout_s = int(timeout)
        enabled = os.environ.get("RAG_ENABLED_CONNECTORS")
        if enabled is not None:
            settings.enabled_connectors = [
                item.strip() for item in enabled.split(",") if item.strip()
            ]
        return settings

    def save(self, path: str | Path | None = None) -> Path:
        config_path = Path(path or os.environ.get("RAG_CONFIG", "config.json")).expanduser()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return config_path

    def to_dict(self) -> dict:
        return asdict(self)


__all__ = ["DEFAULT_CONNECTORS", "Settings"]
