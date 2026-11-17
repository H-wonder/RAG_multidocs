"""Local API-key storage for the CLI.

Keys are stored in a project-local JSON file (default ``.rag/credentials.json``)
with ``0600`` permissions.  The file is never read by library code implicitly:
the CLI calls :func:`apply_keys` once at startup, which copies missing values
into ``os.environ`` for the current process only.

Precedence: an already-set environment variable always wins over the file.
Only ``rag-cli keys export`` prints plaintext values.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

DEFAULT_CREDENTIALS_FILE = ".rag/credentials.json"
CREDENTIALS_ENV = "RAG_CREDENTIALS"
_ENV_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")

# Names copied from the file by apply_keys(); used to report the key source.
_APPLIED_FROM_FILE: set[str] = set()


def credentials_path(configured: str = "") -> Path:
    """Resolve the credentials file: env override > settings > default."""
    raw = os.environ.get(CREDENTIALS_ENV) or configured or DEFAULT_CREDENTIALS_FILE
    return Path(raw).expanduser()


def load_keys(path: str | Path | None = None) -> dict[str, str]:
    """Read stored keys; a missing file means "no keys"."""
    file_path = Path(path) if path is not None else credentials_path()
    if not file_path.exists():
        return {}
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"凭证文件不是合法 JSON: {file_path}") from exc
    keys = payload.get("keys", {}) if isinstance(payload, dict) else {}
    if not isinstance(keys, dict):
        raise ValueError(f"凭证文件格式错误（keys 应为对象）: {file_path}")
    return {
        str(name): str(value)
        for name, value in keys.items()
        if _ENV_NAME_RE.match(str(name)) and str(value)
    }


def save_keys(keys: dict[str, str], path: str | Path | None = None) -> Path:
    """Atomically write keys with owner-only permissions."""
    file_path = Path(path) if path is not None else credentials_path()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    _chmod(file_path.parent, 0o700)

    payload = {"version": 1, "keys": {name: keys[name] for name in sorted(keys)}}
    fd, temp_name = tempfile.mkstemp(
        dir=file_path.parent, prefix=".credentials-", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _chmod(temp_name, 0o600)
        os.replace(temp_name, file_path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    _chmod(file_path, 0o600)
    return file_path


def set_key(env_name: str, value: str, path: str | Path | None = None) -> Path:
    name = env_name.strip().upper()
    if not _ENV_NAME_RE.match(name):
        raise ValueError(f"环境变量名不合法: {env_name}")
    value = value.strip()
    if not value:
        raise ValueError("API Key 不能为空")
    keys = load_keys(path)
    keys[name] = value
    saved = save_keys(keys, path)
    activate_key(name, value)
    return saved


def unset_key(env_name: str, path: str | Path | None = None) -> bool:
    name = env_name.strip().upper()
    keys = load_keys(path)
    existed = name in keys
    if existed:
        del keys[name]
        save_keys(keys, path)
    # Only remove the process variable if this process loaded it from the file.
    if name in _APPLIED_FROM_FILE:
        os.environ.pop(name, None)
        _APPLIED_FROM_FILE.discard(name)
    return existed


def activate_key(env_name: str, value: str) -> None:
    """Make a key usable immediately in this process."""
    os.environ[env_name] = value
    _APPLIED_FROM_FILE.add(env_name)


def apply_keys(path: str | Path | None = None) -> list[str]:
    """Copy stored keys into os.environ without overriding existing values."""
    global _APPLIED_FROM_FILE
    _APPLIED_FROM_FILE = set()
    applied: list[str] = []
    for name, value in load_keys(path).items():
        if not os.environ.get(name):
            os.environ[name] = value
            _APPLIED_FROM_FILE.add(name)
            applied.append(name)
    return applied


def source_of(env_name: str) -> str:
    """Return ``env``, ``file`` or ``""`` for where the value came from."""
    if env_name in _APPLIED_FROM_FILE:
        return "file"
    if os.environ.get(env_name):
        return "env"
    return ""


def mask(value: str) -> str:
    """Hide most of a secret while keeping it recognizable."""
    if not value:
        return ""
    if len(value) <= 8:
        return "****"
    return f"{value[:3]}...{value[-4:]}"


def permission_warning(path: str | Path | None = None) -> str:
    """Warn when the credentials file is readable by group/others."""
    if os.name != "posix":
        return ""
    file_path = Path(path) if path is not None else credentials_path()
    if not file_path.exists():
        return ""
    mode = file_path.stat().st_mode & 0o777
    if mode & 0o077:
        return f"凭证文件权限过宽（{oct(mode)}），建议执行: chmod 600 {file_path}"
    return ""


def _chmod(path: str | Path, mode: int) -> None:
    try:
        os.chmod(path, mode)
    except OSError:
        pass


__all__ = [
    "DEFAULT_CREDENTIALS_FILE",
    "activate_key",
    "apply_keys",
    "credentials_path",
    "load_keys",
    "mask",
    "permission_warning",
    "save_keys",
    "set_key",
    "source_of",
    "unset_key",
]
