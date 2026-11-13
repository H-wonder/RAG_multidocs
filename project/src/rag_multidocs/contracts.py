"""Public Connector contract.

Core only depends on this module. Built-in and external connectors return the
same ``ConnectorResult``; adding a data source must not require changing core.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .model import Document


@dataclass(frozen=True)
class ConnectorConfig:
    name: str
    path: str
    options: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "path": self.path, "options": dict(self.options)}


@dataclass
class ConnectorIssue:
    code: str
    message: str
    source: str = ""

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message, "source": self.source}


@dataclass
class ConnectorResult:
    documents: list[Document] = field(default_factory=list)
    issues: list[ConnectorIssue] = field(default_factory=list)


class ConnectorError(RuntimeError):
    """Fatal connector error; the CLI converts it to a structured message."""

    def __init__(self, code: str, message: str, source: str = ""):
        self.code = code
        self.message = message
        self.source = source
        super().__init__(f"{code}: {message}")


class Connector(Protocol):
    name: str

    def load(self, config: ConnectorConfig) -> ConnectorResult:
        ...


__all__ = [
    "Connector",
    "ConnectorConfig",
    "ConnectorError",
    "ConnectorIssue",
    "ConnectorResult",
]
