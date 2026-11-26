"""CLI entrypoints.

The executable ``rag-cli`` still imports ``rag_multidocs.cli:main``.
"""

from .main import main
from .runtime import Runtime, build_registry, build_runtime

__all__ = ["Runtime", "build_registry", "build_runtime", "main"]
