"""CLI entrypoints.

The executable ``rag-cli`` still imports ``rag_multidocs.cli:main``.
"""

from .app import Runtime, build_registry, build_runtime, main

__all__ = ["Runtime", "build_registry", "build_runtime", "main"]
