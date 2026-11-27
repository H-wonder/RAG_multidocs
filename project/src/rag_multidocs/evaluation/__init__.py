"""Programmatic evaluation; no CLI/REPL/Web ownership."""

from .metrics import run_suite, write_report, ablation_policies

__all__ = ["run_suite", "write_report", "ablation_policies"]
