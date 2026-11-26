"""readline 历史与终端准备。"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def prepare_terminal() -> None:
    """Some terminals advertise TERM=dumb, which disables readline editing."""
    if sys.stdout.isatty() and os.environ.get("TERM", "") in ("", "dumb"):
        os.environ["TERM"] = "xterm-256color"


prepare_terminal()
try:  # readline is stdlib on Linux/macOS; Windows users just lose history
    import readline
except ImportError:  # pragma: no cover - platform dependent
    readline = None


def setup_readline(history_path: Path) -> None:
    if readline is None:
        return
    try:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        readline.read_history_file(history_path)
    except OSError:
        pass
    readline.set_history_length(1000)


def save_readline(history_path: Path) -> None:
    if readline is None:
        return
    try:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        readline.write_history_file(history_path)
    except OSError:
        pass


__all__ = ["prepare_terminal", "readline", "save_readline", "setup_readline"]
