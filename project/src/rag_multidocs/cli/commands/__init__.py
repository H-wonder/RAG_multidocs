"""各子命令的实现。"""

from .ask import cmd_ask
from .connectors import cmd_connectors_list, cmd_connectors_toggle
from .docs import cmd_docs
from .ingest import cmd_ingest
from .keys import (
    cmd_keys_export,
    cmd_keys_list,
    cmd_keys_path,
    cmd_keys_set,
    cmd_keys_unset,
)
from .models import cmd_models_discover, cmd_models_list, cmd_models_use
from .search import cmd_search
from .stats import cmd_stats

__all__ = [
    "cmd_ask",
    "cmd_connectors_list",
    "cmd_connectors_toggle",
    "cmd_docs",
    "cmd_ingest",
    "cmd_keys_export",
    "cmd_keys_list",
    "cmd_keys_path",
    "cmd_keys_set",
    "cmd_keys_unset",
    "cmd_models_discover",
    "cmd_models_list",
    "cmd_models_use",
    "cmd_search",
    "cmd_stats",
]
