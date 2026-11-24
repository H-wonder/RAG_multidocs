"""Built-in connectors and the plugin registry."""

from .registry import ConnectorRegistry, PluginInfo
from .txt import TxtConnector

__all__ = ["ConnectorRegistry", "PluginInfo", "TxtConnector"]
