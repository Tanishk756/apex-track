"""
Universal Dynamic Plugin Hub & Hot-Swapping Registry
====================================================
Manages plugin lifecycles, enabling dynamic registration, discovery,
and zero-downtime hot-swapping of detectors, trackers, and HAL components at runtime.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Dict, Any, Type, Optional, List
import structlog

from apex.engine.plugins.plugin_base import PluginBase, PluginType, PluginMetadata

log = structlog.get_logger(__name__)


class PluginHub:
    """Centralized Plug-and-Play Hub & Dynamic Plugin Registry."""

    def __init__(self) -> None:
        self._registry: Dict[str, Type[PluginBase]] = {}
        self._active_instances: Dict[str, PluginBase] = {}
        self._discover_plugins()

    def _discover_plugins(self) -> None:
        """Auto-discover built-in plugins from plugins package."""
        try:
            from plugins.detectors.rf_detr.plugin import RFDetr2XLPlugin
            from plugins.detectors.ensemble.plugin import EnsembleDetectorPlugin
            from apex.engine.detector.roboflow_plugin import RoboflowDetectorPlugin

            for cls in (RFDetr2XLPlugin, RoboflowDetectorPlugin, EnsembleDetectorPlugin):
                self.register_plugin(cls)
        except Exception as exc:
            log.warning("plugin_discovery_partial_notice", error=str(exc))

    def register_plugin(self, plugin_cls: Type[PluginBase]) -> None:
        """Registers a plugin class in the Hub registry."""
        meta = getattr(plugin_cls, "metadata", None)
        if meta and hasattr(meta, "name"):
            self._registry[meta.name] = plugin_cls
            log.info("plugin_registered", name=meta.name, type=meta.plugin_type.name)

    def get_plugin_class(self, name: str) -> Optional[Type[PluginBase]]:
        """Returns registered plugin class by name."""
        return self._registry.get(name, None)

    def list_registered_plugins(self) -> List[Dict[str, Any]]:
        """Returns catalog of all registered plugins."""
        catalog = []
        for name, cls in self._registry.items():
            meta: PluginMetadata = getattr(cls, "metadata", None)
            if meta:
                catalog.append({
                    "name": meta.name,
                    "version": meta.version,
                    "type": meta.plugin_type.name,
                    "description": meta.description,
                    "author": meta.author,
                })
        return catalog
