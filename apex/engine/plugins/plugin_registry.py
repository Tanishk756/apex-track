"""
Plugin Registry
===============
Runtime registry of all loaded plugin instances.
Provides lookup by name, type, and capability.
Emits PLUGIN_LOADED / PLUGIN_UNLOADED events to the EventEngine.
"""

from __future__ import annotations

import time
from typing import Optional, Type

import structlog

from apex.engine.plugins.plugin_base import PluginBase, PluginHealth, PluginType

log = structlog.get_logger(__name__)


class PluginRegistry:
    """
    Singleton registry of all active plugin instances.

    Usage::
        registry = PluginRegistry.instance()
        registry.register("bytetrack", plugin)
        tracker = registry.get("bytetrack")
        trackers = registry.by_type(PluginType.TRACKER)
    """

    _instance: "PluginRegistry | None" = None

    def __init__(self) -> None:
        self._plugins: dict[str, PluginBase] = {}
        self._registered_at: dict[str, float] = {}
        self._event_engine = None   # injected after EventEngine init

    @classmethod
    def instance(cls) -> "PluginRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    def set_event_engine(self, engine) -> None:
        self._event_engine = engine

    # ── Registration ─────────────────────────────────────────────────────────

    def register(self, name: str, plugin: PluginBase) -> None:
        if name in self._plugins:
            log.warning("plugin_already_registered", name=name)
        self._plugins[name] = plugin
        self._registered_at[name] = time.time()
        log.info("plugin_registered", name=name, type=plugin.metadata.plugin_type.name)
        if self._event_engine:
            from apex.engine.contracts.event import ApexEvent, EventType
            self._event_engine.emit_sync(
                ApexEvent(EventType.PLUGIN_LOADED, source="plugin_registry",
                          payload={"name": name})
            )

    async def unregister(self, name: str) -> bool:
        plugin = self._plugins.pop(name, None)
        if plugin is None:
            return False
        self._registered_at.pop(name, None)
        await plugin.unload()
        log.info("plugin_unregistered", name=name)
        if self._event_engine:
            from apex.engine.contracts.event import ApexEvent, EventType
            self._event_engine.emit_sync(
                ApexEvent(EventType.PLUGIN_UNLOADED, source="plugin_registry",
                          payload={"name": name})
            )
        return True

    # ── Lookup ────────────────────────────────────────────────────────────────

    def get(self, name: str) -> Optional[PluginBase]:
        return self._plugins.get(name)

    def require(self, name: str) -> PluginBase:
        """Like get() but raises if not found."""
        p = self._plugins.get(name)
        if p is None:
            raise KeyError(f"Plugin '{name}' is not registered. Load it first.")
        return p

    def by_type(self, plugin_type: PluginType) -> list[PluginBase]:
        return [p for p in self._plugins.values()
                if p.metadata.plugin_type == plugin_type]

    def names(self) -> list[str]:
        return list(self._plugins.keys())

    # ── Health ────────────────────────────────────────────────────────────────

    def health_report(self) -> dict[str, PluginHealth]:
        return {name: plugin.health() for name, plugin in self._plugins.items()}

    def all_healthy(self) -> bool:
        return all(p.health().is_healthy for p in self._plugins.values())

    def __len__(self) -> int:
        return len(self._plugins)

    def __repr__(self) -> str:
        return f"PluginRegistry(loaded={list(self._plugins.keys())})"
