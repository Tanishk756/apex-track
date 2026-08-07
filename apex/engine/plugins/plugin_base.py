"""
Plugin SDK — Base Classes
==========================
Every loadable component in APEX-Track is a Plugin.

Plugin types (all inherit PluginBase):
    DetectorPlugin   — wraps an inference backend
    TrackerPlugin    — wraps a tracking algorithm
    CameraPlugin     — wraps a video source
    TelemetryPlugin  — wraps a telemetry protocol
    GimbalPlugin     — wraps a gimbal/PTZ controller
    RecordingPlugin  — wraps a recording backend
    MissionPlugin    — wraps a mission policy

Plugin contract:
    1. Every plugin has a plugin.yaml manifest (see below)
    2. load(config, hw_profile) → None
    3. Plugin advertises its capabilities via metadata
    4. unload() cleans up all resources (no leaks)
    5. health() returns PluginHealth for the monitor

Manifest format (plugin.yaml):
    name: bytetrack
    version: "1.0.0"
    type: tracker
    author: "APEX-Track"
    license: MIT
    description: "ByteTrack multi-object tracker"
    entry_point: "plugins.trackers.bytetrack.ByteTrackPlugin"
    requires_capabilities: []   # Capability flag names
    tags: [tracking, realtime]
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional


class PluginType(Enum):
    DETECTOR   = auto()
    TRACKER    = auto()
    CAMERA     = auto()
    TELEMETRY  = auto()
    GIMBAL     = auto()
    RECORDING  = auto()
    MISSION    = auto()
    GENERIC    = auto()


class PluginStatus(Enum):
    UNLOADED  = auto()
    LOADING   = auto()
    ACTIVE    = auto()
    ERROR     = auto()
    UNLOADING = auto()


@dataclass
class PluginMetadata:
    """
    Immutable plugin identity — populated from plugin.yaml manifest.
    """
    name: str
    version: str
    plugin_type: PluginType
    license: str
    author: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)
    requires_capabilities: list[str] = field(default_factory=list)
    """List of Capability flag names required (e.g. ['CUDA', 'TENSORRT'])."""

    homepage: str = ""
    is_agpl: bool = False
    """True when license is AGPL-3.0 — triggers license acceptance gate."""


@dataclass
class PluginHealth:
    """Snapshot of plugin health, returned by plugin.health()."""
    status: PluginStatus
    is_healthy: bool
    uptime_s: float = 0.0
    error_count: int = 0
    last_error: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    """Plugin-specific performance metrics (e.g. fps, latency_ms)."""


class PluginBase(abc.ABC):
    """
    Abstract base for all APEX-Track plugins.

    Subclasses MUST implement:
        metadata  — class attribute of type PluginMetadata
        load()    — initialize resources, load models, open connections
        unload()  — release all resources cleanly

    Subclasses SHOULD implement:
        health()  — return current PluginHealth for monitoring
    """

    # Every concrete plugin class defines this at class level
    metadata: PluginMetadata

    def __init__(self) -> None:
        self._status = PluginStatus.UNLOADED
        self._loaded_at: Optional[float] = None
        self._error_count: int = 0
        self._last_error: str = ""

    @abc.abstractmethod
    async def load(self, config: dict, hw_profile: Any) -> None:
        """
        Initialize the plugin.
        Called by PluginLoader after license validation and capability check.
        Must be idempotent — calling twice without unload() should be safe.
        """

    @abc.abstractmethod
    async def unload(self) -> None:
        """
        Release all resources.
        Must complete within 5 seconds (enforced by RuntimeManager in Phase 8).
        """

    def health(self) -> PluginHealth:
        """Override to provide rich health metrics."""
        return PluginHealth(
            status=self._status,
            is_healthy=self._status == PluginStatus.ACTIVE,
            uptime_s=time.time() - self._loaded_at if self._loaded_at else 0.0,
            error_count=self._error_count,
            last_error=self._last_error,
        )

    def _set_status(self, status: PluginStatus) -> None:
        self._status = status
        if status == PluginStatus.ACTIVE:
            self._loaded_at = time.time()

    def _record_error(self, msg: str) -> None:
        self._error_count += 1
        self._last_error = msg
        self._status = PluginStatus.ERROR

    @property
    def is_active(self) -> bool:
        return self._status == PluginStatus.ACTIVE

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.metadata.name!r}, "
            f"status={self._status.name})"
        )
