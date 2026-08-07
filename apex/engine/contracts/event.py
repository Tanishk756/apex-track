"""
Contract: Event
===============
System events emitted by any module and routed through the EventEngine.

Design decisions:
- EventType is a single flat enum — all modules share one event namespace.
  This prevents collisions and makes log analysis trivial.
- ApexEvent carries an optional payload dict — typed enough for consumers,
  flexible enough not to require a new class per event type.
- severity follows standard log levels so the health monitor can filter.
- source identifies the emitting module for debugging.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class EventType(Enum):
    # ── Camera / Video ───────────────────────────────────────────────────────
    CAMERA_CONNECTED     = auto()
    CAMERA_DISCONNECTED  = auto()
    VIDEO_LOST           = auto()
    VIDEO_RECOVERED      = auto()
    FRAME_DROP           = auto()

    # ── Detector ─────────────────────────────────────────────────────────────
    DETECTOR_LOADED      = auto()
    DETECTOR_UNLOADED    = auto()
    DETECTOR_ERROR       = auto()
    DETECTION_STARTED    = auto()   # scheduler enabled detection this frame

    # ── Tracker ──────────────────────────────────────────────────────────────
    TARGET_DETECTED      = auto()   # new CONFIRMED track appeared
    TARGET_LOST          = auto()   # track entered LOST state
    TARGET_REACQUIRED    = auto()   # LOST track recovered via Re-ID
    TRACK_COASTING       = auto()   # track entered COASTING (UKF only)
    TRACK_DELETED        = auto()   # track permanently removed
    LOW_CONFIDENCE       = auto()   # track confidence fell below threshold
    OCCLUDED             = auto()   # track inferred to be behind obstacle

    # ── Target Lock ──────────────────────────────────────────────────────────
    LOCK_ACQUIRED        = auto()
    LOCK_LOST            = auto()
    LOCK_SWITCHED        = auto()   # primary lock changed to different track

    # ── Mission ──────────────────────────────────────────────────────────────
    ZONE_ENTERED         = auto()
    ZONE_EXITED          = auto()
    MISSION_STARTED      = auto()
    MISSION_COMPLETE     = auto()
    MISSION_ABORTED      = auto()
    THREAT_ALERT         = auto()

    # ── Telemetry ────────────────────────────────────────────────────────────
    TELEMETRY_CONNECTED  = auto()
    TELEMETRY_LOST       = auto()
    TELEMETRY_RECOVERED  = auto()

    # ── Gimbal ───────────────────────────────────────────────────────────────
    GIMBAL_SLEW_START    = auto()
    GIMBAL_LOCKED        = auto()   # gimbal centred on target
    GIMBAL_ERROR         = auto()

    # ── System / Health ──────────────────────────────────────────────────────
    SYSTEM_STATE_CHANGED = auto()
    GPU_OVERLOADED       = auto()   # GPU util > threshold
    CPU_OVERLOADED       = auto()
    MEMORY_PRESSURE      = auto()
    PLUGIN_LOADED        = auto()
    PLUGIN_UNLOADED      = auto()
    PLUGIN_ERROR         = auto()
    RECORDING_STARTED    = auto()
    RECORDING_STOPPED    = auto()
    SHUTDOWN_REQUESTED   = auto()
    ERROR                = auto()   # generic fallback


class EventSeverity(Enum):
    DEBUG    = 10
    INFO     = 20
    WARNING  = 30
    ERROR    = 40
    CRITICAL = 50


@dataclass(frozen=True, slots=True)
class ApexEvent:
    """
    A system event published on the EventEngine.
    All fields are immutable so events can be safely passed across threads.
    """

    type: EventType
    source: str
    """Module/plugin that emitted this event (e.g. 'bytetrack', 'camera.usb.0')."""

    timestamp: float = field(default_factory=time.time)

    severity: EventSeverity = EventSeverity.INFO

    payload: dict[str, Any] = field(default_factory=dict)
    """Flexible key-value payload. Common keys documented per EventType."""

    def __repr__(self) -> str:
        return (
            f"ApexEvent({self.type.name}, src={self.source!r}, "
            f"sev={self.severity.name}, ts={self.timestamp:.3f})"
        )
