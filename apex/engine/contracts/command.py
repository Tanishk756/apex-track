"""
Contract: Command
=================
Commands sent from MissionManager / GUI / API to actuators (gimbal, recorder, etc.).

Design decisions:
- CommandType covers all actuator targets. New actuators add enum values.
- GimbalCommand is a separate typed struct for the most performance-critical
  command path (high-frequency PTZ control loop).
- Commands carry an optional reply_channel so async callers can await ACK.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional


class CommandType(Enum):
    # Gimbal / PTZ
    GIMBAL_SET_ANGLE       = auto()  # absolute yaw/pitch/zoom
    GIMBAL_SET_RATE        = auto()  # velocity control
    GIMBAL_TRACK_TARGET    = auto()  # auto-track a track_id
    GIMBAL_RETURN_HOME     = auto()

    # Detector
    DETECTOR_LOAD          = auto()
    DETECTOR_UNLOAD        = auto()
    DETECTOR_SET_THRESHOLD = auto()

    # Mission
    MISSION_START          = auto()
    MISSION_STOP           = auto()
    MISSION_SET_PROFILE    = auto()

    # Target lock
    LOCK_TARGET            = auto()   # lock on specific track_id
    UNLOCK_TARGET          = auto()
    SET_LOCK_MODE          = auto()   # AUTO | MANUAL | NEAREST | LARGEST

    # Recording
    RECORDING_START        = auto()
    RECORDING_STOP         = auto()

    # System
    SHUTDOWN               = auto()
    RELOAD_CONFIG          = auto()


@dataclass(frozen=True, slots=True)
class GimbalCommand:
    """
    High-frequency gimbal/PTZ command — dedicated struct for the control loop.
    Published on channel /gimbal/commands at up to 50 Hz.
    """

    yaw_deg: float    = 0.0
    """Absolute yaw in degrees (NED frame, 0=North) or relative delta if is_relative=True."""

    pitch_deg: float  = 0.0
    """Absolute pitch in degrees (negative = nose down)."""

    zoom: float       = 1.0
    """Zoom factor (1.0 = optical 1×, >1 = zoom in)."""

    is_relative: bool = False
    """If True, yaw/pitch are rate commands (deg/s), not absolute angles."""

    track_id: Optional[int] = None
    """If set, the GimbalController uses the track's predicted_bbox to aim."""

    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class Command:
    """
    General-purpose command published on /mission/commands or /system/commands.
    """

    type: CommandType
    source: str
    """Originator: 'gui', 'api', 'mission_manager', 'operator', etc."""

    timestamp: float = field(default_factory=time.time)

    params: dict[str, Any] = field(default_factory=dict)
    """Type-specific parameters. See CommandType docstrings."""

    reply_channel: Optional[str] = None
    """If set, the handler publishes the result on this message bus channel."""

    def __repr__(self) -> str:
        return f"Command({self.type.name}, src={self.source!r})"
