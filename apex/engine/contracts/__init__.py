"""
Phase 0 — Core Contracts
========================
All shared dataclasses and interfaces that every module exchanges.
These are the single source of truth for inter-module communication.
Nothing in apex.engine imports from sibling modules — only from contracts.

Design rules:
- All dataclasses are frozen=True (immutable) to be safe across threads
- All fields have explicit types — no Any
- All dataclasses are slots=True for memory efficiency
- World coordinates and pixel coordinates are always kept separate
- Timestamps are always float (seconds since epoch, UTC)
"""

from apex.engine.contracts.frame import Frame, FrameMetadata
from apex.engine.contracts.detection import Detection, DetectionArray, BoundingBox
from apex.engine.contracts.track import Track, TrackArray, TrackState
from apex.engine.contracts.event import ApexEvent, EventType
from apex.engine.contracts.telemetry import Telemetry, Attitude, GPSPosition
from apex.engine.contracts.command import Command, CommandType, GimbalCommand

__all__ = [
    "Frame", "FrameMetadata",
    "Detection", "DetectionArray", "BoundingBox",
    "Track", "TrackArray", "TrackState",
    "ApexEvent", "EventType",
    "Telemetry", "Attitude", "GPSPosition",
    "Command", "CommandType", "GimbalCommand",
]
