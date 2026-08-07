"""
Tracker Base Class
==================
Abstract base class for object tracking plugins.

Design:
- Inherits PluginBase to integrate with PluginLoader and PluginRegistry.
- Converts Detection contracts into Track contracts with persistent IDs.
- Manages track lifecycle state transitions (TENTATIVE -> CONFIRMED -> COASTING -> LOST -> DELETED).
"""

from __future__ import annotations

import abc
import time
from typing import Optional

import structlog

from apex.engine.contracts.detection import Detection
from apex.engine.contracts.frame import Frame
from apex.engine.contracts.track import Track, TrackState
from apex.engine.hal.hw_profile import HWProfile
from apex.engine.plugins.plugin_base import PluginBase, PluginStatus

log = structlog.get_logger(__name__)


class TrackerBase(PluginBase, abc.ABC):
    """Abstract base class for multi-target tracking algorithms."""

    def __init__(self) -> None:
        super().__init__()
        self.track_high_thresh: float = 0.5
        self.track_low_thresh: float = 0.1
        self.new_track_thresh: float = 0.6
        self.match_thresh: float = 0.8
        self.max_time_lost: int = 30  # Frames to coast before deleting
        self.min_hits: int = 3        # Frames required to confirm track
        self._next_track_id: int = 1

    async def load(self, config: dict, hw_profile: HWProfile) -> None:
        """Initialize tracker parameters."""
        self.config = config
        self.hw_profile = hw_profile

        self.track_high_thresh = float(config.get("track_high_thresh", self.track_high_thresh))
        self.track_low_thresh = float(config.get("track_low_thresh", self.track_low_thresh))
        self.new_track_thresh = float(config.get("new_track_thresh", self.new_track_thresh))
        self.match_thresh = float(config.get("match_thresh", self.match_thresh))
        self.max_time_lost = int(config.get("max_time_lost", self.max_time_lost))
        self.min_hits = int(config.get("min_hits", self.min_hits))

        self._set_status(PluginStatus.ACTIVE)

    async def unload(self) -> None:
        """Unload tracker resources."""
        self._set_status(PluginStatus.UNLOADED)

    @abc.abstractmethod
    def update(self, detections: list[Detection], frame: Frame) -> list[Track]:
        """
        Associate new detections with existing tracks and update state machine.
        Returns list of active tracks for current frame.
        """

    def _get_next_id(self) -> int:
        """Generates monotonically increasing track ID."""
        tid = self._next_track_id
        self._next_track_id += 1
        return tid
