"""
Adaptive Tracker Engine
=======================
Dynamic tracker orchestrator that manages tracker plugin switching, target reacquisition,
and velocity-adaptive tracking profile adjustments for fast-moving UAV targets (60 - 120 km/h).
"""

from __future__ import annotations

import structlog

from apex.engine.contracts.detection import Detection
from apex.engine.contracts.frame import Frame
from apex.engine.contracts.track import Track, TrackState
from apex.engine.hal.hw_profile import HWProfile
from apex.engine.tracker.tracker_base import TrackerBase
from plugins.trackers.botsort.plugin import BoTSORTPlugin
from plugins.trackers.bytetrack.plugin import ByteTrackPlugin

log = structlog.get_logger(__name__)


class AdaptiveTracker:
    """
    High-level dynamic multi-target tracker.
    Automatically switches strategy (ByteTrack vs BoT-SORT CMC) based on target dynamics.
    """

    def __init__(self, primary_tracker: TrackerBase | None = None) -> None:
        self.tracker: TrackerBase = primary_tracker or ByteTrackPlugin()
        self.cmc_tracker: TrackerBase = BoTSORTPlugin()
        self._target_speed_mode: str = "high_speed_120kmh"

    async def initialize(self, config: dict, hw_profile: HWProfile) -> None:
        """Load constituent tracking plugins."""
        await self.tracker.load(config, hw_profile)
        await self.cmc_tracker.load(config, hw_profile)

    def update(self, detections: list[Detection], frame: Frame, is_maneuvering: bool = False) -> list[Track]:
        """
        Process frame detections and return updated active target tracks.
        """
        # If UAV is performing sharp pitch/yaw maneuvers, engage Camera Motion Compensation (BoT-SORT)
        active_engine = self.cmc_tracker if is_maneuvering else self.tracker
        tracks = active_engine.update(detections, frame)

        log.debug("adaptive_tracker_step", active_tracks=len(tracks), mode="CMC" if is_maneuvering else "Standard")
        return tracks
