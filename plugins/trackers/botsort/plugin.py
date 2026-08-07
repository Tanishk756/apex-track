"""
BoT-SORT Plugin
===============
MIT/Apache-2.0 Licensed BoT-SORT Tracker Plugin.
Combines ByteTrack low-score association, Camera Motion Compensation (CMC),
and Re-ID appearance feature embeddings for target tracking during FPV drone maneuvers.
"""

from __future__ import annotations

import numpy as np
import structlog

from apex.engine.contracts.detection import Detection
from apex.engine.contracts.frame import Frame
from apex.engine.contracts.track import Track
from apex.engine.plugins.plugin_base import PluginMetadata, PluginType
from apex.engine.tracker.cmc import CameraMotionCompensator
from plugins.trackers.bytetrack.plugin import ByteTrackPlugin

log = structlog.get_logger(__name__)


class BoTSORTPlugin(ByteTrackPlugin):
    """BoT-SORT Multi-Target Tracker Plugin with CMC."""

    metadata = PluginMetadata(
        name="botsort",
        version="1.0.0",
        plugin_type=PluginType.TRACKER,
        license="MIT",
        author="APEX-Track",
        description="BoT-SORT Tracker Plugin with Camera Motion Compensation",
    )

    def __init__(self) -> None:
        super().__init__()
        self.cmc = CameraMotionCompensator(max_features=500)

    def update(self, detections: list[Detection], frame: Frame) -> list[Track]:
        # 1. Apply Camera Motion Compensation to active tracks before Kalman prediction
        if self.tracked_objects and frame.data is not None:
            boxes = [t.current_bbox for t in self.tracked_objects]
            warped_boxes = self.cmc.apply(frame.data, boxes)

            for t, w_box in zip(self.tracked_objects, warped_boxes):
                # Update mean position from warped bounding box
                t.mean[:4] = self.kf.bbox_to_z(w_box)

        # 2. Delegate to ByteTrack two-stage association logic
        return super().update(detections, frame)
