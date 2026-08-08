"""
Master Perception & Tracking Pipeline
=====================================
Asynchronous zero-copy pipeline orchestrator tying together:
  Camera Ingestion -> Adaptive Scheduler -> Detector -> Adaptive Tracker -> Geolocation -> TargetDB -> Gimbal
"""

from __future__ import annotations

import time
from typing import Optional

import structlog

from apex.engine.contracts.detection import DetectionArray
from apex.engine.contracts.frame import Frame
from apex.engine.contracts.track import Track, TrackArray
from apex.engine.db.target_database import TargetDatabase
from apex.engine.detector.detector_base import DetectorBase
from apex.engine.events.event_engine import EventEngine
from apex.engine.hal.hw_profile import HWProfile
from apex.engine.mission.gimbal_controller import GimbalPIDController
from apex.engine.mission.mission_manager import MissionManager
from apex.engine.scheduler.adaptive_scheduler import AdaptiveScheduler
from apex.engine.spatial.geolocation import WorldCoordinateSystem
from apex.engine.telemetry.mavlink_receiver import MAVLinkReceiver
from apex.engine.tracker.adaptive_tracker import AdaptiveTracker
from plugins.detectors.rtdetr.plugin import RTDETRPlugin
from plugins.trackers.bytetrack.plugin import ByteTrackPlugin

log = structlog.get_logger(__name__)


class MasterPipeline:
    """End-to-end perception and tracking pipeline engine."""

    def __init__(
        self,
        detector: DetectorBase | None = None,
        tracker: AdaptiveTracker | None = None,
        hw_profile: HWProfile | None = None,
    ) -> None:
        self.detector = detector or RTDETRPlugin()
        self.tracker = tracker or AdaptiveTracker()
        self.scheduler = AdaptiveScheduler()
        self.target_db = TargetDatabase()
        self.wcs = WorldCoordinateSystem()
        self.telemetry_rx = MAVLinkReceiver()
        self.mission_mgr = MissionManager()
        self.gimbal = GimbalPIDController()

        self._frame_count = 0
        self._total_pipeline_time_ms = 0.0

    async def initialize(self, config: dict, hw_profile: HWProfile) -> None:
        """Initialize all underlying pipeline components."""
        await self.detector.load(config.get("detector", {}), hw_profile)
        await self.tracker.initialize(config.get("tracker", {}), hw_profile)
        await self.telemetry_rx.start()
        log.info("master_pipeline_initialized")

    async def process_frame(self, frame: Frame) -> TrackArray:
        """
        Execute full pipeline cycle for a single frame.
        """
        t0 = time.perf_counter()
        self._frame_count += 1

        # 1. Fetch latest flight telemetry
        telemetry = self.telemetry_rx.get_latest_telemetry()

        # 2. Check Adaptive Scheduler: Neural Detection vs Kalman Coasting
        should_detect = self.scheduler.should_detect(frame)

        detections = []
        if should_detect:
            detections = self.detector.detect(frame)

        # 3. Target Tracking Update
        tracks = self.tracker.update(detections, frame, is_maneuvering=False)

        # 4. World Coordinates Geolocation
        updated_tracks: list[Track] = []
        for tr in tracks:
            lat, lon, alt = self.wcs.pixel_to_world(
                pixel_x=tr.bbox.cx,
                pixel_y=tr.bbox.cy,
                image_w=frame.metadata.width,
                image_h=frame.metadata.height,
                uav_lat=telemetry.lat,
                uav_lon=telemetry.lon,
                uav_alt_m=telemetry.alt_m,
                gimbal_pitch_deg=telemetry.pitch_deg,
                gimbal_yaw_deg=telemetry.yaw_deg,
            )
            # Compute real-time RTOS velocity and speed from Kalman filter state
            vx, vy = tr.velocity_px
            speed_px_s = (vx**2 + vy**2) ** 0.5
            # Static target threshold: if pixel velocity < 2.5 px/s, target is static (0.0 km/h)
            if speed_px_s < 2.5:
                calc_speed = 0.0
            else:
                calc_speed = round(float(speed_px_s * 0.18), 1)

            # Re-instantiate Track with populated world_point and dynamic real-time speed
            new_tr = Track(
                track_id=tr.track_id,
                state=tr.state,
                bbox=tr.bbox,
                predicted_bbox=tr.predicted_bbox,
                confidence=tr.confidence,
                class_id=tr.class_id,
                class_name=tr.class_name,
                frame_timestamp=tr.frame_timestamp,
                camera_id=tr.camera_id,
                velocity_px=tr.velocity_px,
                world_point=(lat, lon, alt),
                speed_kmh=calc_speed,
                age_frames=tr.age_frames,
                hits=tr.hits,
                misses=tr.misses,
            )
            updated_tracks.append(new_tr)


        # 5. Target Database Update
        self.target_db.update_tracks(updated_tracks)

        # 6. Gimbal Lead-Angle Control calculation for locked target
        locked_target = self.mission_mgr.get_locked_target(updated_tracks)
        if locked_target is not None:
            pan, tilt, _ = self.gimbal.compute_slew_rates(
                locked_target,
                image_w=frame.metadata.width,
                image_h=frame.metadata.height,
            )
            log.debug("gimbal_slew_command", pan=pan, tilt=tilt)

        pipeline_lat_ms = (time.perf_counter() - t0) * 1000.0
        self._total_pipeline_time_ms += pipeline_lat_ms

        return TrackArray(
            tracks=tuple(updated_tracks),
            frame_timestamp=frame.timestamp,
            camera_id=frame.metadata.camera_id,
            tracker_id="master_pipeline",
            tracking_latency_ms=pipeline_lat_ms,
        )

    @property
    def avg_latency_ms(self) -> float:
        return self._total_pipeline_time_ms / max(1, self._frame_count)
