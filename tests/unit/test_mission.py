"""
Unit Tests — Mission Layer (Phase 8)
"""

import pytest

from apex.engine.contracts.detection import BoundingBox
from apex.engine.contracts.track import Track, TrackState
from apex.engine.mission.gimbal_controller import GimbalPIDController
from apex.engine.mission.mission_manager import MissionManager
from apex.engine.mission.mission_profile import MissionProfile


class TestMissionProfile:

    def test_load_mission_profile_yaml(self):
        profile = MissionProfile.load_from_yaml("configs/missions/road_vehicles.yaml")
        assert profile.name == "road_vehicles"
        assert profile.detector_plugin == "rtdetr"
        assert profile.tracker_plugin == "bytetrack"
        assert "car" in profile.target_classes


class TestMissionManager:

    def test_mission_manager_lifecycle(self):
        mm = MissionManager()
        profile = mm.load_mission_profile("configs/missions/drone_tracking.yaml")

        assert profile.name == "drone_tracking"
        assert mm.active_profile.speed_mode_kmh == 120

        # Acquire lock
        assert mm.acquire_target_lock(track_id=42)
        assert mm.locked_track_id == 42

        # Check locked target matching active tracks list
        box = BoundingBox(100, 100, 200, 200)
        track = Track(
            track_id=42,
            state=TrackState.CONFIRMED,
            bbox=box,
            predicted_bbox=box,
            confidence=0.9,
            class_id=0,
            class_name="drone",
            frame_timestamp=100.0,
        )

        locked = mm.get_locked_target([track])
        assert locked is not None
        assert locked.track_id == 42

        # Release lock
        mm.release_target_lock()
        assert mm.locked_track_id is None


class TestGimbalPIDController:

    def test_gimbal_pid_slew_rates(self):
        gimbal = GimbalPIDController()
        box = BoundingBox(400, 300, 500, 400)  # Offset to right and down from center (320, 240)
        track = Track(
            track_id=1,
            state=TrackState.CONFIRMED,
            bbox=box,
            predicted_bbox=box,
            confidence=0.9,
            class_id=0,
            class_name="vehicle",
            frame_timestamp=100.0,
        )

        pan, tilt, dist = gimbal.compute_slew_rates(track, image_w=640, image_h=480, use_predictive_lead=True)
        assert pan > 0.0   # Pan right towards target
        assert tilt < 0.0  # Tilt down towards target
