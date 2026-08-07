"""
Unit Tests — Tracking Engine (Phase 6)
"""

import pytest
import numpy as np

from apex.engine.contracts.detection import BoundingBox, Detection
from apex.engine.contracts.frame import Frame, FrameMetadata
from apex.engine.contracts.track import TrackState
from apex.engine.hal.hw_profile import HWCapabilities, HWProfile
from apex.engine.tracker.adaptive_tracker import AdaptiveTracker
from apex.engine.tracker.cmc import CameraMotionCompensator
from apex.engine.tracker.kalman import KalmanFilterTarget
from plugins.trackers.botsort.plugin import BoTSORTPlugin
from plugins.trackers.bytetrack.plugin import ByteTrackPlugin


@pytest.fixture
def sample_frame():
    data = np.zeros((480, 640, 3), dtype=np.uint8)
    meta = FrameMetadata(camera_id="cam0", width=640, height=480)
    return Frame(data=data, metadata=meta, timestamp=100.0, sequence_id=1)


class TestKalmanFilter:

    def test_kalman_initiate_and_predict(self):
        kf = KalmanFilterTarget()
        box = BoundingBox(100, 100, 200, 200)
        z = kf.bbox_to_z(box)
        mean, cov = kf.initiate(z)

        assert mean.shape == (8,)
        assert cov.shape == (8, 8)

        pred_mean, pred_cov = kf.predict(mean, cov)
        pred_box = kf.z_to_bbox(pred_mean)
        assert isinstance(pred_box, BoundingBox)


class TestCMC:

    def test_cmc_returns_boxes(self):
        cmc = CameraMotionCompensator(max_features=100)
        frame_data = np.zeros((480, 640, 3), dtype=np.uint8)
        boxes = [BoundingBox(50, 50, 150, 150)]

        warped = cmc.apply(frame_data, boxes)
        assert len(warped) == 1
        assert isinstance(warped[0], BoundingBox)


class TestByteTrack:

    @pytest.mark.asyncio
    async def test_bytetrack_tracking_lifecycle(self, sample_frame):
        tracker = ByteTrackPlugin()
        hw = HWProfile(capabilities=HWCapabilities(), profile_name="cpu_test")
        await tracker.load({"min_hits": 2}, hw)

        det = Detection(
            bbox=BoundingBox(100, 100, 200, 200),
            confidence=0.9,
            class_id=0,
            class_name="car",
            camera_id="cam0",
            detector_id="det0",
            frame_timestamp=100.0,
        )

        # Frame 1: Create tentative track
        tracks_f1 = tracker.update([det], sample_frame)
        assert len(tracks_f1) == 1
        assert tracks_f1[0].track_id == 1

        # Frame 2: Move slightly and update
        det2 = Detection(
            bbox=BoundingBox(105, 105, 205, 205),
            confidence=0.88,
            class_id=0,
            class_name="car",
            camera_id="cam0",
            detector_id="det0",
            frame_timestamp=100.033,
        )
        sample_frame.sequence_id = 2
        tracks_f2 = tracker.update([det2], sample_frame)

        assert len(tracks_f2) == 1
        # ID should remain preserved as 1!
        assert tracks_f2[0].track_id == 1


class TestBoTSORT:

    @pytest.mark.asyncio
    async def test_botsort_update(self, sample_frame):
        tracker = BoTSORTPlugin()
        hw = HWProfile(capabilities=HWCapabilities(), profile_name="cpu_test")
        await tracker.load({}, hw)

        det = Detection(
            bbox=BoundingBox(50, 50, 100, 100),
            confidence=0.85,
            class_id=0,
            class_name="vehicle",
            camera_id="cam0",
            detector_id="det0",
            frame_timestamp=100.0,
        )

        tracks = tracker.update([det], sample_frame)
        assert len(tracks) == 1


class TestAdaptiveTracker:

    @pytest.mark.asyncio
    async def test_adaptive_tracker_initialization_and_update(self, sample_frame):
        at = AdaptiveTracker()
        hw = HWProfile(capabilities=HWCapabilities(), profile_name="cpu_test")
        await at.initialize({}, hw)

        det = Detection(
            bbox=BoundingBox(10, 10, 80, 80),
            confidence=0.9,
            class_id=0,
            class_name="truck",
            camera_id="cam0",
            detector_id="det0",
            frame_timestamp=100.0,
        )

        tracks_std = at.update([det], sample_frame, is_maneuvering=False)
        assert len(tracks_std) == 1

        tracks_cmc = at.update([det], sample_frame, is_maneuvering=True)
        assert len(tracks_cmc) == 1
