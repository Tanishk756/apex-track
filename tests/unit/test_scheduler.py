"""
Unit Tests — Adaptive Scheduler (Phase 5)
"""

import pytest
import numpy as np

from apex.engine.contracts.frame import Frame, FrameMetadata
from apex.engine.scheduler.adaptive_scheduler import AdaptiveScheduler


@pytest.fixture
def dummy_frame():
    data = np.zeros((120, 160, 3), dtype=np.uint8)
    meta = FrameMetadata(camera_id="cam0", width=160, height=120)
    return Frame(data=data, metadata=meta, timestamp=100.0, sequence_id=1)


class TestAdaptiveScheduler:

    def test_base_interval_trigger(self, dummy_frame):
        scheduler = AdaptiveScheduler(base_detection_interval=3)

        # Frame 1: Should run (no previous detection)
        assert scheduler.should_detect(dummy_frame, active_track_count=0) is True

        # Frame 2: Skip
        assert scheduler.should_detect(dummy_frame, active_track_count=0) is False

        # Frame 3: Skip
        assert scheduler.should_detect(dummy_frame, active_track_count=0) is False

        # Frame 4: Run (interval of 3 elapsed)
        assert scheduler.should_detect(dummy_frame, active_track_count=0) is True

    def test_force_detection(self, dummy_frame):
        scheduler = AdaptiveScheduler(base_detection_interval=10)

        assert scheduler.should_detect(dummy_frame, active_track_count=1) is True

        # Next frame would normally be skipped
        assert scheduler.should_detect(dummy_frame, active_track_count=1) is False

        # Force detection
        scheduler.force_detection()
        assert scheduler.should_detect(dummy_frame, active_track_count=1) is True

    def test_gpu_load_throttling(self, dummy_frame):
        scheduler = AdaptiveScheduler(base_detection_interval=3, max_gpu_load_percent=80.0)

        # High GPU load (90%) should increase interval
        scheduler.should_detect(dummy_frame, active_track_count=1, current_gpu_load=90.0)
        assert scheduler.current_interval > 3
