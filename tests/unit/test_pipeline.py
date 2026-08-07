"""
Unit Tests — Pipeline & Recording (Phase 10)
"""

import pytest
import numpy as np

from apex.engine.contracts.frame import Frame, FrameMetadata
from apex.engine.contracts.track import TrackArray
from apex.engine.hal.hw_profile import HWCapabilities, HWProfile
from apex.engine.pipeline.master_pipeline import MasterPipeline
from apex.engine.pipeline.video_recorder import VideoRecorder


class TestMasterPipeline:

    @pytest.mark.asyncio
    async def test_master_pipeline_execution(self):
        hw = HWProfile(capabilities=HWCapabilities(), profile_name="cpu_test")
        pipeline = MasterPipeline()
        await pipeline.initialize({}, hw)

        data = np.zeros((480, 640, 3), dtype=np.uint8)
        meta = FrameMetadata(camera_id="cam0", width=640, height=480)
        frame = Frame(data=data, metadata=meta, timestamp=100.0, sequence_id=1)

        result = await pipeline.process_frame(frame)
        assert isinstance(result, TrackArray)
        assert result.camera_id == "cam0"
        assert pipeline.avg_latency_ms > 0.0


class TestVideoRecorder:

    def test_video_recorder_start_stop(self, tmp_path):
        recorder = VideoRecorder(output_dir=str(tmp_path))
        recorder.start_recording(width=160, height=120, session_name="test_rec")

        data = np.zeros((120, 160, 3), dtype=np.uint8)
        meta = FrameMetadata(camera_id="cam0", width=160, height=120)
        frame = Frame(data=data, metadata=meta, timestamp=100.0, sequence_id=1)

        track_array = TrackArray(tracks=(), frame_timestamp=100.0, camera_id="cam0", tracker_id="test")
        recorder.write_frame(frame, track_array)
        recorder.stop_recording()

        assert (tmp_path / "test_rec_raw.mp4").exists()
        assert (tmp_path / "test_rec_hud.mp4").exists()
