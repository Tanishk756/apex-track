"""
Unit Tests — Camera Layer (Phase 2)
"""

import asyncio
import time
import pytest
import numpy as np

from apex.engine.bus.channels import Ch
from apex.engine.bus.message_bus import MessageBus
from apex.engine.camera.camera_manager import CameraManager
from apex.engine.camera.frame_synchronizer import FrameSynchronizer
from apex.engine.contracts.frame import Frame, FrameMetadata
from apex.engine.hal.hw_profile import HWCapabilities, HWProfile
from plugins.cameras.file_camera.plugin import FileCameraPlugin


@pytest.fixture(autouse=True)
def reset_singletons():
    MessageBus.reset()
    yield
    MessageBus.reset()


@pytest.fixture
def mock_hw():
    return HWProfile(capabilities=HWCapabilities(), profile_name="test_cpu")


class TestFileCameraPlugin:

    @pytest.mark.asyncio
    async def test_synthetic_camera_load_and_stream(self, mock_hw):
        cam = FileCameraPlugin(camera_id="cam_synth")
        cfg = {"source": "synthetic", "fps": 50.0, "width": 640, "height": 480}
        await cam.load(cfg, mock_hw)
        assert cam.is_active is True

        bus = MessageBus.instance()
        received_frames = []

        async def collect():
            async for _, frame in bus.subscribe(Ch.camera_frame("cam_synth")):
                received_frames.append(frame)
                if len(received_frames) >= 3:
                    break

        collector_task = asyncio.create_task(collect())
        await asyncio.sleep(0.01)
        await cam.start_streaming()

        await asyncio.wait_for(collector_task, timeout=2.0)
        await cam.stop_streaming()
        await cam.unload()

        assert len(received_frames) >= 3
        first = received_frames[0]
        assert isinstance(first, Frame)
        assert first.metadata.camera_id == "cam_synth"
        assert first.metadata.width == 640
        assert first.metadata.height == 480
        assert isinstance(first.data, np.ndarray)


class TestFrameSynchronizer:

    def test_single_camera_push_pop(self):
        sync = FrameSynchronizer(camera_ids=["cam0"], tolerance_ms=50.0)
        meta = FrameMetadata(camera_id="cam0", width=640, height=480)
        f0 = Frame(data=np.zeros((10, 10, 3)), metadata=meta, timestamp=100.0, sequence_id=1)

        sync.push(f0)
        bundle = sync.pop_synced_bundle()
        assert bundle is not None
        assert "cam0" in bundle
        assert bundle["cam0"].sequence_id == 1

    def test_multi_camera_synced_matching(self):
        sync = FrameSynchronizer(camera_ids=["eo", "ir"], tolerance_ms=50.0)
        meta_eo = FrameMetadata(camera_id="eo", width=640, height=480)
        meta_ir = FrameMetadata(camera_id="ir", width=640, height=480)

        # Timestamps within 20ms (tolerance is 50ms)
        f_eo = Frame(data=np.zeros((10, 10, 3)), metadata=meta_eo, timestamp=10.01, sequence_id=1)
        f_ir = Frame(data=np.zeros((10, 10, 3)), metadata=meta_ir, timestamp=10.02, sequence_id=101)

        sync.push(f_eo)
        assert sync.pop_synced_bundle() is None  # ir frame not pushed yet

        sync.push(f_ir)
        bundle = sync.pop_synced_bundle()
        assert bundle is not None
        assert bundle["eo"].sequence_id == 1
        assert bundle["ir"].sequence_id == 101
        assert sync.synced_count == 1

    def test_out_of_tolerance_drop(self):
        sync = FrameSynchronizer(camera_ids=["eo", "ir"], tolerance_ms=30.0)
        meta_eo = FrameMetadata(camera_id="eo", width=640, height=480)
        meta_ir = FrameMetadata(camera_id="ir", width=640, height=480)

        # Timestamps 100ms apart (exceeds 30ms tolerance)
        f_eo = Frame(data=np.zeros((10, 10, 3)), metadata=meta_eo, timestamp=10.00, sequence_id=1)
        f_ir = Frame(data=np.zeros((10, 10, 3)), metadata=meta_ir, timestamp=10.10, sequence_id=101)

        sync.push(f_eo)
        sync.push(f_ir)
        bundle = sync.pop_synced_bundle()
        assert bundle is None  # timestamps don't match within tolerance


class TestCameraManager:

    @pytest.mark.asyncio
    async def test_camera_manager_lifecycle(self, mock_hw):
        mgr = CameraManager(hw_profile=mock_hw)

        configs = [
            {"camera_id": "test_cam_0", "plugin": "file_camera", "source": "synthetic", "fps": 60.0},
            {"camera_id": "test_cam_1", "plugin": "file_camera", "source": "synthetic", "fps": 60.0},
        ]

        loaded = await mgr.load_cameras(configs)
        assert loaded == 2
        assert len(mgr.camera_ids) == 2

        await mgr.start_all()
        await asyncio.sleep(0.15)

        stats = mgr.get_stats()
        assert "test_cam_0" in stats
        assert stats["test_cam_0"]["total_frames"] > 0

        await mgr.stop_all()
        assert len(mgr.camera_ids) == 0
