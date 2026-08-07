"""
Unit Tests — Detector Plugins & Ensemble (Phase 4)
"""

import os
import pytest
import numpy as np

from apex.engine.contracts.detection import BoundingBox, Detection
from apex.engine.contracts.frame import Frame, FrameMetadata
from apex.engine.detector.ensemble import EnsembleDetector
from apex.engine.hal.hw_profile import HWCapabilities, HWProfile
from plugins.detectors.rtdetr.plugin import RTDETRPlugin
from plugins.detectors.rtmdet.plugin import RTMDetPlugin
from plugins.detectors.yolo11.plugin import YOLO11Plugin


@pytest.fixture
def mock_hw():
    return HWProfile(capabilities=HWCapabilities(), profile_name="test_cpu")


@pytest.fixture
def sample_frame():
    data = np.zeros((480, 640, 3), dtype=np.uint8)
    meta = FrameMetadata(camera_id="cam_test", width=640, height=480)
    return Frame(data=data, metadata=meta, timestamp=100.0, sequence_id=1)


class TestDetectorPlugins:

    @pytest.mark.asyncio
    async def test_rtdetr_load_and_mock_detect(self, mock_hw, sample_frame):
        detector = RTDETRPlugin()
        await detector.load({"confidence_threshold": 0.5}, mock_hw)
        assert detector.is_active is True

        dets = detector.detect(sample_frame)
        assert isinstance(dets, list)
        assert len(dets) == 1
        assert dets[0].class_name == "vehicle"
        assert dets[0].confidence > 0.5

    @pytest.mark.asyncio
    async def test_rtmdet_load_and_mock_detect(self, mock_hw, sample_frame):
        detector = RTMDetPlugin()
        await detector.load({"confidence_threshold": 0.4}, mock_hw)
        assert detector.is_active is True

        dets = detector.detect(sample_frame)
        assert len(dets) == 1

    @pytest.mark.asyncio
    async def test_yolo11_agpl_metadata(self, mock_hw):
        detector = YOLO11Plugin()
        assert detector.metadata.is_agpl is True
        assert detector.metadata.license == "AGPL-3.0"


class TestEnsembleDetector:

    @pytest.mark.asyncio
    async def test_ensemble_weighted_nms(self, mock_hw, sample_frame):
        d1 = RTDETRPlugin()
        d2 = RTMDetPlugin()
        await d1.load({}, mock_hw)
        await d2.load({}, mock_hw)

        ensemble = EnsembleDetector(detectors=[d1, d2], weights=[1.0, 0.8], iou_threshold=0.5)
        fused = ensemble.detect(sample_frame)

        assert isinstance(fused, list)
        # Should fuse the two overlapping mock bounding boxes into one high-confidence prediction
        assert len(fused) == 1
        assert fused[0].confidence >= 0.9
