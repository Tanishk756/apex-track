"""
Unit Tests — APEX-Track v8.0 Strategic Autonomous Supremacy Edition
Verifies Weighted Boxes Fusion (WBF), Autonomous Dataset Harvester & Fine-Tuner, and Inspection Diagnostics.
"""

import numpy as np
import pytest

from apex.engine.contracts.detection import BoundingBox, Detection
from apex.engine.detector.ensemble import WeightedBoxesFusion, compute_iou
from apex.engine.training.training_pipeline import AutonomousFineTuner


class TestV8StrategicAutonomousSupremacy:

    def test_wbf_ensemble_bounding_box_fusion(self):
        wbf = WeightedBoxesFusion(iou_thresh=0.50)

        det1 = Detection(BoundingBox(100, 100, 200, 200), confidence=0.85, class_id=0, class_name="drone")
        det2 = Detection(BoundingBox(105, 105, 205, 205), confidence=0.90, class_id=0, class_name="drone")

        fused = wbf.fuse_detections([[det1], [det2]])
        assert len(fused) == 1
        assert fused[0].class_name == "drone"
        assert fused[0].confidence > 0.90  # Consensus confidence bonus!
        assert 100.0 < fused[0].bbox.x1 < 105.0

    def test_autonomous_dataset_harvester_and_fine_tuner(self):
        tuner = AutonomousFineTuner.instance()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        ok = tuner.harvest_edge_case(frame, [], reason="unit_test")
        assert ok is True

        status = tuner.get_status()
        assert status["harvested_samples"] > 0
        assert "fine_tune_sessions" in status

    def test_compute_iou_utility(self):
        b1 = BoundingBox(0, 0, 10, 10)
        b2 = BoundingBox(0, 0, 10, 10)
        assert compute_iou(b1, b2) == pytest.approx(1.0)

        b3 = BoundingBox(20, 20, 30, 30)
        assert compute_iou(b1, b3) == pytest.approx(0.0)
