"""
Unit Tests — Tactical AI Copilot RAG Agent & Instance Segmentation Subsystems
"""

import pytest
from apex.engine.agent.tactical_rag_agent import TacticalAgentRAG
from apex.engine.contracts.detection import BoundingBox, Detection
from apex.engine.contracts.track import Track, TrackState
from plugins.detectors.yolo_seg.plugin import YOLOInstanceSegmentationPlugin


class TestTacticalCopilotAndSegmentation:

    def test_tactical_copilot_rag_queries(self):
        copilot = TacticalAgentRAG.instance()

        # 1. Audit / History Query
        res_history = copilot.query("Show me total history records")
        assert "ReAct Agent Audit" in res_history["agent"]
        assert "SQLite Target Database" in res_history["data_sources"][0]

        # 2. Threat Matrix Query
        res_threat = copilot.query("What is the threat matrix score?")
        assert "ReAct Agent Threat Matrix" in res_threat["agent"]

        # 3. RL Engine Query
        res_rl = copilot.query("What is the RL agent status?")
        assert "ReAct Agent RL Policy" in res_rl["agent"]

        # 4. General Telemetry Query
        res_general = copilot.query("Status report")
        assert "ReAct Agent" in res_general["agent"]

    def test_instance_segmentation_polygon_masks(self):
        plugin = YOLOInstanceSegmentationPlugin()
        # Mock frame with width and height metadata
        from apex.engine.contracts.frame import Frame, FrameMetadata
        import numpy as np

        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame = Frame(data=img, metadata=FrameMetadata(camera_id="cam_0", width=1280, height=720))

        detections = plugin.detect(frame)
        assert len(detections) >= 2
        for det in detections:
            assert det.segmentation_mask is not None
            assert len(det.segmentation_mask) > 3  # Valid polygon vertices
