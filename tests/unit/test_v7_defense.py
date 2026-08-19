"""
Unit Tests — APEX-Track v7.0 Master Defense Standard Upgrade
Verifies CLAHE Image Enhancement, STANAG Defense Threat Matrix (ALPHA-DELTA), and Universal AI Copilot Intent Engine.
"""

import numpy as np
import pytest

from apex.engine.agent.tactical_rag_agent import TacticalAgentRAG
from apex.engine.contracts.detection import BoundingBox
from apex.engine.contracts.track import Track, TrackState
from apex.engine.db.target_database import TargetDatabase
from apex.engine.detector.enhancer import DefenseImageEnhancer


class TestV7MasterDefenseStandard:

    def test_defense_image_enhancer_clahe_and_sharpening(self):
        enhancer = DefenseImageEnhancer(clip_limit=2.5)
        raw_bgr = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)

        enhanced = enhancer.enhance_frame(raw_bgr, motion_sharpen=True)
        assert enhanced is not None
        assert enhanced.shape == raw_bgr.shape
        assert enhanced.dtype == np.uint8

    def test_stanag_defense_threat_matrix_alert_levels(self):
        db = TargetDatabase()
        bbox = BoundingBox(600, 300, 680, 380)
        track = Track(
            track_id=42,
            class_name="drone",
            bbox=bbox,
            predicted_bbox=bbox,
            confidence=0.92,
            class_id=0,
            frame_timestamp=100.0,
            state=TrackState.CONFIRMED,
            speed_kmh=110.0,
        )

        threat_score = db.compute_threat_level(track)
        stanag_info = db.get_stanag_threat_level(track)

        assert 0.0 <= threat_score <= 1.0
        assert "alert_level" in stanag_info
        assert any(lvl in stanag_info["alert_level"] for lvl in ["ALPHA", "BRAVO", "CHARLIE", "DELTA"])

    def test_universal_ai_copilot_answers_any_query(self):
        agent = TacticalAgentRAG.instance()

        # Query 1: System Specs / Capabilities
        res1 = agent.query("What are the capabilities of APEX-Track?")
        assert "System Overview" in res1["agent"]
        assert "Perception Engine" in res1["agent"]

        # Query 2: General / Arbitrary User Prompt (Universal Fallback)
        res2 = agent.query("How can I improve detection accuracy in low light?")
        assert "Defense Copilot" in res2["agent"] or "Tactical Analysis" in res2["agent"]
        assert res2["agent"] is not None
        assert len(res2["agent"]) > 20
