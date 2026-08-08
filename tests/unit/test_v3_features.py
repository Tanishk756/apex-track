"""
Unit tests for APEX-Track v3.0 Advanced Defense & C4ISR Features
"""

import pytest
import numpy as np

from apex.engine.contracts.detection import BoundingBox, Detection
from apex.engine.contracts.track import Track, TrackState
from apex.engine.spatial.trajectory_predictor import TrajectoryPredictor
from apex.engine.mission.threat_matrix import ThreatMatrixEngine
from apex.engine.pipeline.thermal_fusion import ThermalFusionShader, ThermalVisionMode
from apex.engine.fusion.sensor_fusion import SensorFusionEngine
from apex.engine.spatial.swarm_defense import SwarmDefenseGrid
from apex.engine.analytics.anomaly_detector import AnomalyDetector


def create_mock_track(tid: int, cx: float, cy: float, cls_name: str = "drone") -> Track:
    bbox = BoundingBox(cx - 20, cy - 20, cx + 20, cy + 20)
    return Track(
        track_id=tid,
        state=TrackState.CONFIRMED,
        bbox=bbox,
        predicted_bbox=bbox,
        confidence=0.95,
        class_id=0,
        class_name=cls_name,
        frame_timestamp=100.0,
        camera_id="cam0",
        velocity_px=(5.0, 2.0),
    )


def test_trajectory_predictor():
    predictor = TrajectoryPredictor()
    tr = create_mock_track(1, 100.0, 100.0)
    res = predictor.update_and_predict([tr])
    assert 1 in res
    assert len(res[1]["future_points"]) == 4
    assert res[1]["future_points"][0] == (105.0, 102.0)


def test_threat_matrix_engine():
    engine = ThreatMatrixEngine()
    tr1 = create_mock_track(1, 640.0, 360.0, "drone")
    tr2 = create_mock_track(2, 10.0, 10.0, "person")

    traj_data = {
        1: {"speed_px": 25.0, "is_closing": True, "ttc_seconds": 3.5},
        2: {"speed_px": 2.0, "is_closing": False, "ttc_seconds": 999.0},
    }

    res = engine.evaluate_threats([tr1, tr2], traj_data)
    assert res["primary_lock_id"] == 1
    assert res["threat_matrix"][1]["level"] == "ALPHA"


def test_thermal_fusion_shader():
    shader = ThermalFusionShader(mode=ThermalVisionMode.FLIR_IRONBOW)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    out = shader.apply_fusion(frame)
    assert out.shape == (100, 100, 3)

    shader.mode = ThermalVisionMode.NVG_GREEN
    out_nvg = shader.apply_fusion(frame)
    assert out_nvg.shape == (100, 100, 3)


def test_swarm_defense_grid():
    grid = SwarmDefenseGrid()
    d1 = create_mock_track(1, 500.0, 500.0, "drone")
    d2 = create_mock_track(2, 530.0, 510.0, "drone")
    d3 = create_mock_track(3, 490.0, 540.0, "drone")

    res = grid.analyze_swarms([d1, d2, d3])
    assert res["swarm_detected"] is True
    assert res["drone_count"] == 3


def test_sensor_fusion_engine():
    engine = SensorFusionEngine()
    tr = create_mock_track(1, 640.0, 360.0, "vehicle")
    fused = engine.correlate_tracks([tr])
    assert len(fused) == 1
    assert fused[0]["optical_azimuth_deg"] == 0.0


def test_anomaly_detector():
    detector = AnomalyDetector(high_speed_thresh=20.0)
    tr = create_mock_track(1, 100.0, 100.0, "car")
    traj_data = {1: {"speed_px": 45.0, "is_closing": True}}
    anomalies = detector.detect_anomalies([tr], traj_data)
    assert len(anomalies) == 1
    assert anomalies[0]["anomaly_type"] == "HIGH_SPEED_BREAKOUT"
