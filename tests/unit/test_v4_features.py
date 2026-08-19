"""
Unit tests for APEX-Track v4.0 Ultimate Defense Suite (RF-DETR 2XL, Intercept Geometry, Countermeasures)
"""

import pytest
import numpy as np

from apex.engine.contracts.detection import BoundingBox
from apex.engine.contracts.track import Track, TrackState
from plugins.detectors.rf_detr.plugin import RFDetr2XLPlugin
from apex.engine.spatial.intercept_calculator import InterceptCalculator
from apex.engine.mission.countermeasures import CountermeasureEngine


def create_mock_track(tid: int, cx: float, cy: float, cls_name: str = "drone") -> Track:
    bbox = BoundingBox(cx - 20, cy - 20, cx + 20, cy + 20)
    return Track(
        track_id=tid,
        state=TrackState.CONFIRMED,
        bbox=bbox,
        predicted_bbox=bbox,
        confidence=0.98,
        class_id=0,
        class_name=cls_name,
        frame_timestamp=100.0,
        camera_id="cam0",
        velocity_px=(10.0, 5.0),
        speed_kmh=60.0,
    )


def test_rf_detr_2xl_plugin():
    plugin = RFDetr2XLPlugin()
    assert plugin.metadata.name == "rf_detr_2xl"
    assert plugin.model_name == "rfdetr-2xl"
    assert "PML-1.0" in plugin.metadata.license


def test_intercept_calculator():
    calc = InterceptCalculator(fov_h_deg=60.0, fov_v_deg=35.0)
    tr = create_mock_track(1, 640.0, 360.0, "drone")
    vector = calc.compute_intercept(tr)

    assert vector["track_id"] == 1
    assert vector["azimuth_deg"] == 0.0
    assert vector["elevation_deg"] == 0.0
    assert vector["slant_range_m"] > 0.0
    assert vector["tti_seconds"] > 0.0


def test_countermeasure_engine():
    engine = CountermeasureEngine()
    threat_data = {"alpha_target_id": 1, "max_threat_score": 95.0}
    intercept_data = [{"track_id": 1, "slant_range_m": 50.0}]

    res = engine.evaluate_countermeasures(threat_data, intercept_data)
    assert res["rf_jamming_active"] is True
    assert res["kinetic_intercept_engaged"] is True

    jam_res = engine.trigger_manual_jamming(2)
    assert jam_res["mode"] == "RF_JAMMING"
    assert jam_res["target_id"] == 2

    kinetic_res = engine.trigger_manual_intercept(2)
    assert kinetic_res["mode"] == "KINETIC_INTERCEPT"
    assert kinetic_res["target_id"] == 2
