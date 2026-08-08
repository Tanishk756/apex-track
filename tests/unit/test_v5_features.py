"""
Unit tests for APEX-Track v5.0 Master Enterprise C4ISR Suite
"""

import pytest
from plugins.detectors.ensemble.plugin import EnsembleDetectorPlugin
from apex.engine.hal.plugin_hub import PluginHub
from apex.engine.hal.mavlink_stanag import MAVLinkStanagEngine


def test_ensemble_detector_plugin():
    plugin = EnsembleDetectorPlugin()
    assert plugin.metadata.name == "ensemble_detector"
    assert "yolov8x.pt" in plugin.active_models


def test_plugin_hub():
    hub = PluginHub()
    catalog = hub.list_registered_plugins()
    assert len(catalog) >= 1
    plugin_names = [p["name"] for p in catalog]
    assert "ensemble_detector" in plugin_names or "rf_detr_2xl" in plugin_names or "roboflow_detector" in plugin_names


def test_mavlink_stanag_engine():
    engine = MAVLinkStanagEngine()
    status = engine.get_interop_status()
    assert status["stanag_4609_klv"] == "ACTIVE_STREAMING"
    assert status["telemetry_rate_hz"] == 50.0

    klv = engine.encode_stanag_klv([{"id": 1}])
    assert len(klv) > 16
