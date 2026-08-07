"""
Unit Tests — Config Manager
"""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from apex.engine.config.config_manager import ConfigManager
from apex.engine.config.schema import ApexConfig


@pytest.fixture(autouse=True)
def reset_singleton():
    """Ensure ConfigManager singleton is reset between tests."""
    ConfigManager.reset()
    yield
    ConfigManager.reset()


@pytest.fixture
def tmp_configs(tmp_path):
    """Create minimal config files in a temp dir."""
    default = {
        "version": "1.0",
        "detector": {"plugin": "rtdetr", "confidence_threshold": 0.45},
        "tracker": {"plugin": "bytetrack"},
    }
    (tmp_path / "default.yaml").write_text(yaml.dump(default))

    hw = {"detector": {"confidence_threshold": 0.35}}
    (tmp_path / "desktop_rtx.yaml").write_text(yaml.dump(hw))

    return tmp_path


class TestConfigManager:

    def test_singleton(self):
        a = ConfigManager.instance()
        b = ConfigManager.instance()
        assert a is b

    def test_load_returns_apex_config(self, tmp_configs):
        mgr = ConfigManager.instance()
        cfg = mgr.load(hw_profile_name="desktop_rtx", configs_dir=tmp_configs)
        assert isinstance(cfg, ApexConfig)

    def test_hw_profile_merges(self, tmp_configs):
        mgr = ConfigManager.instance()
        cfg = mgr.load(hw_profile_name="desktop_rtx", configs_dir=tmp_configs)
        # desktop_rtx.yaml overrides confidence to 0.35
        assert cfg.detector.confidence_threshold == pytest.approx(0.35)

    def test_env_var_override(self, tmp_configs, monkeypatch):
        monkeypatch.setenv("APEX_DETECTOR__PLUGIN", "yolo11")
        mgr = ConfigManager.instance()
        cfg = mgr.load(hw_profile_name="cpu_only", configs_dir=tmp_configs)
        assert cfg.detector.plugin == "yolo11"

    def test_env_var_bool(self, tmp_configs, monkeypatch):
        monkeypatch.setenv("APEX_HARDWARE__FORCE_CPU", "true")
        mgr = ConfigManager.instance()
        cfg = mgr.load(configs_dir=tmp_configs)
        assert cfg.hardware.force_cpu is True

    def test_hot_reload(self, tmp_configs):
        mgr = ConfigManager.instance()
        mgr.load(configs_dir=tmp_configs)
        mgr.update({"detector": {"confidence_threshold": 0.10}})
        assert mgr.config.detector.confidence_threshold == pytest.approx(0.10)

    def test_listener_called_on_update(self, tmp_configs):
        mgr = ConfigManager.instance()
        mgr.load(configs_dir=tmp_configs)
        received = []
        mgr.on_update(lambda cfg: received.append(cfg))
        mgr.update({"detector": {"confidence_threshold": 0.55}})
        assert len(received) == 1
        assert isinstance(received[0], ApexConfig)

    def test_user_config_file(self, tmp_configs):
        user = tmp_configs / "user.yaml"
        user.write_text(yaml.dump({"detector": {"max_detections": 999}}))
        mgr = ConfigManager.instance()
        cfg = mgr.load(configs_dir=tmp_configs, user_config=user)
        assert cfg.detector.max_detections == 999

    def test_missing_hw_profile_uses_defaults(self, tmp_configs):
        """A missing hw profile file should not raise — just use defaults."""
        mgr = ConfigManager.instance()
        cfg = mgr.load(hw_profile_name="nonexistent_hw", configs_dir=tmp_configs)
        assert isinstance(cfg, ApexConfig)

    def test_config_not_initialized_raises(self):
        mgr = ConfigManager.instance()
        with pytest.raises(RuntimeError, match="not initialized"):
            _ = mgr.config
