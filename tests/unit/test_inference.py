"""
Unit Tests — Inference Engine (Phase 3)
"""

import numpy as np
import pytest

from apex.engine.hal.hw_profile import HWCapabilities, HWProfile
from apex.engine.inference.base import InferenceEngine
from apex.engine.inference.factory import EngineFactory
from apex.engine.inference.onnx_engine import ONNXEngine
from apex.engine.inference.pytorch_engine import PyTorchEngine


class DummyEngine(InferenceEngine):
    """Mock engine for testing base behavior."""

    def load(self, model_path: str, precision: str = "fp32", device_id: int = 0) -> None:
        self.model_path = model_path
        self.precision = precision
        self._is_loaded = True

    def infer(self, input_tensor: np.ndarray) -> np.ndarray:
        return input_tensor * 2.0


class TestInferenceBase:

    def test_dummy_engine_warmup(self):
        engine = DummyEngine()
        engine.load("dummy.onnx")
        assert engine.is_loaded is True

        engine.warmup(warmup_runs=3)
        res = engine.infer(np.ones((1, 3, 640, 640), dtype=np.float32))
        assert np.array_equal(res, np.ones((1, 3, 640, 640), dtype=np.float32) * 2.0)


class TestEngineFactory:

    def test_factory_creates_onnx_or_fallback(self):
        hw = HWProfile(capabilities=HWCapabilities(), profile_name="cpu_test")
        engine = EngineFactory.create("test_model.onnx", hw_profile=hw)
        assert isinstance(engine, InferenceEngine)

    def test_factory_creates_pytorch_engine(self):
        hw = HWProfile(capabilities=HWCapabilities(), profile_name="cpu_test")
        engine = EngineFactory.create("test_model.pt", hw_profile=hw)
        assert isinstance(engine, InferenceEngine)
