"""
Engine Factory
==============
Auto-selects and instantiates the optimal InferenceEngine backend based on HWProfile capabilities
and model file extension (.engine, .onnx, .pt).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import structlog

from apex.engine.hal.hw_profile import Capability, HWProfile
from apex.engine.inference.base import InferenceEngine
from apex.engine.inference.onnx_engine import ONNXEngine
from apex.engine.inference.pytorch_engine import PyTorchEngine
from apex.engine.inference.tensorrt_engine import TensorRTEngine

log = structlog.get_logger(__name__)


class EngineFactory:
    """Factory for creating hardware-matched inference engine instances."""

    @staticmethod
    def create(
        model_path: str,
        hw_profile: HWProfile,
        preferred_precision: Optional[str] = None,
        device_id: int = 0,
    ) -> InferenceEngine:
        """
        Instantiate and initialize the best inference engine for the target model & hardware.
        """
        path = Path(model_path)
        ext = path.suffix.lower()
        precision = preferred_precision or hw_profile.capabilities.recommended_fp_precision

        log.info("creating_inference_engine", model=model_path, ext=ext, hw=hw_profile.profile_name)

        # 1. Native TensorRT engine file
        if ext in (".engine", ".trt"):
            if hw_profile.has(Capability.TENSORRT):
                try:
                    engine = TensorRTEngine()
                    engine.load(model_path, precision=precision, device_id=device_id)
                    return engine
                except Exception as exc:
                    log.warning("tensorrt_native_load_failed", error=str(exc))

            log.info("falling_back_to_onnx_engine_for_trt")
            # Attempt ONNX fallback if available
            onnx_alt = path.with_suffix(".onnx")
            if onnx_alt.exists():
                return EngineFactory.create(str(onnx_alt), hw_profile, precision, device_id)

        # 2. ONNX model
        if ext == ".onnx":
            try:
                engine = ONNXEngine()
                engine.load(model_path, precision=precision, device_id=device_id)
                return engine
            except Exception as exc:
                log.warning("onnx_engine_load_failed", error=str(exc))

        # 3. PyTorch model
        if ext in (".pt", ".pth", ".pts"):
            try:
                engine = PyTorchEngine()
                engine.load(model_path, precision=precision, device_id=device_id)
                return engine
            except Exception as exc:
                log.warning("pytorch_engine_load_failed", error=str(exc))

        # Fallback default: ONNX engine if installed, else PyTorch
        log.warning("using_generic_onnx_fallback")
        engine = ONNXEngine()
        if path.exists():
            engine.load(model_path, precision=precision, device_id=device_id)
        return engine
