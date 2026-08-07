"""
ONNX Runtime Inference Engine
=============================
Cross-platform inference engine using ONNX Runtime.
Automatically selects optimal Execution Provider (TensorRT → CUDA → CPU).
"""

from __future__ import annotations

import time
from typing import Any, Union

import numpy as np
import structlog

from apex.engine.inference.base import InferenceEngine

log = structlog.get_logger(__name__)

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False


class ONNXEngine(InferenceEngine):
    """ONNX Runtime execution backend."""

    def __init__(self) -> None:
        super().__init__()
        self._session: Any = None
        self._input_name: str = ""
        self._output_names: list[str] = []

    def load(self, model_path: str, precision: str = "fp32", device_id: int = 0) -> None:
        if not HAS_ORT:
            raise RuntimeError("onnxruntime is not installed in the python environment.")

        self.model_path = model_path
        self.precision = precision

        available_eps = ort.get_available_providers()
        providers = []

        if "TensorrtExecutionProvider" in available_eps and precision in ("fp16", "int8"):
            providers.append(
                (
                    "TensorrtExecutionProvider",
                    {
                        "device_id": device_id,
                        "trt_fp16_enable": precision == "fp16",
                        "trt_int8_enable": precision == "int8",
                    },
                )
            )

        if "CUDAExecutionProvider" in available_eps:
            providers.append(("CUDAExecutionProvider", {"device_id": device_id}))

        providers.append("CPUExecutionProvider")

        log.info("loading_onnx_model", path=model_path, providers=[p[0] if isinstance(p, tuple) else p for p in providers])
        self._session = ort.InferenceSession(model_path, providers=providers)

        # Inspect model inputs/outputs
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()

        self._input_name = inputs[0].name
        shape = inputs[0].shape
        # Handle dynamic dimensions by defaulting to 1 or 640
        self.input_shape = tuple(
            dim if isinstance(dim, int) and dim > 0 else (1 if i == 0 else 640)
            for i, dim in enumerate(shape)
        )
        self._output_names = [out.name for out in outputs]
        self._is_loaded = True

    def infer(self, input_tensor: Union[np.ndarray, Any]) -> list[np.ndarray]:
        if not self._is_loaded or self._session is None:
            raise RuntimeError("ONNXEngine model is not loaded.")

        if not isinstance(input_tensor, np.ndarray):
            input_tensor = np.array(input_tensor, dtype=np.float32)

        t0 = time.perf_counter()
        outputs = self._session.run(self._output_names, {self._input_name: input_tensor})
        self._last_latency_ms = (time.perf_counter() - t0) * 1000.0

        return outputs
