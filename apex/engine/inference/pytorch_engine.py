"""
PyTorch Inference Engine
========================
PyTorch fallback inference engine for TorchScript / PyTorch models (.pt / .pts).
Supports CUDA acceleration and FP16 autocast.
"""

from __future__ import annotations

import time
from typing import Any, Union

import numpy as np
import structlog

from apex.engine.inference.base import InferenceEngine

log = structlog.get_logger(__name__)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class PyTorchEngine(InferenceEngine):
    """PyTorch execution backend."""

    def __init__(self) -> None:
        super().__init__()
        self._model: Any = None
        self._device: str = "cpu"

    def load(self, model_path: str, precision: str = "fp32", device_id: int = 0) -> None:
        if not HAS_TORCH:
            raise RuntimeError("torch is not installed in the python environment.")

        self.model_path = model_path
        self.precision = precision
        self._device = f"cuda:{device_id}" if torch.cuda.is_available() and device_id >= 0 else "cpu"

        log.info("loading_pytorch_model", path=model_path, device=self._device)
        try:
            self._model = torch.jit.load(model_path, map_location=self._device)
        except Exception:
            self._model = torch.load(model_path, map_location=self._device)

        if hasattr(self._model, "eval"):
            self._model.eval()

        if precision == "fp16" and self._device.startswith("cuda"):
            self._model = self._model.half()

        self._is_loaded = True

    def infer(self, input_tensor: Union[np.ndarray, Any]) -> Any:
        if not self._is_loaded or self._model is None:
            raise RuntimeError("PyTorchEngine model is not loaded.")

        if isinstance(input_tensor, np.ndarray):
            tensor = torch.from_numpy(input_tensor).to(self._device)
        elif isinstance(input_tensor, torch.Tensor):
            tensor = input_tensor.to(self._device)
        else:
            tensor = torch.tensor(input_tensor, dtype=torch.float32, device=self._device)

        if self.precision == "fp16" and self._device.startswith("cuda"):
            tensor = tensor.half()

        t0 = time.perf_counter()
        with torch.no_grad():
            output = self._model(tensor)
        self._last_latency_ms = (time.perf_counter() - t0) * 1000.0

        if isinstance(output, torch.Tensor):
            return output.cpu().numpy()
        elif isinstance(output, (tuple, list)):
            return [out.cpu().numpy() if isinstance(out, torch.Tensor) else out for out in output]
        return output
