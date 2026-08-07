"""
Inference Engine Base Class
===========================
Hardware-agnostic abstract base class for neural network inference execution.

Design:
- Decouples detector logic from the underlying compute backend (TensorRT, ONNX Runtime, PyTorch).
- Zero-copy buffer support for GPU input tensors.
- Built-in latency timing and warmup routines.
"""

from __future__ import annotations

import abc
import time
from typing import Any, Optional, Union

import numpy as np
import structlog

log = structlog.get_logger(__name__)


class InferenceEngine(abc.ABC):
    """
    Abstract base class for inference execution backends.

    Concrete implementations:
    - TensorRTEngine (NVIDIA GPU / Jetson optimized, FP16/INT8)
    - ONNXEngine (CPU / CUDA / TensorRT Execution Providers)
    - PyTorchEngine (CPU / CUDA fallback)
    """

    def __init__(self) -> None:
        self.model_path: str = ""
        self.input_shape: tuple[int, ...] = (1, 3, 640, 640)
        self.precision: str = "fp32"
        self._last_latency_ms: float = 0.0
        self._is_loaded: bool = False

    @abc.abstractmethod
    def load(self, model_path: str, precision: str = "fp32", device_id: int = 0) -> None:
        """Load and compile/optimize model for target hardware."""

    @abc.abstractmethod
    def infer(self, input_tensor: Union[np.ndarray, Any]) -> Any:
        """
        Execute forward pass on input_tensor.
        Returns raw output tensors (numpy ndarray or PyTorch tensor).
        """

    def warmup(self, warmup_runs: int = 5) -> None:
        """Run dummy inference iterations to warm up GPU caches & CUDA streams."""
        if not self._is_loaded:
            log.warning("warmup_skipped_model_not_loaded")
            return

        dummy_input = np.zeros(self.input_shape, dtype=np.float32)
        log.info("starting_engine_warmup", runs=warmup_runs)
        for _ in range(warmup_runs):
            self.infer(dummy_input)
        log.info("engine_warmup_complete")

    @property
    def last_latency_ms(self) -> float:
        return self._last_latency_ms

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded
