"""
TensorRT Native Inference Engine
================================
Native TensorRT execution engine for NVIDIA GPUs and Jetson hardware.
Provides sub-10ms neural inference using TensorRT C++ runtime bindings (via python tensorrt module)
with page-locked pinned host memory and CUDA streams.
"""

from __future__ import annotations

import time
from typing import Any, Union

import numpy as np
import structlog

from apex.engine.inference.base import InferenceEngine

log = structlog.get_logger(__name__)

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    HAS_TRT = True
except ImportError:
    HAS_TRT = False


class TensorRTEngine(InferenceEngine):
    """Native TensorRT execution engine."""

    def __init__(self) -> None:
        super().__init__()
        self._trt_logger: Any = None
        self._engine: Any = None
        self._context: Any = None
        self._inputs: list[dict] = []
        self._outputs: list[dict] = []
        self._bindings: list[int] = []
        self._stream: Any = None

    def load(self, model_path: str, precision: str = "fp16", device_id: int = 0) -> None:
        if not HAS_TRT:
            log.warning("tensorrt_not_installed_falling_back")
            raise RuntimeError("TensorRT or PyCUDA is not installed in the python environment.")

        self.model_path = model_path
        self.precision = precision

        log.info("loading_tensorrt_engine", path=model_path)
        self._trt_logger = trt.Logger(trt.Logger.WARNING)

        with open(model_path, "rb") as f, trt.Runtime(self._trt_logger) as runtime:
            self._engine = runtime.deserialize_cuda_engine(f.read())

        self._context = self._engine.create_execution_context()
        self._stream = cuda.Stream()

        self._inputs = []
        self._outputs = []
        self._bindings = []

        for binding in self._engine:
            size = trt.volume(self._engine.get_binding_shape(binding)) * self._engine.max_batch_size
            dtype = trt.nbytes(self._engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            self._bindings.append(int(cuda_mem))

            if self._engine.binding_is_input(binding):
                self._inputs.append({"host": host_mem, "device": cuda_mem})
            else:
                self._outputs.append({"host": host_mem, "device": cuda_mem})

        self._is_loaded = True

    def infer(self, input_tensor: Union[np.ndarray, Any]) -> list[np.ndarray]:
        if not self._is_loaded or self._context is None:
            raise RuntimeError("TensorRTEngine model is not loaded.")

        t0 = time.perf_counter()

        # Copy input data to pagelocked memory
        np.copyto(self._inputs[0]["host"], input_tensor.ravel())

        # Transfer to GPU asynchronously
        cuda.memcpy_htod_async(self._inputs[0]["device"], self._inputs[0]["host"], self._stream)

        # Execute TensorRT context asynchronously
        self._context.execute_async_v2(bindings=self._bindings, stream_handle=self._stream.handle)

        # Transfer predictions back to CPU host asynchronously
        for out in self._outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], self._stream)

        # Synchronize CUDA stream
        self._stream.synchronize()
        self._last_latency_ms = (time.perf_counter() - t0) * 1000.0

        return [out["host"] for out in self._outputs]
