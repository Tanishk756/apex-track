"""
Detector Base Class
===================
Abstract base class for object detection plugins.

Design:
- Inherits PluginBase to integrate with PluginLoader and PluginRegistry.
- Wraps an InferenceEngine instance for model execution.
- Standardizes pre-processing (letterbox scaling, normalization) and post-processing (NMS, coordinate scaling).
- Emits Detection objects aligned with Frame metadata.
"""

from __future__ import annotations

import abc
import time
from typing import Any, Optional

import numpy as np
import structlog

from apex.engine.contracts.detection import BoundingBox, Detection
from apex.engine.contracts.frame import Frame
from apex.engine.hal.hw_profile import HWProfile
from apex.engine.inference.base import InferenceEngine
from apex.engine.inference.factory import EngineFactory
from apex.engine.plugins.plugin_base import PluginBase, PluginStatus

log = structlog.get_logger(__name__)


class DetectorBase(PluginBase, abc.ABC):
    """Abstract base class for object detector plugins."""

    def __init__(self) -> None:
        super().__init__()
        self.engine: Optional[InferenceEngine] = None
        self.conf_threshold: float = 0.45
        self.nms_threshold: float = 0.45
        self.input_size: tuple[int, int] = (640, 640)
        self.class_names: list[str] = []
        self._detect_count = 0
        self._total_infer_time_ms = 0.0

    @abc.abstractmethod
    def _preprocess(self, frame: Frame) -> tuple[np.ndarray, float, tuple[int, int]]:
        """
        Preprocess input image for neural model.
        Returns (blob_nchw, scale_factor, (pad_w, pad_h)).
        """

    @abc.abstractmethod
    def _postprocess(
        self,
        raw_outputs: Any,
        frame: Frame,
        scale: float,
        pad: tuple[int, int],
    ) -> list[Detection]:
        """Convert raw network output tensors into Detection contracts."""

    async def load(self, config: dict, hw_profile: HWProfile) -> None:
        """Initialize detector configuration and load backend InferenceEngine."""
        self.config = config
        self.hw_profile = hw_profile
        self.conf_threshold = float(config.get("confidence_threshold", self.conf_threshold))
        self.nms_threshold = float(config.get("nms_iou_threshold", self.nms_threshold))

        model_path = config.get("model_path", "")
        if not model_path:
            # Synthetic / Mock detector fallback for testing
            log.info("detector_using_mock_fallback", plugin=self.metadata.name)
            self._set_status(PluginStatus.ACTIVE)
            return

        precision = config.get("fp_precision", hw_profile.capabilities.recommended_fp_precision)
        self.engine = EngineFactory.create(model_path, hw_profile, preferred_precision=precision)
        self._set_status(PluginStatus.ACTIVE)

    async def unload(self) -> None:
        """Unload detector model and free engine resources."""
        self._set_status(PluginStatus.UNLOADED)
        self.engine = None

    def detect(self, frame: Frame) -> list[Detection]:
        """Run end-to-end detection on a Frame."""
        if not self.is_active:
            raise RuntimeError(f"Detector {self.metadata.name} is not active.")

        t0 = time.perf_counter()

        if self.engine is None or not self.engine.is_loaded:
            # Return synthetic test detection if no backend model is loaded
            return self._generate_mock_detections(frame)

        blob, scale, pad = self._preprocess(frame)
        raw_outputs = self.engine.infer(blob)
        detections = self._postprocess(raw_outputs, frame, scale, pad)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._detect_count += 1
        self._total_infer_time_ms += elapsed_ms

        return detections

    def _generate_mock_detections(self, frame: Frame) -> list[Detection]:
        """Generate test detection bounding box for synthetic verification."""
        h, w = frame.hw
        # Center target
        cx, cy = w / 2, h / 2
        bw, bh = 80, 80
        bbox = BoundingBox(cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)
        det = Detection(
            bbox=bbox,
            confidence=0.92,
            class_id=0,
            class_name="vehicle",
            camera_id=frame.metadata.camera_id,
            detector_id=self.metadata.name,
            frame_timestamp=frame.timestamp,
        )
        return [det]

    @property
    def avg_latency_ms(self) -> float:
        return self._total_infer_time_ms / self._detect_count if self._detect_count > 0 else 0.0
