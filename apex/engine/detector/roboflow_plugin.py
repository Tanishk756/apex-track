"""
Roboflow Vision Detector Plugin
===============================
Integrates Roboflow Universe model inference for fine-tuned tactical datasets
(e.g., UAV/Drone detection, aerial surveillance, military vehicle identification).
"""

from __future__ import annotations

import time
from typing import Any, List, Optional
import numpy as np
import structlog

from apex.engine.contracts.detection import BoundingBox, Detection
from apex.engine.contracts.frame import Frame
from apex.engine.detector.detector_base import DetectorBase, IGNORED_CLUTTER_CLASSES, VALID_TARGET_CLASSES
from apex.engine.plugins.plugin_base import PluginMetadata, PluginType

log = structlog.get_logger(__name__)


class RoboflowDetectorPlugin(DetectorBase):
    """Roboflow Universe specialized neural object detector plugin."""

    metadata = PluginMetadata(
        name="roboflow_detector",
        version="1.0.0",
        plugin_type=PluginType.DETECTOR,
        license="Apache-2.0",
        author="APEX-Track",
        description="Roboflow Universe fine-tuned tactical vision model engine",
    )

    def __init__(self) -> None:
        super().__init__()
        self.model_id: str = "uav-detection/3"
        self.api_key: Optional[str] = None
        self._rf_model: Any = None

    async def load(self, config: dict[str, Any], hw_profile: Any) -> None:
        """Initialize Roboflow model with API key or local model cache."""
        await super().load(config, hw_profile)
        self.model_id = config.get("roboflow_model_id", "uav-detection/3")
        self.api_key = config.get("roboflow_api_key", None)

        try:
            from inference import get_model
            log.info("loading_roboflow_inference_model", model_id=self.model_id)
            self._rf_model = get_model(model_id=self.model_id, api_key=self.api_key)
        except Exception as exc:
            log.warning("roboflow_inference_sdk_not_available_using_fallback", error=str(exc))
            self._rf_model = None

    def detect(self, frame: Frame) -> list[Detection]:
        """Run detection through Roboflow model if available, else delegate to DetectorBase."""
        if self._rf_model is not None:
            t0 = time.perf_counter()
            try:
                results = self._rf_model.infer(frame.data, confidence=self.conf_threshold)
                detections: list[Detection] = []
                if results and hasattr(results[0], "predictions"):
                    for pred in results[0].predictions:
                        cx, cy, w, h = pred.x, pred.y, pred.width, pred.height
                        score = float(pred.confidence)
                        cls_name = str(pred.class_name).lower().strip()

                        if cls_name in IGNORED_CLUTTER_CLASSES:
                            continue

                        x1 = cx - w / 2
                        y1 = cy - h / 2
                        x2 = cx + w / 2
                        y2 = cy + h / 2

                        det = Detection(
                            bbox=BoundingBox(x1, y1, x2, y2),
                            confidence=score,
                            class_id=getattr(pred, "class_id", 0),
                            class_name=cls_name,
                            camera_id=frame.metadata.camera_id,
                            detector_id=self.metadata.name,
                            frame_timestamp=frame.timestamp,
                        )
                        detections.append(det)

                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                self._detect_count += 1
                self._total_infer_time_ms += elapsed_ms
                return detections
            except Exception as e:
                log.error("roboflow_infer_error", error=str(e))

        return super().detect(frame)
