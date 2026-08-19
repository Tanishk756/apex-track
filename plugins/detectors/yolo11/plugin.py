"""
YOLOv11 Detector Plugin
=======================
AGPL-3.0 Licensed YOLOv11 Object Detector Plugin.
Triggers AGPL license verification gate in PluginLoader before initialization.
"""

from __future__ import annotations

from typing import Any
import cv2
import numpy as np
import structlog

from apex.engine.contracts.detection import BoundingBox, Detection
from apex.engine.contracts.frame import Frame
from apex.engine.detector.detector_base import DetectorBase
from apex.engine.plugins.plugin_base import PluginMetadata, PluginType
from plugins.detectors.rtdetr.plugin import COCO_CLASSES

log = structlog.get_logger(__name__)


class YOLO11Plugin(DetectorBase):
    """YOLOv11 Object Detector Plugin (AGPL-3.0)."""

    metadata = PluginMetadata(
        name="yolo11",
        version="1.0.0",
        plugin_type=PluginType.DETECTOR,
        license="AGPL-3.0",
        author="Ultralytics / APEX-Track",
        description="YOLOv11 Object Detector Plugin (Requires AGPL Acceptance)",
        is_agpl=True,
    )

    def __init__(self) -> None:
        super().__init__()
        self.class_names = COCO_CLASSES

    def _preprocess(self, frame: Frame) -> tuple[np.ndarray, float, tuple[int, int]]:
        img = frame.data
        h, w = img.shape[:2]
        target_w, target_h = self.input_size

        scale = min(target_w / w, target_h / h)
        nw, nh = int(w * scale), int(h * scale)

        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)

        pad_w = (target_w - nw) // 2
        pad_h = (target_h - nh) // 2
        canvas[pad_h:pad_h + nh, pad_w:pad_w + nw] = resized

        blob = canvas.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)
        return blob, scale, (pad_w, pad_h)

    def _postprocess(
        self,
        raw_outputs: Any,
        frame: Frame,
        scale: float,
        pad: tuple[int, int],
    ) -> list[Detection]:
        if not raw_outputs:
            return []

        pad_w, pad_h = pad
        detections: list[Detection] = []
        preds = raw_outputs[0]

        # YOLOv11 output shape: (1, 84, 8400)
        if preds.ndim == 3:
            preds = preds[0].T  # transpose to (8400, 84)

        for pred in preds:
            scores = pred[4:]
            cls_id = int(np.argmax(scores))
            score = float(scores[cls_id])

            if score < self.conf_threshold:
                continue

            cx, cy, bw, bh = pred[0], pred[1], pred[2], pred[3]
            x1 = (cx - bw / 2 - pad_w) / scale
            y1 = (cy - bh / 2 - pad_h) / scale
            x2 = (cx + bw / 2 - pad_w) / scale
            y2 = (cy + bh / 2 - pad_h) / scale

            bbox = BoundingBox(x1, y1, x2, y2)
            cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}"

            det = Detection(
                bbox=bbox,
                confidence=score,
                class_id=cls_id,
                class_name=cls_name,
                camera_id=frame.metadata.camera_id,
                detector_id=self.metadata.name,
                frame_timestamp=frame.timestamp,
            )
            detections.append(det)

        return detections
