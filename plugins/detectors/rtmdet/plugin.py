"""
RTMDet Detector Plugin
======================
Apache-2.0 Licensed RTMDet Ultra-Fast Object Detector Plugin.
High-throughput alternative to RT-DETR for resource-constrained Jetson / Edge hardware.
"""

from __future__ import annotations

import cv2
import numpy as np
import structlog

from apex.engine.contracts.detection import BoundingBox, Detection
from apex.engine.contracts.frame import Frame
from apex.engine.detector.detector_base import DetectorBase
from apex.engine.plugins.plugin_base import PluginMetadata, PluginType
from plugins.detectors.rtdetr.plugin import COCO_CLASSES

log = structlog.get_logger(__name__)


class RTMDetPlugin(DetectorBase):
    """RTMDet Object Detector Plugin."""

    metadata = PluginMetadata(
        name="rtmdet",
        version="1.0.0",
        plugin_type=PluginType.DETECTOR,
        license="Apache-2.0",
        author="APEX-Track",
        description="RTMDet High-Speed Object Detector Plugin",
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

        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32)
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

        for pred in preds:
            if len(pred) < 6:
                continue
            score = float(pred[4])
            if score < self.conf_threshold:
                continue

            x1 = (pred[0] - pad_w) / scale
            y1 = (pred[1] - pad_h) / scale
            x2 = (pred[2] - pad_w) / scale
            y2 = (pred[3] - pad_h) / scale

            bbox = BoundingBox(x1, y1, x2, y2)
            cls_id = int(pred[5])
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
