"""
RT-DETR Detector Plugin
=======================
Apache-2.0 Licensed Real-Time Detection Transformer Plugin.
Primary commercial-grade detector choice for APEX-Track platform.
"""

from __future__ import annotations

import cv2
import numpy as np
import structlog

from apex.engine.contracts.detection import BoundingBox, Detection
from apex.engine.contracts.frame import Frame
from apex.engine.detector.detector_base import DetectorBase
from apex.engine.plugins.plugin_base import PluginMetadata, PluginType

log = structlog.get_logger(__name__)

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


class RTDETRPlugin(DetectorBase):
    """RT-DETR Transformer Object Detector Plugin."""

    metadata = PluginMetadata(
        name="rtdetr",
        version="1.0.0",
        plugin_type=PluginType.DETECTOR,
        license="Apache-2.0",
        author="APEX-Track",
        description="RT-DETR Real-Time Detection Transformer Detector",
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

        # BGR -> RGB & Normalize to [0, 1]
        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))  # HWC to CHW
        blob = np.expand_dims(blob, axis=0)   # CHW to NCHW
        return blob, scale, (pad_w, pad_h)

    def _postprocess(
        self,
        raw_outputs: Any,
        frame: Frame,
        scale: float,
        pad: tuple[int, int],
    ) -> list[Detection]:
        if isinstance(raw_outputs, (list, tuple)) and len(raw_outputs) >= 2:
            # Standard RT-DETR output: [labels/scores, boxes]
            boxes, scores = raw_outputs[0], raw_outputs[1]
        else:
            return []

        pad_w, pad_h = pad
        detections: list[Detection] = []

        for i in range(len(scores)):
            score = float(scores[i])
            if score < self.conf_threshold:
                continue

            box = boxes[i]
            # Convert normalized center-cx,cy,w,h to frame coords
            cx, cy, bw, bh = box[0] * self.input_size[0], box[1] * self.input_size[1], box[2] * self.input_size[0], box[3] * self.input_size[1]
            x1 = (cx - bw / 2 - pad_w) / scale
            y1 = (cy - bh / 2 - pad_h) / scale
            x2 = (cx + bw / 2 - pad_w) / scale
            y2 = (cy + bh / 2 - pad_h) / scale

            bbox = BoundingBox(x1, y1, x2, y2)
            cls_id = int(box[4]) if len(box) > 4 else 0
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
