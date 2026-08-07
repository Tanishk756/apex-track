"""
Ensemble Detector
=================
Combines bounding box predictions from multiple detector plugins using Weighted NMS (W-NMS) / Soft-NMS.
Enables high-confidence fusion across multi-model ensembles (e.g., RT-DETR + RTMDet).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import structlog

from apex.engine.contracts.detection import BoundingBox, Detection
from apex.engine.contracts.frame import Frame
from apex.engine.detector.detector_base import DetectorBase

log = structlog.get_logger(__name__)


class EnsembleDetector:
    """Multi-model weighted NMS box fusion ensemble."""

    def __init__(
        self,
        detectors: Sequence[DetectorBase],
        weights: Sequence[float] | None = None,
        iou_threshold: float = 0.5,
        conf_threshold: float = 0.4,
    ) -> None:
        self.detectors = list(detectors)
        self.weights = list(weights) if weights else [1.0] * len(detectors)
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold

    def detect(self, frame: Frame) -> list[Detection]:
        """Run all constituent detectors and merge predictions via Weighted NMS."""
        all_detections: list[tuple[Detection, float]] = []

        for det_engine, weight in zip(self.detectors, self.weights):
            dets = det_engine.detect(frame)
            for d in dets:
                all_detections.append((d, weight))

        if not all_detections:
            return []

        return self._weighted_nms(all_detections)

    def _weighted_nms(self, weighted_dets: list[tuple[Detection, float]]) -> list[Detection]:
        """Apply weighted box fusion & NMS."""
        if not weighted_dets:
            return []

        # Group detections by class_id
        by_class: dict[int, list[tuple[Detection, float]]] = {}
        for det, w in weighted_dets:
            by_class.setdefault(det.class_id, []).append((det, w))

        fused_detections: list[Detection] = []

        for cls_id, group in by_class.items():
            # Sort by weighted confidence score descending
            group.sort(key=lambda x: x[0].confidence * x[1], reverse=True)

            used = [False] * len(group)
            for i in range(len(group)):
                if used[i]:
                    continue

                det_i, w_i = group[i]
                box_cluster = [(det_i, w_i)]
                used[i] = True

                for j in range(i + 1, len(group)):
                    if used[j]:
                        continue
                    det_j, w_j = group[j]
                    iou = det_i.bbox.iou(det_j.bbox)
                    if iou >= self.iou_threshold:
                        box_cluster.append((det_j, w_j))
                        used[j] = True

                # Compute weighted average bounding box coordinates
                total_w = sum(d.confidence * w for d, w in box_cluster)
                avg_x1 = sum(d.bbox.x1 * d.confidence * w for d, w in box_cluster) / total_w
                avg_y1 = sum(d.bbox.y1 * d.confidence * w for d, w in box_cluster) / total_w
                avg_x2 = sum(d.bbox.x2 * d.confidence * w for d, w in box_cluster) / total_w
                avg_y2 = sum(d.bbox.y2 * d.confidence * w for d, w in box_cluster) / total_w

                # Average confidence weighted across cluster
                avg_conf = min(1.0, total_w / sum(w for _, w in box_cluster))

                cam_id = box_cluster[0][0].camera_id
                ts = box_cluster[0][0].frame_timestamp

                fused_det = Detection(
                    bbox=BoundingBox(avg_x1, avg_y1, avg_x2, avg_y2),
                    confidence=avg_conf,
                    class_id=cls_id,
                    class_name=det_i.class_name,
                    camera_id=cam_id,
                    detector_id="ensemble",
                    frame_timestamp=ts,
                )
                fused_detections.append(fused_det)

        return fused_detections
