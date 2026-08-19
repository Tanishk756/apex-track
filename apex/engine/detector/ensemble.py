"""
Weighted Boxes Fusion (WBF) Perception Ensemble Module
=========================================================
Fuses multi-scale bounding box predictions across neural passes and model ensembles,
maximizing IoU overlap consensus and precision for low-contrast/high-speed tactical assets.
"""

from __future__ import annotations

from typing import List
import numpy as np
import structlog

from apex.engine.contracts.detection import BoundingBox, Detection

log = structlog.get_logger(__name__)


def compute_iou(b1: BoundingBox, b2: BoundingBox) -> float:
    """Compute Intersection over Union (IoU) between two bounding boxes."""
    x1 = max(b1.x1, b2.x1)
    y1 = max(b1.y1, b2.y1)
    x2 = min(b1.x2, b2.x2)
    y2 = min(b1.y2, b2.y2)

    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    b1_area = (b1.x2 - b1.x1) * (b1.y2 - b1.y1)
    b2_area = (b2.x2 - b2.x1) * (b2.y2 - b2.y1)

    union_area = b1_area + b2_area - inter_area
    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)


class WeightedBoxesFusion:
    """Weighted Boxes Fusion engine for tactical multi-pass detection consensus."""

    def __init__(self, iou_thresh: float = 0.55, skip_box_thresh: float = 0.20) -> None:
        self.iou_thresh = iou_thresh
        self.skip_box_thresh = skip_box_thresh

    def fuse_detections(self, detection_lists: List[List[Detection]]) -> List[Detection]:
        """
        Fuses multiple lists of detections (from multiple passes or models)
        using box confidence weighting.
        """
        all_dets = [d for dlist in detection_lists for d in dlist if d.confidence >= self.skip_box_thresh]
        if not all_dets:
            return []

        # Group by class_name
        grouped_by_class: dict[str, List[Detection]] = {}
        for det in all_dets:
            grouped_by_class.setdefault(det.class_name, []).append(det)

        fused_results: List[Detection] = []

        for class_name, dets in grouped_by_class.items():
            # Sort descending by confidence
            dets = sorted(dets, key=lambda x: x.confidence, reverse=True)

            clusters: List[List[Detection]] = []

            for det in dets:
                matched_cluster = None
                for cluster in clusters:
                    # Compute average box in cluster
                    avg_x1 = np.mean([d.bbox.x1 for d in cluster])
                    avg_y1 = np.mean([d.bbox.y1 for d in cluster])
                    avg_x2 = np.mean([d.bbox.x2 for d in cluster])
                    avg_y2 = np.mean([d.bbox.y2 for d in cluster])
                    cluster_bbox = BoundingBox(avg_x1, avg_y1, avg_x2, avg_y2)

                    if compute_iou(det.bbox, cluster_bbox) >= self.iou_thresh:
                        matched_cluster = cluster
                        break

                if matched_cluster is not None:
                    matched_cluster.append(det)
                else:
                    clusters.append([det])

            # Merge clusters using confidence weighting
            for cluster in clusters:
                weights = np.array([d.confidence for d in cluster], dtype=np.float32)
                sum_weights = float(np.sum(weights))

                weighted_x1 = float(np.sum([d.bbox.x1 * w for d, w in zip(cluster, weights)]) / sum_weights)
                weighted_y1 = float(np.sum([d.bbox.y1 * w for d, w in zip(cluster, weights)]) / sum_weights)
                weighted_x2 = float(np.sum([d.bbox.x2 * w for d, w in zip(cluster, weights)]) / sum_weights)
                weighted_y2 = float(np.sum([d.bbox.y2 * w for d, w in zip(cluster, weights)]) / sum_weights)

                # Consensus confidence score
                max_conf = max(d.confidence for d in cluster)
                consensus_bonus = min(0.15, 0.05 * (len(cluster) - 1))
                final_conf = min(1.0, max_conf + consensus_bonus)

                base_det = cluster[0]
                fused_det = Detection(
                    bbox=BoundingBox(weighted_x1, weighted_y1, weighted_x2, weighted_y2),
                    confidence=float(final_conf),
                    class_id=base_det.class_id,
                    class_name=class_name,
                    camera_id=base_det.camera_id,
                    detector_id=f"wbf_ensemble({len(cluster)})",
                    frame_timestamp=base_det.frame_timestamp,
                )
                fused_results.append(fused_det)

        # Cross-class spatial NMS to eliminate overlapping false positive categories on the same object
        fused_sorted = sorted(fused_results, key=lambda x: x.confidence, reverse=True)
        return self.suppress_cross_class_overlaps(fused_sorted, iou_thresh=self.iou_thresh)

    def suppress_cross_class_overlaps(self, detections: List[Detection], iou_thresh: float = 0.35) -> List[Detection]:
        """
        Aggressive Spatial Non-Maximum Suppression.
        - Same class: IoU >= 0.15 suppresses lower-confidence duplicate crops (e.g. hand/torso person duplicates).
        - Cross class: IoU >= 0.35 suppresses lower-confidence false positive categories.
        """
        if not detections:
            return []

        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        kept_dets: List[Detection] = []

        for det in sorted_dets:
            overlap = False
            for kept in kept_dets:
                overlap_limit = 0.15 if det.class_name.lower() == kept.class_name.lower() else iou_thresh
                if compute_iou(det.bbox, kept.bbox) >= overlap_limit:
                    overlap = True
                    break
            if not overlap:
                kept_dets.append(det)

        return kept_dets


class EnsembleDetector:
    """Multi-model ensemble detector combining multiple detector plugins."""

    def __init__(self, detectors: list, weights: list[float] | None = None, iou_threshold: float = 0.5) -> None:
        self.detectors = detectors
        self.weights = weights or [1.0] * len(detectors)
        self.wbf = WeightedBoxesFusion(iou_thresh=iou_threshold)

    def detect(self, frame) -> List[Detection]:
        all_passes = []
        for d in self.detectors:
            try:
                dets = d.detect(frame)
                all_passes.append(dets)
            except Exception as e:
                log.warning("ensemble_sub_detector_failed", error=str(e))
        return self.wbf.fuse_detections(all_passes)
