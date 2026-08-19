"""
Contract: Detection
===================
The output of any DetectorPlugin for a single object in a single frame.

Design decisions:
- BoundingBox uses xyxy format internally (x1,y1,x2,y2) — most efficient
  for IoU computation and NMS. Helpers convert to/from xywh and cxcywh.
- confidence is always in [0.0, 1.0]
- class_id refers to the detector's class list; class_name is resolved
  by the ModelManager from the registry at runtime.
- world_point is None until the WorldCoordinateSystem processes the detection.
- Detection is frozen — immutable after creation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """
    Axis-aligned bounding box in pixel coordinates.
    All values are floats to support sub-pixel precision from model output.
    """

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    def to_xywh(self) -> tuple[float, float, float, float]:
        """Returns (x_top_left, y_top_left, width, height)."""
        return self.x1, self.y1, self.width, self.height

    def to_cxcywh(self) -> tuple[float, float, float, float]:
        """Returns (center_x, center_y, width, height)."""
        return self.cx, self.cy, self.width, self.height

    def iou(self, other: "BoundingBox") -> float:
        """Intersection-over-Union with another box."""
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    @classmethod
    def from_xywh(cls, x: float, y: float, w: float, h: float) -> "BoundingBox":
        return cls(x1=x, y1=y, x2=x + w, y2=y + h)

    @classmethod
    def from_cxcywh(cls, cx: float, cy: float, w: float, h: float) -> "BoundingBox":
        return cls(x1=cx - w / 2, y1=cy - h / 2, x2=cx + w / 2, y2=cy + h / 2)


@dataclass(frozen=True, slots=True)
class Detection:
    """
    A single detected object, output of a DetectorPlugin.
    """

    bbox: BoundingBox
    """Bounding box in pixel space (xyxy, image coordinates)."""

    confidence: float
    """Detection confidence score in [0.0, 1.0]."""

    class_id: int
    """Integer class index as output by the model."""

    class_name: str
    """Human-readable class label resolved from model registry."""

    frame_timestamp: float = 0.0
    """Timestamp of the frame this detection came from (seconds, UTC)."""

    camera_id: str = "unknown"
    """Camera that produced the source frame."""

    detector_id: str = "unknown"
    """Identifier of the detector plugin that produced this detection."""

    world_point: Optional[tuple[float, float, float]] = None
    """(lat_deg, lon_deg, alt_m) — populated by WorldCoordinateSystem, else None."""

    embedding: Optional[bytes] = None
    """Re-ID feature embedding (bytes-serialized float32 vector) for appearance matching."""

    segmentation_mask: Optional[tuple[tuple[float, float], ...]] = None
    """Polygon segmentation mask vertices normalized in [0, 1] relative to bounding box."""

    def __repr__(self) -> str:
        return (
            f"Detection({self.class_name!r} {self.confidence:.2f} "
            f"@ [{self.bbox.x1:.0f},{self.bbox.y1:.0f},"
            f"{self.bbox.x2:.0f},{self.bbox.y2:.0f}])"
        )


@dataclass(frozen=True, slots=True)
class DetectionArray:
    """
    The full set of detections for one frame, as published on the message bus.
    """

    detections: tuple[Detection, ...]
    frame_timestamp: float
    camera_id: str
    detector_id: str
    inference_latency_ms: float = 0.0
    """Wall-clock inference time in milliseconds for diagnostics."""

    def __len__(self) -> int:
        return len(self.detections)

    def filter_by_class(self, *class_names: str) -> "DetectionArray":
        filtered = tuple(d for d in self.detections if d.class_name in class_names)
        return DetectionArray(
            detections=filtered,
            frame_timestamp=self.frame_timestamp,
            camera_id=self.camera_id,
            detector_id=self.detector_id,
            inference_latency_ms=self.inference_latency_ms,
        )

    def filter_by_confidence(self, min_conf: float) -> "DetectionArray":
        filtered = tuple(d for d in self.detections if d.confidence >= min_conf)
        return DetectionArray(
            detections=filtered,
            frame_timestamp=self.frame_timestamp,
            camera_id=self.camera_id,
            detector_id=self.detector_id,
            inference_latency_ms=self.inference_latency_ms,
        )
