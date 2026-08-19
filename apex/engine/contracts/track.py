"""
Contract: Track
===============
A tracked object with persistent identity across frames.

Design decisions:
- TrackState is an explicit state machine enum — every subscriber knows
  exactly what phase a track is in without inspecting raw counters.
- Pixel coords (bbox) and world coords (world_point) are kept parallel.
- velocity_px is in pixels/second; velocity_world is in m/s.
- predicted_bbox is the UKF-predicted position for the *next* frame —
  GimbalController uses this for predictive lead targeting.
- Track is frozen (immutable). The Tracker creates a new Track object
  each frame rather than mutating in place — this makes the bus safe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from apex.engine.contracts.detection import BoundingBox


class TrackState(Enum):
    """
    Lifecycle states of a tracked object.

    TENTATIVE   — seen for fewer than min_hits frames; not yet confirmed
    CONFIRMED   — stably tracked; published to downstream consumers
    COASTING    — detector missed it; UKF predicts position (occlusion/FPV dropout)
    LOST        — coasted too long; track is being reacquired via Re-ID
    DELETED     — expired; will be removed from TargetDatabase next cycle
    """
    TENTATIVE = auto()
    CONFIRMED = auto()
    COASTING  = auto()
    LOST      = auto()
    DELETED   = auto()


@dataclass(frozen=True, slots=True)
class Track:
    """
    A single tracked target, produced by the Tracker and published on /tracker/tracks.
    Carries both pixel-space and world-space coordinates when available.
    """

    track_id: int
    """Globally unique, monotonically assigned track identifier."""

    state: TrackState
    """Current lifecycle state of this track."""

    bbox: BoundingBox
    """Current bounding box in pixel coordinates (xyxy)."""

    predicted_bbox: BoundingBox
    """UKF-predicted bbox for the *next* frame — used by GimbalController."""

    confidence: float
    """Fused detection+track confidence in [0.0, 1.0]."""

    class_id: int
    class_name: str

    frame_timestamp: float
    """Timestamp of the frame this track was updated from."""

    camera_id: str = "unknown"

    # Motion state (pixel space)
    velocity_px: tuple[float, float] = (0.0, 0.0)
    """(vx, vy) pixels/second."""

    acceleration_px: tuple[float, float] = (0.0, 0.0)
    """(ax, ay) pixels/second²."""

    # Motion state (world space) — populated by WorldCoordinateSystem
    world_point: Optional[tuple[float, float, float]] = None
    """(lat_deg, lon_deg, alt_m) — None until geolocation runs."""

    velocity_world: Optional[tuple[float, float]] = None
    """(north_mps, east_mps) ground speed — None until geolocation runs."""

    speed_kmh: Optional[float] = None
    """Estimated ground speed in km/h — None until geolocation runs."""

    heading_deg: Optional[float] = None
    """True north heading in degrees [0, 360) — None until world coords available."""

    # Track history
    age_frames: int = 0
    """Number of frames since this track was first created."""

    hits: int = 0
    """Number of frames where a detection was directly associated."""

    misses: int = 0
    """Consecutive frames without a detection (currently coasting)."""

    # Re-ID
    embedding: Optional[bytes] = None
    """Latest Re-ID feature embedding — used for reacquisition after LOST."""

    segmentation_mask: Optional[tuple[tuple[float, float], ...]] = None
    """Polygon segmentation mask vertices normalized in [0, 1] relative to bounding box."""

    def is_active(self) -> bool:
        return self.state in (TrackState.TENTATIVE, TrackState.CONFIRMED, TrackState.COASTING)

    def __repr__(self) -> str:
        return (
            f"Track(id={self.track_id}, {self.class_name!r}, "
            f"{self.state.name}, conf={self.confidence:.2f}, "
            f"age={self.age_frames})"
        )


@dataclass(frozen=True, slots=True)
class TrackArray:
    """
    The complete set of active tracks for one frame cycle,
    published on channel /tracker/tracks.
    """

    tracks: tuple[Track, ...]
    frame_timestamp: float
    camera_id: str
    tracker_id: str
    tracking_latency_ms: float = 0.0

    def __len__(self) -> int:
        return len(self.tracks)

    def confirmed(self) -> tuple[Track, ...]:
        return tuple(t for t in self.tracks if t.state == TrackState.CONFIRMED)

    def by_id(self, track_id: int) -> Optional[Track]:
        return next((t for t in self.tracks if t.track_id == track_id), None)
