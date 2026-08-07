"""
Contract: Frame
===============
The fundamental unit flowing through the entire pipeline.
A Frame is the output of a CameraPlugin and the input to the Detector/Tracker.

Design decisions:
- data is numpy ndarray; dtype is uint8 (BGR for OpenCV, RGB for models)
- gpu_data is optional — populated when zero-copy GPU buffer is available (NVDEC/CUDA)
- camera_id allows CameraManager to route frames from multiple sources
- sequence_id is monotonically increasing per camera — gaps indicate dropped frames
- Frames are NOT frozen because numpy arrays cannot be hashed.
  Ownership: produced by CameraPlugin, consumed by AdaptiveScheduler. Do NOT mutate
  after handing off to the bus.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class FrameMetadata:
    """Ancillary metadata attached to a captured frame."""

    camera_id: str
    """Unique identifier of the camera that produced this frame."""

    width: int
    """Frame width in pixels."""

    height: int
    """Frame height in pixels."""

    fps: float = 30.0
    """Reported FPS of the source at capture time."""

    source_type: str = "unknown"
    """e.g. 'usb', 'rtsp', 'gstreamer', 'synthetic', 'ros'"""

    encoding: str = "bgr"
    """Color encoding: 'bgr' (OpenCV default) | 'rgb' | 'nv12' | 'yuv420'"""

    hardware_decoded: bool = False
    """True if decoded by NVDEC / V4L2M2M (hardware path)."""

    extra: dict = field(default_factory=dict)
    """Extensible key-value bag for plugin-specific metadata."""


@dataclass
class Frame:
    """
    A single captured video frame plus all associated metadata.

    Memory model:
    - `data` always contains the CPU-side image (may be a pinned-memory
      array when CUDA is active to allow fast H→D transfers).
    - `gpu_data` is an optional CuPy/CUDA tensor for zero-copy GPU pipelines.
      When populated, modules MUST NOT copy to CPU unless necessary.
    """

    data: np.ndarray
    """CPU image array — shape (H, W, C), dtype uint8."""

    timestamp: float = field(default_factory=time.time)
    """Capture timestamp — seconds since Unix epoch (UTC)."""

    sequence_id: int = 0
    """Monotonically increasing frame counter per camera."""

    metadata: FrameMetadata = field(
        default_factory=lambda: FrameMetadata(
            camera_id="unknown", width=0, height=0, fps=0.0, source_type="unknown"
        )
    )

    gpu_data: Optional[object] = None
    """CuPy ndarray or CUDA tensor — None when GPU path not active."""

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.data.shape  # type: ignore[return-value]

    @property
    def hw(self) -> tuple[int, int]:
        """(height, width) tuple."""
        h, w = self.data.shape[:2]
        return h, w

    def __repr__(self) -> str:
        h, w = self.hw
        return (
            f"Frame(cam={self.metadata.camera_id!r}, "
            f"seq={self.sequence_id}, {w}x{h}, "
            f"ts={self.timestamp:.3f})"
        )
