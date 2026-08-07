"""
Adaptive Scheduler
==================
Dynamic execution throttling engine for high-speed AI target perception.

Design:
- Controls when neural detection runs vs when optical flow / Kalman propagation handles target tracking.
- Reduces GPU utilization from 100% (60 FPS inference) down to 20% (10-15 FPS inference + 60 FPS Kalman prediction).
- Monitors system resource load (GPU/CPU) via HealthMonitor to automatically scale inference rate.
- Triggers immediate re-detection on optical flow motion spikes or target lock lost events.
"""

from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np
import structlog

from apex.engine.contracts.frame import Frame

log = structlog.get_logger(__name__)


class AdaptiveScheduler:
    """
    Perception scheduling policy engine.

    Determines whether a given video frame requires a full neural detection pass.
    """

    def __init__(
        self,
        base_detection_interval: int = 3,  # Run detection every N frames
        motion_threshold: float = 12.0,      # Optical flow velocity threshold to force detection
        max_gpu_load_percent: float = 85.0,  # GPU load budget
    ) -> None:
        self.base_interval = base_detection_interval
        self.motion_threshold = motion_threshold
        self.max_gpu_load = max_gpu_load_percent

        self._frame_count = 0
        self._current_interval = base_detection_interval
        self._prev_gray_frame: Optional[np.ndarray] = None
        self._force_next_detection = False
        self._last_detection_frame_id = -base_detection_interval

    def should_detect(self, frame: Frame, active_track_count: int = 0, current_gpu_load: float = 0.0) -> bool:
        """
        Evaluate scheduling policy to determine if neural detection should run on this frame.
        """
        self._frame_count += 1

        # 1. Force detection flag (e.g. Target lost event or system state change)
        if self._force_next_detection:
            self._force_next_detection = False
            self._last_detection_frame_id = self._frame_count
            return True

        # 2. Dynamic load-based interval adjustment
        if current_gpu_load > self.max_gpu_load:
            # Back off inference frequency under heavy thermal/GPU stress
            self._current_interval = min(10, self.base_interval * 2)
        else:
            self._current_interval = self.base_interval

        # 3. Always run detection if no active tracks exist
        if active_track_count == 0:
            if (self._frame_count - self._last_detection_frame_id) >= self._current_interval:
                self._last_detection_frame_id = self._frame_count
                return True

        # 4. Optical Flow motion spike detection
        if self._detect_motion_spike(frame.data):
            log.info("motion_spike_detected_forcing_detection", frame_id=self._frame_count)
            self._last_detection_frame_id = self._frame_count
            return True

        # 5. Periodic detection interval check
        if (self._frame_count - self._last_detection_frame_id) >= self._current_interval:
            self._last_detection_frame_id = self._frame_count
            return True

        return False

    def force_detection(self) -> None:
        """Explicitly request immediate detection on the next incoming frame."""
        self._force_next_detection = True

    def _detect_motion_spike(self, image_data: np.ndarray) -> bool:
        """Compute sparse optical flow motion metric between consecutive frames."""
        gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY) if image_data.ndim == 3 else image_data
        # Downsample for ultra-fast motion estimation
        small_gray = cv2.resize(gray, (160, 120), interpolation=cv2.INTER_NEAREST)

        if self._prev_gray_frame is None:
            self._prev_gray_frame = small_gray
            return False

        # Absolute frame difference metric
        diff = cv2.absdiff(small_gray, self._prev_gray_frame)
        self._prev_gray_frame = small_gray
        mean_diff = float(np.mean(diff))

        return mean_diff > self.motion_threshold

    @property
    def current_interval(self) -> int:
        return self._current_interval
