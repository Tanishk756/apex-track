"""
File & Synthetic Camera Plugin
==============================
Reads video files (MP4/MKV/AVI) or generates synthetic moving target video streams.
Ideal for unit testing, offline playback, bench-testing, and CI pipelines without physical cameras.
"""

from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np
import structlog

from apex.engine.camera.camera_plugin import CameraPlugin
from apex.engine.plugins.plugin_base import PluginMetadata, PluginType

log = structlog.get_logger(__name__)


class FileCameraPlugin(CameraPlugin):
    """File and Synthetic Video Source camera plugin."""

    metadata = PluginMetadata(
        name="file_camera",
        version="1.0.0",
        plugin_type=PluginType.CAMERA,
        license="Apache-2.0",
        author="APEX-Track",
        description="File and Synthetic Video Generator Plugin",
    )

    def __init__(self, camera_id: str = "file_cam_0") -> None:
        super().__init__(camera_id=camera_id)
        self._cap: Optional[cv2.VideoCapture] = None
        self._is_synthetic = False
        self._synth_step = 0

    async def _connect(self) -> bool:
        src = str(self.config.get("source", "synthetic"))
        log.info("connecting_file_camera", camera_id=self.camera_id, source=src)

        if src.lower() in ("synthetic", "synth", "test"):
            self._is_synthetic = True
            log.info("using_synthetic_video_generator", camera_id=self.camera_id)
            return True

        self._is_synthetic = False
        self._cap = cv2.VideoCapture(src)
        if not self._cap.isOpened():
            log.error("file_camera_open_failed", source=src)
            return False

        log.info("file_camera_opened", camera_id=self.camera_id, source=src)
        return True

    async def _grab_frame(self) -> Optional[tuple[np.ndarray, float]]:
        ts = time.time()

        if self._is_synthetic:
            return self._generate_synthetic_frame(), ts

        if self._cap is None or not self._cap.isOpened():
            return None

        ret, frame = self._cap.read()
        if not ret or frame is None:
            loop_video = self.config.get("loop", True)
            if loop_video:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()
                if not ret or frame is None:
                    return None
            else:
                return None

        return frame, ts

    def _generate_synthetic_frame(self) -> np.ndarray:
        """Generate synthetic test frame with a moving target circle."""
        width = int(self.config.get("width", 1280))
        height = int(self.config.get("height", 720))

        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Background subtle grid pattern
        cv2.grid = True
        grid_size = 80
        for x in range(0, width, grid_size):
            cv2.line(frame, (x, 0), (x, height), (30, 30, 30), 1)
        for y in range(0, height, grid_size):
            cv2.line(frame, (0, y), (width, y), (30, 30, 30), 1)

        # Calculate moving target trajectory (sine wave bounce)
        self._synth_step += 1
        t = self._synth_step * 0.05
        cx = int((width / 2) + (width / 3) * np.sin(t))
        cy = int((height / 2) + (height / 4) * np.cos(t * 1.5))
        radius = 25

        # Draw target vehicle box & circle
        cv2.rectangle(frame, (cx - radius, cy - radius), (cx + radius, cy + radius), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        # Add timestamp HUD text
        cv2.putText(
            frame,
            f"SYNTHETIC STREAM - FRAME {self._synth_step}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
        return frame

    async def _disconnect(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        log.info("file_camera_disconnected", camera_id=self.camera_id)
