"""
USB / V4L2 Camera Plugin
========================
OpenCV-based USB / V4L2 camera capture plugin.
Supports local webcam devices, USB capture cards, and standard V4L2 video nodes.
Non-blocking execution using worker threads.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

import cv2
import numpy as np
import structlog

from apex.engine.camera.camera_plugin import CameraPlugin
from apex.engine.plugins.plugin_base import PluginMetadata, PluginType

log = structlog.get_logger(__name__)


class USBCameraPlugin(CameraPlugin):
    """USB/V4L2 Camera plugin using OpenCV VideoCapture."""

    metadata = PluginMetadata(
        name="usb_camera",
        version="1.0.0",
        plugin_type=PluginType.CAMERA,
        license="Apache-2.0",
        author="APEX-Track",
        description="USB / V4L2 OpenCV Camera Plugin",
    )

    def __init__(self, camera_id: str = "usb_cam_0") -> None:
        super().__init__(camera_id=camera_id)
        self._cap: Optional[cv2.VideoCapture] = None
        self._failed_grabs: int = 0

    def _open_capture_sync(self, src: Any) -> bool:
        try:
            device_idx = int(src)
        except ValueError:
            device_idx = src

        log.info("connecting_usb_camera", camera_id=self.camera_id, source=src)
        self._cap = cv2.VideoCapture(device_idx, cv2.CAP_V4L2 if isinstance(device_idx, int) else cv2.CAP_ANY)

        if not self._cap.isOpened():
            self._cap = cv2.VideoCapture(device_idx)

        if not self._cap.isOpened():
            log.error("usb_camera_open_failed", source=src)
            return False

        width = int(self.config.get("width", 1280))
        height = int(self.config.get("height", 720))
        fps = float(self.config.get("fps", 30.0))

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, int(self.config.get("buffer_size", 1)))
        self._failed_grabs = 0

        log.info("usb_camera_connected", camera_id=self.camera_id, width=width, height=height, fps=fps)
        return True

    async def _connect(self) -> bool:
        src = self.config.get("source", "0")
        return await asyncio.to_thread(self._open_capture_sync, src)

    def _grab_frame_sync(self) -> Optional[tuple[np.ndarray, float]]:
        if self._cap is None or not self._cap.isOpened():
            return None

        ts = time.time()
        ret, frame = self._cap.read()
        if not ret or frame is None:
            self._failed_grabs += 1
            if self._failed_grabs >= 5:
                log.warning("usb_camera_stream_failed_reconnecting", camera_id=self.camera_id)
                if self._cap is not None:
                    self._cap.release()
                    self._cap = None
                self._failed_grabs = 0
            return None

        self._failed_grabs = 0
        return frame, ts

    async def _grab_frame(self) -> Optional[tuple[np.ndarray, float]]:
        if self._cap is None or not self._cap.isOpened():
            return None
        return await asyncio.to_thread(self._grab_frame_sync)

    async def _disconnect(self) -> None:
        if self._cap is not None:
            cap = self._cap
            self._cap = None
            await asyncio.to_thread(cap.release)
        self._failed_grabs = 0
        log.info("usb_camera_disconnected", camera_id=self.camera_id)
