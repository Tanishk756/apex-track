"""
RTSP / IP Camera Plugin
=======================
Ultra-low latency RTSP/HTTP/UDP network video stream capture plugin.
Supports GStreamer pipelines (with hardware decode NVDEC/V4L2) and FFmpeg/OpenCV fallbacks.
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


class RTSPCameraPlugin(CameraPlugin):
    """RTSP/IP Network camera plugin."""

    metadata = PluginMetadata(
        name="rtsp_camera",
        version="1.0.0",
        plugin_type=PluginType.CAMERA,
        license="Apache-2.0",
        author="APEX-Track",
        description="Low-latency RTSP/IP Network Stream Plugin",
    )

    def __init__(self, camera_id: str = "rtsp_cam_0") -> None:
        super().__init__(camera_id=camera_id)
        self._cap: Optional[cv2.VideoCapture] = None

    async def _connect(self) -> bool:
        url = str(self.config.get("source", "rtsp://127.0.0.1:8554/live"))
        hw_decode = self.config.get("hw_decode", True)
        latency_ms = int(self.config.get("latency_ms", 100))

        log.info("connecting_rtsp_camera", camera_id=self.camera_id, url=url)

        # Attempt low-latency GStreamer pipeline string if enabled
        pipeline = (
            f"rtspsrc location={url} latency={latency_ms} ! "
            f"rtph264depay ! h264parse ! "
            f"{'nvdec ! gpubufferimport ! ' if hw_decode else ''}decodebin ! "
            f"videoconvert ! appsink drop=true sync=false"
        )

        # Try GStreamer backend first
        self._cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

        if not self._cap.isOpened():
            log.warning("gstreamer_rtsp_failed_fallback_ffmpeg", url=url)
            # Fallback to standard OpenCV / FFmpeg RTSP
            # Configure RTSP transport via environment/OpenCV flags
            self._cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

        if not self._cap.isOpened():
            log.error("rtsp_camera_open_failed", url=url)
            return False

        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep buffer minimal for zero latency
        log.info("rtsp_camera_connected", camera_id=self.camera_id)
        return True

    async def _grab_frame(self) -> Optional[tuple[np.ndarray, float]]:
        if self._cap is None or not self._cap.isOpened():
            return None

        ts = time.time()
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return None

        return frame, ts

    async def _disconnect(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        log.info("rtsp_camera_disconnected", camera_id=self.camera_id)
