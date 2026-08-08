"""
RTSP / HTTP IP Camera Plugin
============================
Ultra-low latency RTSP/HTTP network video stream capture plugin.
Supports HTTP MJPEG streams (DroidCam, IP Webcam), GStreamer pipelines, and FFmpeg/OpenCV fallbacks.
Non-blocking execution using worker threads.
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional

import cv2
import numpy as np
import structlog

from apex.engine.camera.camera_plugin import CameraPlugin
from apex.engine.plugins.plugin_base import PluginMetadata, PluginType

log = structlog.get_logger(__name__)


class RTSPCameraPlugin(CameraPlugin):
    """RTSP/HTTP IP Network camera plugin."""

    metadata = PluginMetadata(
        name="rtsp_camera",
        version="1.0.0",
        plugin_type=PluginType.CAMERA,
        license="Apache-2.0",
        author="APEX-Track",
        description="Low-latency RTSP/HTTP Network Stream Plugin",
    )

    def __init__(self, camera_id: str = "rtsp_cam_0") -> None:
        super().__init__(camera_id=camera_id)
        self._cap: Optional[cv2.VideoCapture] = None

    def _open_capture_sync(self, url: str, hw_decode: bool, latency_ms: int) -> bool:
        """Synchronous backend capture opening executed in thread."""
        log.info("connecting_rtsp_camera", camera_id=self.camera_id, url=url)

        # For HTTP/HTTPS streams (DroidCam, IP Webcam), connect directly via OpenCV FFmpeg/MJPEG backend
        if url.startswith("http://") or url.startswith("https://"):
            self._cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            if not self._cap.isOpened():
                self._cap = cv2.VideoCapture(url)
        else:
            # For RTSP streams, attempt low-latency GStreamer pipeline first
            pipeline = (
                f"rtspsrc location={url} latency={latency_ms} ! "
                f"rtph264depay ! h264parse ! "
                f"{'nvdec ! gpubufferimport ! ' if hw_decode else ''}decodebin ! "
                f"videoconvert ! appsink drop=true sync=false"
            )
            self._cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if not self._cap.isOpened():
                log.warning("gstreamer_rtsp_failed_fallback_ffmpeg", url=url)
                self._cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

        if self._cap is not None and self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            log.info("rtsp_camera_connected", camera_id=self.camera_id)
            return True

        log.error("rtsp_camera_open_failed", url=url)
        return False

    async def _connect(self) -> bool:
        url = str(self.config.get("source", "rtsp://127.0.0.1:8554/live"))
        hw_decode = self.config.get("hw_decode", True)
        latency_ms = int(self.config.get("latency_ms", 100))

        # Offload blocking connection to thread worker
        return await asyncio.to_thread(self._open_capture_sync, url, hw_decode, latency_ms)

    def _grab_frame_sync(self) -> Optional[tuple[np.ndarray, float]]:
        if self._cap is None or not self._cap.isOpened():
            return None
        ts = time.time()
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return None
        return frame, ts


    async def _grab_frame(self) -> Optional[tuple[np.ndarray, float]]:
        if self._cap is None or not self._cap.isOpened():
            return None
        # Offload blocking frame read to thread worker
        return await asyncio.to_thread(self._grab_frame_sync)

    async def _disconnect(self) -> None:
        if self._cap is not None:
            cap = self._cap
            self._cap = None
            await asyncio.to_thread(cap.release)
        log.info("rtsp_camera_disconnected", camera_id=self.camera_id)
