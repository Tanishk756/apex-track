"""
Camera Plugin Base Class
========================
Abstract base class for all camera source plugins (USB, RTSP, GStreamer, File, Synthetic).

Design:
- Inherits PluginBase to integrate with PluginLoader and PluginRegistry.
- Async frame stream via read_frame() or async generator stream_frames().
- Publishes captured frames to MessageBus channel Ch.camera_frame(camera_id).
- Maintained background capture loop to prevent blocking pipeline ingestion.
"""

from __future__ import annotations

import abc
import asyncio
import time
from typing import Any, AsyncGenerator, Optional

import structlog

from apex.engine.bus.channels import Ch
from apex.engine.bus.message_bus import MessageBus
from apex.engine.contracts.frame import Frame, FrameMetadata
from apex.engine.plugins.plugin_base import PluginBase, PluginMetadata, PluginStatus, PluginType

log = structlog.get_logger(__name__)


class CameraPlugin(PluginBase, abc.ABC):
    """
    Abstract base for camera capture plugins.

    Concrete implementations must implement:
        _connect() -> bool
        _grab_frame() -> Optional[tuple[np.ndarray, float]]  # (image_data, capture_timestamp)
        _disconnect() -> None
    """

    def __init__(self, camera_id: str = "cam_0") -> None:
        super().__init__()
        self.camera_id = camera_id
        self._bus = MessageBus.instance()
        self._streaming = False
        self._capture_task: Optional[asyncio.Task] = None
        self._frame_count = 0
        self._drop_count = 0
        self._start_time = 0.0
        self._last_fps = 0.0
        self._fps_calc_time = 0.0
        self._fps_frame_count = 0

    @abc.abstractmethod
    async def _connect(self) -> bool:
        """Initialize and connect to camera device/stream."""

    @abc.abstractmethod
    async def _grab_frame(self) -> Optional[tuple[Any, float]]:
        """Grab raw image array and timestamp from hardware/stream."""

    @abc.abstractmethod
    async def _disconnect(self) -> None:
        """Close hardware connection/stream."""

    async def load(self, config: dict, hw_profile: Any) -> None:
        """Initialize camera config and attempt connection."""
        self.camera_id = config.get("camera_id", self.camera_id)
        self.config = config
        self.hw_profile = hw_profile

        connected = await self._connect()
        if not connected:
            self._set_status(PluginStatus.ERROR)
            self._record_error(f"Failed to connect to camera {self.camera_id}")
            raise RuntimeError(f"Camera connection failed for {self.camera_id}")

        self._set_status(PluginStatus.ACTIVE)

    async def start_streaming(self) -> None:
        """Start async background capture loop."""
        if self._streaming:
            return
        self._streaming = True
        self._start_time = time.time()
        self._fps_calc_time = time.time()
        self._fps_frame_count = 0
        self._capture_task = asyncio.create_task(self._capture_loop())
        log.info("camera_streaming_started", camera_id=self.camera_id)

    async def stop_streaming(self) -> None:
        """Stop capture loop."""
        self._streaming = False
        if self._capture_task:
            self._capture_task.cancel()
            try:
                await self._capture_task
            except asyncio.CancelledError:
                pass
            self._capture_task = None
        log.info("camera_streaming_stopped", camera_id=self.camera_id)

    async def _capture_loop(self) -> None:
        """Background loop continuously capturing and publishing frames."""
        target_fps = float(self.config.get("fps", 30.0))
        frame_interval = 1.0 / target_fps if target_fps > 0 else 0.033

        while self._streaming:
            t0 = time.time()
            try:
                res = await self._grab_frame()
                if res is not None:
                    img_data, ts = res
                    self._frame_count += 1
                    self._fps_frame_count += 1

                    # Create Frame contract
                    h, w = img_data.shape[:2]
                    channels = img_data.shape[2] if img_data.ndim == 3 else 1
                    meta = FrameMetadata(
                        camera_id=self.camera_id,
                        width=w,
                        height=h,
                        fps=float(self.config.get("fps", 30.0)),
                        source_type=str(self.config.get("plugin", "unknown")),
                        extra={
                            "exposure_ms": float(self.config.get("exposure_ms", 0.0)),
                            "gain": float(self.config.get("gain", 0.0)),
                            "channels": channels,
                        },
                    )
                    frame = Frame(
                        data=img_data,
                        metadata=meta,
                        timestamp=ts,
                        sequence_id=self._frame_count,
                    )

                    # Publish frame to camera-specific channel and general channel
                    await self._bus.publish(Ch.camera_frame(self.camera_id), frame)
                    await self._bus.publish(Ch.FRAME, frame)
                else:
                    self._drop_count += 1
                    await asyncio.sleep(0.005)

                # Calculate FPS periodically
                now = time.time()
                elapsed = now - self._fps_calc_time
                if elapsed >= 1.0:
                    self._last_fps = self._fps_frame_count / elapsed
                    self._fps_calc_time = now
                    self._fps_frame_count = 0

                # Frame rate pacing
                elapsed_frame = time.time() - t0
                sleep_time = frame_interval - elapsed_frame
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._drop_count += 1
                log.warning("camera_frame_grab_error", camera_id=self.camera_id, error=str(exc))
                await asyncio.sleep(0.01)

    async def unload(self) -> None:
        """Stop stream and disconnect camera hardware."""
        await self.stop_streaming()
        await self._disconnect()
        self._set_status(PluginStatus.UNLOADED)

    @property
    def fps(self) -> float:
        return self._last_fps

    @property
    def total_frames(self) -> int:
        return self._frame_count

    @property
    def dropped_frames(self) -> int:
        return self._drop_count
