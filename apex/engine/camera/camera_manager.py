"""
Camera Manager
==============
High-level management of all active video capture sources.

Responsibilities:
- Initializes, loads, and manages lifecycle of CameraPlugin instances.
- Monitors camera health, FPS, latency, and frame drops.
- Automatically handles camera reconnection on stream failures.
- Interacts with MessageBus to publish system events (CAMERA_CONNECTED, CAMERA_DISCONNECTED, VIDEO_LOST).
- Integrates FrameSynchronizer for multi-camera stream alignment.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

import structlog

from apex.engine.bus.channels import Ch
from apex.engine.bus.message_bus import MessageBus
from apex.engine.camera.camera_plugin import CameraPlugin
from apex.engine.camera.frame_synchronizer import FrameSynchronizer
from apex.engine.contracts.event import ApexEvent, EventSeverity, EventType
from apex.engine.events.event_engine import EventEngine
from apex.engine.hal.hw_profile import HWProfile
from apex.engine.plugins.plugin_loader import PluginLoader
from apex.engine.plugins.plugin_registry import PluginRegistry

log = structlog.get_logger(__name__)


class CameraManager:
    """
    Manager for single or multi-camera pipelines.

    Usage::
        mgr = CameraManager(hw_profile=hw, event_engine=events)
        await mgr.load_cameras(camera_configs)
        await mgr.start_all()
    """

    def __init__(
        self,
        hw_profile: HWProfile,
        event_engine: Optional[EventEngine] = None,
        bus: Optional[MessageBus] = None,
        sync_tolerance_ms: float = 50.0,
    ) -> None:
        self.hw_profile = hw_profile
        self.events = event_engine
        self.bus = bus or MessageBus.instance()
        self.registry = PluginRegistry.instance()
        self.loader = PluginLoader(hw_profile=hw_profile)

        self._cameras: dict[str, CameraPlugin] = {}
        self._camera_configs: dict[str, dict] = {}
        self._frame_queues: dict[str, asyncio.Queue] = {}
        self._max_queue_size = 30
        self._synchronizer: Optional[FrameSynchronizer] = None
        self._sync_tolerance_ms = sync_tolerance_ms
        self._health_check_task: Optional[asyncio.Task] = None
        self._is_running = False

    async def add_camera(self, camera_id: str, plugin_name: str, config: dict) -> bool:
        """Dynamically load and add a camera plugin."""
        config_copy = dict(config)
        config_copy["camera_id"] = camera_id
        self._camera_configs[camera_id] = config_copy

        log.info("loading_camera_plugin", camera_id=camera_id, plugin=plugin_name)
        plugin = await self.loader.load(plugin_name, config_copy)
        if plugin is None or not isinstance(plugin, CameraPlugin):
            log.error("camera_plugin_load_failed", camera_id=camera_id, plugin=plugin_name)
            if self.events:
                await self.events.emit(
                    ApexEvent(
                        type=EventType.CAMERA_DISCONNECTED,
                        source=f"camera_manager.{camera_id}",
                        severity=EventSeverity.ERROR,
                        payload={"camera_id": camera_id, "plugin": plugin_name},
                    )
                )
            return False

        self._cameras[camera_id] = plugin
        self.registry.register(f"camera.{camera_id}", plugin)

        if self.events:
            await self.events.emit(
                ApexEvent(
                    type=EventType.CAMERA_CONNECTED,
                    source=f"camera_manager.{camera_id}",
                    severity=EventSeverity.INFO,
                    payload={"camera_id": camera_id, "plugin": plugin_name},
                )
            )

        # Update synchronizer if multi-camera
        self._synchronizer = FrameSynchronizer(
            camera_ids=list(self._cameras.keys()),
            tolerance_ms=self._sync_tolerance_ms,
        )
        return True

    async def load_cameras(self, camera_configs: list[dict]) -> int:
        """Load multiple cameras from configuration array."""
        loaded = 0
        for i, cfg in enumerate(camera_configs):
            cid = cfg.get("camera_id", f"cam_{i}")
            pname = cfg.get("plugin", "file_camera")
            ok = await self.add_camera(cid, pname, cfg)
            if ok:
                loaded += 1
        return loaded

    async def start_all(self) -> None:
        """Start streaming on all loaded cameras and initiate health monitoring."""
        if self._is_running:
            return
        self._is_running = True

        for cid, cam in self._cameras.items():
            await cam.start_streaming()

        self._health_check_task = asyncio.create_task(self._monitor_health())
        log.info("all_cameras_started", count=len(self._cameras))

    async def stop_all(self) -> None:
        """Stop streaming on all cameras."""
        self._is_running = False
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None

        for cid, cam in list(self._cameras.items()):
            await cam.stop_streaming()
            await cam.unload()

        self._cameras.clear()
        log.info("all_cameras_stopped")

    async def _monitor_health(self) -> None:
        """Periodic background task checking camera responsiveness and auto-reconnecting."""
        while self._is_running:
            try:
                await asyncio.sleep(2.0)
                for cid, cam in list(self._cameras.items()):
                    if cam.dropped_frames > 50 and cam.fps < 1.0:
                        log.warning("camera_unresponsive_attempting_reconnect", camera_id=cid)
                        if self.events:
                            await self.events.emit(
                                ApexEvent(
                                    type=EventType.VIDEO_LOST,
                                    source=f"camera_manager.{cid}",
                                    severity=EventSeverity.WARNING,
                                    payload={"camera_id": cid},
                                )
                            )
                        # Reconnect logic
                        cfg = self._camera_configs.get(cid, {})
                        if cfg.get("reconnect_on_failure", True):
                            await cam.stop_streaming()
                            await cam._connect()
                            await cam.start_streaming()
                            if self.events:
                                await self.events.emit(
                                    ApexEvent(
                                        type=EventType.VIDEO_RECOVERED,
                                        source=f"camera_manager.{cid}",
                                        severity=EventSeverity.INFO,
                                        payload={"camera_id": cid},
                                    )
                                )

            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.warning("camera_health_monitor_error", error=str(exc))

    def push_frame(self, camera_id: str, frame: Any) -> bool:
        """Push frame into camera bounded ring buffer queue with drop-oldest backpressure."""
        if camera_id not in self._frame_queues:
            self._frame_queues[camera_id] = asyncio.Queue(maxsize=self._max_queue_size)

        queue = self._frame_queues[camera_id]
        if queue.full():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        queue.put_nowait(frame)
        return True

    def get_latest_frame(self, camera_id: str) -> Optional[Any]:
        """Fetch latest frame from camera queue without blocking."""
        queue = self._frame_queues.get(camera_id)
        if queue is None or queue.empty():
            return None
        try:
            return queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def get_camera(self, camera_id: str) -> Optional[CameraPlugin]:
        return self._cameras.get(camera_id)

    @property
    def synchronizer(self) -> Optional[FrameSynchronizer]:
        return self._synchronizer

    @property
    def camera_ids(self) -> list[str]:
        return list(self._cameras.keys())

    def get_stats(self) -> dict[str, dict]:
        """Return FPS and drop metrics per camera."""
        return {
            cid: {
                "fps": cam.fps,
                "total_frames": cam.total_frames,
                "dropped_frames": cam.dropped_frames,
                "is_active": cam.is_active,
            }
            for cid, cam in self._cameras.items()
        }
