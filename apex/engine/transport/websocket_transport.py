"""
WebSocket Transport Adapter Implementation
===========================================
Async WebSocket client/server transport adapter for streaming C4ISR telemetry and visual reticle frames.
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional
import structlog

from apex.engine.transport.transport_base import (
    TransportBase,
    TransportMessage,
    TransportState,
)

log = structlog.get_logger(__name__)


class WebSocketTransport(TransportBase):
    """Async WebSocket Transport Adapter."""

    def __init__(self, endpoint_uri: str = "ws://127.0.0.1:8000/ws/telemetry") -> None:
        super().__init__(endpoint_uri)
        self._inbound_queue: asyncio.Queue[TransportMessage] = asyncio.Queue(maxsize=1000)
        self._outbound_queue: asyncio.Queue[TransportMessage] = asyncio.Queue(maxsize=1000)

    async def connect(self) -> bool:
        self._state = TransportState.CONNECTING
        # Simulate websocket channel connection setup
        await asyncio.sleep(0.01)
        self._state = TransportState.CONNECTED
        log.info("websocket_transport_connected", endpoint=self.endpoint_uri)
        return True

    async def disconnect(self) -> None:
        self._state = TransportState.DISCONNECTED
        log.info("websocket_transport_disconnected")

    async def send(self, message: TransportMessage) -> bool:
        if not self.is_connected:
            return False

        try:
            if self._outbound_queue.full():
                # Queue backpressure: drop oldest message
                try:
                    self._outbound_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            self._outbound_queue.put_nowait(message)
            self._bytes_sent += len(message.payload)
            return True
        except Exception as exc:
            log.warning("websocket_send_failed", error=str(exc))
            return False

    async def receive(self, timeout_s: Optional[float] = None) -> Optional[TransportMessage]:
        if not self.is_connected:
            return None

        try:
            if timeout_s is not None:
                msg = await asyncio.wait_for(self._inbound_queue.get(), timeout=timeout_s)
            else:
                msg = self._inbound_queue.get_nowait()
            self._bytes_received += len(msg.payload)
            return msg
        except (asyncio.TimeoutError, asyncio.QueueEmpty):
            return None

    def inject_inbound(self, message: TransportMessage) -> None:
        """Helper to inject simulated incoming WebSocket frame for testing."""
        if self._inbound_queue.full():
            try:
                self._inbound_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        self._inbound_queue.put_nowait(message)
