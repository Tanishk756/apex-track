"""
Transport Abstraction Base Contract
====================================
Abstract interface for network/communication transports (UDP, TCP, WebSocket, ZeroMQ, ROS2 DDS, SharedMemory).

Design Requirements:
- Technology-agnostic: core engine interacts with TransportBase, not raw socket calls.
- Unified lifecycle: connect(), disconnect(), send(), receive(), health(), reconnect().
- Asynchronous non-blocking message transport.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
import enum
from typing import Any, Optional
import structlog

log = structlog.get_logger(__name__)


class TransportState(enum.Enum):
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    DEGRADED = "DEGRADED"
    ERROR = "ERROR"


@dataclass(slots=True)
class TransportMessage:
    """Standardized transport message payload wrapper."""

    channel: str
    payload: bytes
    timestamp: float
    metadata: dict[str, Any]


class TransportBase(abc.ABC):
    """Abstract base class for all communication transport adapters."""

    def __init__(self, endpoint_uri: str) -> None:
        self.endpoint_uri = endpoint_uri
        self._state = TransportState.DISCONNECTED
        self._bytes_sent = 0
        self._bytes_received = 0

    @property
    def state(self) -> TransportState:
        return self._state

    @property
    def is_connected(self) -> bool:
        return self._state == TransportState.CONNECTED

    @abc.abstractmethod
    async def connect(self) -> bool:
        """Establish transport connection to endpoint."""

    @abc.abstractmethod
    async def disconnect(self) -> None:
        """Close transport connection and clean up resources."""

    @abc.abstractmethod
    async def send(self, message: TransportMessage) -> bool:
        """Send a message across the transport."""

    @abc.abstractmethod
    async def receive(self, timeout_s: Optional[float] = None) -> Optional[TransportMessage]:
        """Receive the next available message from the transport."""

    async def reconnect(self) -> bool:
        """Attempt reconnection to endpoint."""
        await self.disconnect()
        return await self.connect()

    def health(self) -> dict[str, Any]:
        """Return transport health and throughput metrics."""
        return {
            "endpoint": self.endpoint_uri,
            "state": self._state.value,
            "bytes_sent": self._bytes_sent,
            "bytes_received": self._bytes_received,
        }
