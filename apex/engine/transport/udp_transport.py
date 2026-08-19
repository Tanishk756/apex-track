"""
UDP Transport Adapter Implementation
====================================
Async non-blocking UDP socket transport supporting unicast and broadcast endpoints.
"""

from __future__ import annotations

import asyncio
import socket
import time
from typing import Optional
import structlog

from apex.engine.transport.transport_base import (
    TransportBase,
    TransportMessage,
    TransportState,
)

log = structlog.get_logger(__name__)


class UDPTransport(TransportBase):
    """Async UDP Socket Transport Adapter."""

    def __init__(self, endpoint_uri: str = "udp://127.0.0.1:14550") -> None:
        super().__init__(endpoint_uri)
        # Parse uri: e.g. udp://127.0.0.1:14550 or udp://0.0.0.0:14550
        uri = endpoint_uri.replace("udp://", "")
        if ":" in uri:
            host, port_str = uri.split(":", 1)
            self.host = host
            self.port = int(port_str)
        else:
            self.host = "127.0.0.1"
            self.port = 14550

        self._socket: Optional[socket.socket] = None
        self._rx_queue: asyncio.Queue[TransportMessage] = asyncio.Queue(maxsize=1000)

    async def connect(self) -> bool:
        self._state = TransportState.CONNECTING
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.setblocking(False)
            # Bind if receiving or listening on wildcard
            if self.host in ("0.0.0.0", "127.0.0.1", "localhost"):
                try:
                    self._socket.bind((self.host, self.port))
                except Exception:
                    pass

            self._state = TransportState.CONNECTED
            log.info("udp_transport_connected", host=self.host, port=self.port)
            return True
        except Exception as exc:
            self._state = TransportState.ERROR
            log.error("udp_transport_connect_failed", error=str(exc))
            return False

    async def disconnect(self) -> None:
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None
        self._state = TransportState.DISCONNECTED
        log.info("udp_transport_disconnected")

    async def send(self, message: TransportMessage) -> bool:
        if not self.is_connected or self._socket is None:
            return False

        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, self._socket.sendto, message.payload, (self.host, self.port))
            self._bytes_sent += len(message.payload)
            return True
        except Exception as exc:
            log.warning("udp_send_failed", error=str(exc))
            return False

    async def receive(self, timeout_s: Optional[float] = None) -> Optional[TransportMessage]:
        if not self.is_connected or self._socket is None:
            return None

        loop = asyncio.get_running_loop()
        try:
            if timeout_s is not None:
                data, addr = await asyncio.wait_for(
                    loop.run_in_executor(None, self._socket.recvfrom, 65535), timeout=timeout_s
                )
            else:
                data, addr = await loop.run_in_executor(None, self._socket.recvfrom, 65535)

            self._bytes_received += len(data)
            return TransportMessage(
                channel="udp_raw",
                payload=data,
                timestamp=time.time(),
                metadata={"sender_addr": addr},
            )
        except asyncio.TimeoutError:
            return None
        except Exception as exc:
            log.debug("udp_recv_exception", error=str(exc))
            return None
