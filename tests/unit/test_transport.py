"""
Unit Tests — Transport Abstraction Layer (UDP & WebSocket Transports)
"""

import asyncio
import pytest

from apex.engine.transport.transport_base import TransportMessage, TransportState
from apex.engine.transport.udp_transport import UDPTransport
from apex.engine.transport.websocket_transport import WebSocketTransport


class TestTransportLayer:

    @pytest.mark.asyncio
    async def test_udp_transport_lifecycle(self):
        transport = UDPTransport(endpoint_uri="udp://127.0.0.1:18888")
        assert transport.state == TransportState.DISCONNECTED

        connected = await transport.connect()
        assert connected is True
        assert transport.is_connected is True

        # Send packet
        msg = TransportMessage(channel="telemetry", payload=b"PING", timestamp=100.0, metadata={})
        sent = await transport.send(msg)
        assert sent is True

        health = transport.health()
        assert health["bytes_sent"] == 4
        assert health["state"] == "CONNECTED"

        await transport.disconnect()
        assert transport.is_connected is False

    @pytest.mark.asyncio
    async def test_websocket_transport_lifecycle(self):
        transport = WebSocketTransport(endpoint_uri="ws://127.0.0.1:8000/ws")
        assert transport.state == TransportState.DISCONNECTED

        connected = await transport.connect()
        assert connected is True

        # Inject inbound message
        msg = TransportMessage(channel="hud", payload=b"{\"status\":\"OK\"}", timestamp=100.0, metadata={})
        transport.inject_inbound(msg)

        recvd = await transport.receive(timeout_s=0.5)
        assert recvd is not None
        assert recvd.payload == b"{\"status\":\"OK\"}"

        await transport.disconnect()
        assert transport.is_connected is False
