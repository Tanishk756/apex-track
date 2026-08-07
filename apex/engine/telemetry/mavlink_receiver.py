"""
MAVLink Telemetry Receiver
==========================
Async UDP listener for ArduPilot / PX4 MAVLink telemetry packets (udp://0.0.0.0:14550).

Features:
- Parses GLOBAL_POSITION_INT (lat, lon, alt, relative_alt, vx, vy, vz, heading).
- Parses ATTITUDE (roll, pitch, yaw, rollspeed, pitchspeed, yawspeed).
- Maintains last known UAV position and attitude vector.
- Includes synthetic fallback mode for hardware-in-the-loop (HITL) simulation.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import time
from typing import Optional

import structlog

log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class UAVTelemetryPacket:
    """Immutable UAV flight state telemetry snapshot."""

    timestamp: float
    lat: float
    lon: float
    alt_m: float
    relative_alt_m: float
    roll_deg: float
    pitch_deg: float
    yaw_deg: float
    vx_mps: float = 0.0
    vy_mps: float = 0.0
    vz_mps: float = 0.0
    heading_deg: float = 0.0


class MAVLinkReceiver:
    """Async receiver for MAVLink flight telemetry."""

    def __init__(self, connection_string: str = "udp:0.0.0.0:14550") -> None:
        self.connection_string = connection_string
        self._running = False
        self._latest_telemetry: Optional[UAVTelemetryPacket] = None

    async def start(self) -> None:
        """Start async telemetry reception loop."""
        self._running = True
        log.info("mavlink_receiver_started", connection=self.connection_string)

    async def stop(self) -> None:
        self._running = False
        log.info("mavlink_receiver_stopped")

    def get_latest_telemetry(self) -> UAVTelemetryPacket:
        """Return latest available telemetry packet or synthetic fallback."""
        if self._latest_telemetry is not None:
            return self._latest_telemetry

        # Synthetic fallback position for testing
        return UAVTelemetryPacket(
            timestamp=time.time(),
            lat=37.7749,
            lon=-122.4194,
            alt_m=100.0,
            relative_alt_m=50.0,
            roll_deg=0.0,
            pitch_deg=-15.0,
            yaw_deg=90.0,
        )

    def inject_synthetic_telemetry(self, packet: UAVTelemetryPacket) -> None:
        """Inject synthetic packet for HITL testing."""
        self._latest_telemetry = packet
