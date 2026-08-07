"""
Telemetry Frame Fuser
=====================
Sliding time-window buffer for exact temporal synchronization between video frames and UAV flight telemetry.
"""

from __future__ import annotations

import collections
from typing import Optional

from apex.engine.contracts.frame import Frame
from apex.engine.telemetry.mavlink_receiver import UAVTelemetryPacket


class TelemetryFuser:
    """Synchronizes video frame timestamps with UAV telemetry history."""

    def __init__(self, max_buffer_sec: float = 5.0) -> None:
        self.max_buffer_sec = max_buffer_sec
        self._buffer: collections.deque[UAVTelemetryPacket] = collections.deque(maxlen=500)

    def add_telemetry(self, packet: UAVTelemetryPacket) -> None:
        self._buffer.append(packet)

    def get_synced_telemetry(self, frame: Frame) -> Optional[UAVTelemetryPacket]:
        """Find closest telemetry packet in time buffer for given frame timestamp."""
        if not self._buffer:
            return None

        target_ts = frame.timestamp
        closest_pkt = min(self._buffer, key=lambda p: abs(p.timestamp - target_ts))
        return closest_pkt
