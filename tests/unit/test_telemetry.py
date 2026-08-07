"""
Unit Tests — Telemetry & Sensor Fusion (Phase 9)
"""

import pytest
import numpy as np

from apex.engine.contracts.frame import Frame, FrameMetadata
from apex.engine.telemetry.mavlink_receiver import MAVLinkReceiver, UAVTelemetryPacket
from apex.engine.telemetry.telemetry_fuser import TelemetryFuser


class TestMAVLinkReceiver:

    @pytest.mark.asyncio
    async def test_mavlink_receiver_synthetic(self):
        receiver = MAVLinkReceiver()
        await receiver.start()

        telemetry = receiver.get_latest_telemetry()
        assert telemetry.lat == 37.7749
        assert telemetry.lon == -122.4194

        syn_packet = UAVTelemetryPacket(
            timestamp=123.456,
            lat=10.0,
            lon=20.0,
            alt_m=150.0,
            relative_alt_m=100.0,
            roll_deg=1.0,
            pitch_deg=2.0,
            yaw_deg=180.0,
        )
        receiver.inject_synthetic_telemetry(syn_packet)

        updated = receiver.get_latest_telemetry()
        assert updated.lat == 10.0
        assert updated.lon == 20.0

        await receiver.stop()


class TestTelemetryFuser:

    def test_telemetry_fuser_sync(self):
        fuser = TelemetryFuser()

        p1 = UAVTelemetryPacket(timestamp=10.0, lat=1.0, lon=1.0, alt_m=10, relative_alt_m=10, roll_deg=0, pitch_deg=0, yaw_deg=0)
        p2 = UAVTelemetryPacket(timestamp=20.0, lat=2.0, lon=2.0, alt_m=20, relative_alt_m=20, roll_deg=0, pitch_deg=0, yaw_deg=0)
        p3 = UAVTelemetryPacket(timestamp=30.0, lat=3.0, lon=3.0, alt_m=30, relative_alt_m=30, roll_deg=0, pitch_deg=0, yaw_deg=0)

        fuser.add_telemetry(p1)
        fuser.add_telemetry(p2)
        fuser.add_telemetry(p3)

        frame = Frame(
            data=np.zeros((10, 10, 3), dtype=np.uint8),
            metadata=FrameMetadata(camera_id="c0", width=640, height=480),
            timestamp=21.2,
            sequence_id=5,
        )

        synced = fuser.get_synced_telemetry(frame)
        assert synced is not None
        # Should pick p2 (timestamp 20.0 is closest to 21.2)
        assert synced.timestamp == 20.0
        assert synced.lat == 2.0
