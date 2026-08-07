"""
Contract: Telemetry
===================
MAVLink/PX4/ArduPilot telemetry data, time-aligned with video frames.

Design decisions:
- Attitude uses degrees (not radians) for human readability in the GUI.
- GPSPosition uses WGS84 degrees/metres — the universal standard.
- Telemetry is separated from Frame because telemetry arrives at a
  different rate (typically 50–200 Hz) and must be interpolated to
  match frame timestamps in TelemetryFuser.
- is_valid flags allow consumers to gracefully degrade when specific
  sensor data is unavailable (e.g. GPS fix lost, compass uncalibrated).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Attitude:
    """UAV body attitude from IMU/AHRS."""
    roll_deg: float    = 0.0   # positive = right wing down
    pitch_deg: float   = 0.0   # positive = nose up
    yaw_deg: float     = 0.0   # 0=North, 90=East (magnetic heading)
    rollspeed_dps: float  = 0.0
    pitchspeed_dps: float = 0.0
    yawspeed_dps: float   = 0.0
    is_valid: bool     = False


@dataclass(frozen=True, slots=True)
class GPSPosition:
    """GPS fix in WGS84."""
    latitude_deg: float  = 0.0
    longitude_deg: float = 0.0
    altitude_msl_m: float = 0.0   # metres above mean sea level
    altitude_agl_m: float = 0.0   # metres above ground level (if terrain available)
    fix_type: int = 0             # 0=no fix, 2=2D, 3=3D, 4=DGPS, 5=RTK
    hdop: float = 99.9
    vdop: float = 99.9
    satellites: int = 0
    is_valid: bool = False


@dataclass(frozen=True, slots=True)
class Telemetry:
    """
    A complete telemetry snapshot published on /telemetry/raw.
    TelemetryFuser interpolates these onto /telemetry/fused aligned to frame timestamps.
    """

    timestamp: float = field(default_factory=time.time)
    """MAVLink system timestamp (seconds, UTC)."""

    attitude: Attitude = field(default_factory=Attitude)
    gps: GPSPosition   = field(default_factory=GPSPosition)

    # Velocity (NED frame, m/s)
    vn_mps: float = 0.0   # North
    ve_mps: float = 0.0   # East
    vd_mps: float = 0.0   # Down (positive = descending)

    # Airspeed / groundspeed
    groundspeed_mps: float = 0.0
    airspeed_mps: float    = 0.0

    # System
    battery_voltage_v: float  = 0.0
    battery_pct: float        = 0.0
    flight_mode: str          = "UNKNOWN"
    armed: bool               = False
    autopilot_type: str       = "unknown"  # 'ardupilot' | 'px4' | 'custom'

    # Camera platform (gimbal attitude from MAVLink MOUNT_STATUS)
    gimbal_pitch_deg: float = 0.0
    gimbal_roll_deg: float  = 0.0
    gimbal_yaw_deg: float   = 0.0

    is_valid: bool = False
    """False when the telemetry source is disconnected."""

    def __repr__(self) -> str:
        return (
            f"Telemetry(att=({self.attitude.roll_deg:.1f}°,"
            f"{self.attitude.pitch_deg:.1f}°,{self.attitude.yaw_deg:.1f}°) "
            f"alt={self.gps.altitude_agl_m:.1f}m valid={self.is_valid})"
        )
