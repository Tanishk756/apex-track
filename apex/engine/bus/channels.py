"""
Message Bus Channel Definitions
================================
All canonical channel names in one place.
Modules import from here — no magic strings scattered in code.

Channel naming convention:
    /<domain>/<subdomain>/<qualifier>

Wildcards work in subscriptions: '/camera/*' matches all camera channels.
"""

from __future__ import annotations


class Ch:
    """
    Canonical channel name constants.
    Import: from apex.engine.bus.channels import Ch
    Usage:  await bus.publish(Ch.DETECTIONS, detection_array)
    """

    # ── Camera / Video ────────────────────────────────────────────────────────
    FRAME          = "/camera/frame"          # Frame (per camera, per tick)
    FRAME_TEMPLATE = "/camera/{cam_id}/frame" # Per-camera variant

    # ── Detection ────────────────────────────────────────────────────────────
    DETECTIONS     = "/detector/detections"   # DetectionArray

    # ── Tracking ─────────────────────────────────────────────────────────────
    TRACKS         = "/tracker/tracks"        # TrackArray
    TRACK_EVENTS   = "/tracker/events"        # ApexEvent (TARGET_DETECTED, etc.)

    # ── Telemetry ────────────────────────────────────────────────────────────
    TELEMETRY_RAW    = "/telemetry/raw"       # Telemetry (raw MAVLink rate)
    TELEMETRY_FUSED  = "/telemetry/fused"     # Telemetry (frame-aligned)

    # ── Mission ──────────────────────────────────────────────────────────────
    MISSION_EVENTS   = "/mission/events"      # ApexEvent (ZONE_ENTERED, etc.)
    MISSION_COMMANDS = "/mission/commands"    # Command

    # ── Target Lock ──────────────────────────────────────────────────────────
    TARGET_LOCK    = "/target/lock"           # Track (currently locked target)

    # ── Gimbal / PTZ ─────────────────────────────────────────────────────────
    GIMBAL_COMMANDS = "/gimbal/commands"      # GimbalCommand

    # ── System ───────────────────────────────────────────────────────────────
    SYSTEM_STATE   = "/system/state"          # str (SystemState.name)
    SYSTEM_EVENTS  = "/system/events"         # ApexEvent (GPU_OVERLOADED, etc.)
    SYSTEM_COMMANDS = "/system/commands"      # Command (SHUTDOWN, RELOAD_CONFIG)

    # ── Health / Metrics ─────────────────────────────────────────────────────
    HEALTH         = "/health/snapshot"       # dict (periodic health report)
    METRICS        = "/metrics/snapshot"      # dict (performance metrics)

    # ── Recording ────────────────────────────────────────────────────────────
    RECORDING_CTRL = "/recording/control"     # Command (START/STOP)
    RECORDING_FRAME = "/recording/frame"      # Frame (annotated HUD frame)

    @staticmethod
    def camera_frame(cam_id: str) -> str:
        """Per-camera frame channel: /camera/<cam_id>/frame"""
        return f"/camera/{cam_id}/frame"
