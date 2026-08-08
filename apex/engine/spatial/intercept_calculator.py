"""
3D Target Intercept Bearing & Slant Range Calculator
=====================================================
Computes 3D target intercept parameters for optical/radar tracking:
- Target Azimuth Angle (0° to 360° True North)
- Target Elevation Angle (-90° to +90°)
- Slant Range (meters)
- Intercept Lead Vectors & Time-To-Impact (TTI)
"""

from __future__ import annotations

import math
from typing import Dict, List, Any


class InterceptCalculator:
    """Computes 3D fire-control intercept geometry and tracking vectors."""

    def __init__(self, fov_h_deg: float = 60.0, fov_v_deg: float = 35.0) -> None:
        self.fov_h_deg = fov_h_deg
        self.fov_v_deg = fov_v_deg

    def compute_intercept(self, track: Any, frame_w: int = 1280, frame_h: int = 720, uav_alt_m: float = 120.0) -> Dict[str, Any]:
        """
        Calculates 3D intercept vector parameters from pixel coordinates & telemetry.
        """
        cx = track.bbox.cx
        cy = track.bbox.cy

        # Normalized screen offset (-1.0 to +1.0)
        norm_x = (cx - (frame_w / 2.0)) / (frame_w / 2.0)
        norm_y = ((frame_h / 2.0) - cy) / (frame_h / 2.0)

        # Angular bearings relative to boresight optical center
        azimuth_offset_deg = norm_x * (self.fov_h_deg / 2.0)
        elevation_offset_deg = norm_y * (self.fov_v_deg / 2.0)

        # Estimate slant range from bounding box height (perspective size)
        bbox_h = max(1.0, track.bbox.height)
        slant_range_m = max(5.0, round((720.0 / bbox_h) * 15.0, 1))

        # Velocity & Time-To-Impact (TTI)
        speed_kmh = getattr(track, "speed_kmh", 0.0) or 0.0
        speed_mps = (speed_kmh * 1000.0) / 3600.0

        tti_seconds = 999.0
        if speed_mps > 1.0:
            tti_seconds = round(slant_range_m / speed_mps, 1)

        return {
            "track_id": track.track_id,
            "class_name": track.class_name,
            "azimuth_deg": round(azimuth_offset_deg, 1),
            "elevation_deg": round(elevation_offset_deg, 1),
            "slant_range_m": slant_range_m,
            "speed_kmh": speed_kmh,
            "tti_seconds": tti_seconds,
        }
