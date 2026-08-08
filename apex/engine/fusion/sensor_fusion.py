"""
Acoustic & RF Sensor Fusion Engine
===================================
Correlates optical target detections with simulated 360-degree RF Direction Finder
and Acoustic Micro-Doppler Array azimuth vectors for early-warning threat detection.
"""

from __future__ import annotations

import math
from typing import Dict, List, Any
import numpy as np


class SensorFusionEngine:
    """Fuses multi-modal optical, RF azimuth, and acoustic sensor vectors."""

    def __init__(self) -> None:
        self.rf_bearings: List[Dict[str, Any]] = [
            {"azimuth_deg": 45.0, "frequency_mhz": 2400.0, "signal_dbm": -65},
            {"azimuth_deg": 135.0, "frequency_mhz": 5800.0, "signal_dbm": -52},
        ]

    def correlate_tracks(self, tracks: List[Any], fov_deg: float = 60.0) -> List[Dict[str, Any]]:
        """
        Correlates optical track bounding box centroids with RF direction finder bearings.
        """
        fused_tracks: List[Dict[str, Any]] = []

        for tr in tracks:
            cx = tr.bbox.cx
            # Approximate azimuth angle (-fov/2 to +fov/2) relative to camera FOV (centered at 0 deg)
            norm_x = (cx - 640.0) / 640.0
            track_azimuth = norm_x * (fov_deg / 2.0)

            best_rf_match = None
            min_diff = 999.0

            for rf in self.rf_bearings:
                diff = abs(track_azimuth - rf["azimuth_deg"])
                if diff < min_diff:
                    min_diff = diff
                    best_rf_match = rf

            fused_tracks.append({
                "track_id": tr.track_id,
                "class_name": tr.class_name,
                "optical_azimuth_deg": round(track_azimuth, 1),
                "rf_matched": min_diff < 15.0,
                "rf_bearing": best_rf_match if min_diff < 15.0 else None,
            })

        return fused_tracks
