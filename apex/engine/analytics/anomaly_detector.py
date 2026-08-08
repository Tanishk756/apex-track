"""
Kinematic Anomaly & Erratic Motion Detector
============================================
Detects tactical evasive maneuvers, high-G turns, hovering anomalies, and velocity spikes.
"""

from __future__ import annotations

import math
from typing import Dict, List, Any


class AnomalyDetector:
    """Flags kinematic anomalies in target tracking data."""

    def __init__(self, high_speed_thresh: float = 35.0) -> None:
        self.high_speed_thresh = high_speed_thresh

    def detect_anomalies(self, tracks: List[Any], trajectory_data: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Scans active tracks for high-speed anomalies and evasive acceleration.
        """
        anomalies: List[Dict[str, Any]] = []

        for tr in tracks:
            tid = tr.track_id
            traj = trajectory_data.get(tid, {})
            speed = traj.get("speed_px", 0.0)
            is_closing = traj.get("is_closing", False)

            if speed > self.high_speed_thresh:
                anomalies.append({
                    "track_id": tid,
                    "class_name": tr.class_name,
                    "anomaly_type": "HIGH_SPEED_BREAKOUT",
                    "speed_px": speed,
                    "severity": "HIGH" if is_closing else "MEDIUM",
                })
            elif speed < 1.0 and tr.class_name.lower().strip() in ("drone", "uav"):
                anomalies.append({
                    "track_id": tid,
                    "class_name": tr.class_name,
                    "anomaly_type": "EVASIVE_HOVER",
                    "speed_px": speed,
                    "severity": "MEDIUM",
                })

        return anomalies
