"""
Trajectory Predictor & Kinematic Intercept Vector Engine
=======================================================
Predicts 5-second future flight paths (t+1s, t+2s, t+3s, t+5s) and Time-to-Collision (TTC)
for active targets based on 2nd-order polynomial kinematic state modeling.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple
import numpy as np

from apex.engine.contracts.track import Track
from apex.engine.contracts.detection import BoundingBox


class TrajectoryPredictor:
    """Computes future motion vectors and intercept trajectories for tracked targets."""

    def __init__(self, history_len: int = 15) -> None:
        self.history_len = history_len
        self.track_histories: Dict[int, List[Tuple[float, float, float]]] = {}  # tid -> [(x, y, ts)]

    def update_and_predict(self, tracks: List[Track]) -> Dict[int, Dict[str, Any]]:
        """
        Updates track history and computes predicted future trajectory points.
        Returns map of track_id -> trajectory metrics:
          - future_points: list of (x, y) coordinates for +1s, +2s, +3s, +5s
          - ttc_seconds: estimated time to reach sensor origin (if closing)
          - is_closing: bool
        """
        results: Dict[int, Dict[str, Any]] = {}

        for tr in tracks:
            tid = tr.track_id
            cx, cy = tr.bbox.cx, tr.bbox.cy
            ts = tr.frame_timestamp

            if tid not in self.track_histories:
                self.track_histories[tid] = []

            self.track_histories[tid].append((cx, cy, ts))
            if len(self.track_histories[tid]) > self.history_len:
                self.track_histories[tid].pop(0)

            history = self.track_histories[tid]
            vx, vy = tr.velocity_px

            # Predict future points at t + dt
            future_points: List[Tuple[float, float]] = []
            dts = [1.0, 2.0, 3.0, 5.0]

            if len(history) >= 3:
                # 2nd order acceleration estimation
                pts = np.array([(p[0], p[1]) for p in history])
                times = np.array([p[2] - history[0][2] for p in history])

                if times[-1] > 0.001:
                    # Velocity & Acceleration vectors
                    ax = 0.0
                    ay = 0.0
                    if len(history) >= 5:
                        dt = times[-1] - times[-2]
                        if dt > 0.001:
                            v_prev_x = (history[-2][0] - history[-3][0]) / dt
                            v_prev_y = (history[-2][1] - history[-3][1]) / dt
                            ax = (vx - v_prev_x) / dt
                            ay = (vy - v_prev_y) / dt

                    for dt_fut in dts:
                        px = cx + vx * dt_fut + 0.5 * ax * (dt_fut ** 2)
                        py = cy + vy * dt_fut + 0.5 * ay * (dt_fut ** 2)
                        future_points.append((float(px), float(py)))
            else:
                for dt_fut in dts:
                    future_points.append((cx + vx * dt_fut, cy + vy * dt_fut))

            # Calculate closing speed and Time-to-Collision (TTC) to center of frame
            speed = math.sqrt(vx**2 + vy**2)
            ttc_seconds = 999.0
            is_closing = False

            if speed > 1.0:
                dist_to_center = math.sqrt(cx**2 + cy**2)
                # Check vector direction
                dot_prod = (cx * vx + cy * vy)
                if dot_prod < 0:
                    is_closing = True
                    ttc_seconds = dist_to_center / speed

            results[tid] = {
                "future_points": future_points,
                "ttc_seconds": round(ttc_seconds, 2),
                "is_closing": is_closing,
                "speed_px": round(speed, 2),
            }

        return results
