"""
Tactical Tracking Reinforcement Learning Environment
=====================================================
Maps 10D UKF target state kinematics, relative range, camera FOV offset, and geofence boundary metrics
into normalized observation vectors. Computes tactical reward signals for dynamic gimbal & intercept policies.

State Space Observation Vector (12D):
[0:3] Normalized Target Position (x, y, z)
[3:6] Normalized Target Velocity (vx, vy, vz)
[6:9] Normalized Target Acceleration (ax, ay, az)
[9]   Target Turn Rate (rad/s)
[10]  FOV Centroid Offsetting Distance (pixels / max_dim)
[11]  Geofence Breach Flag (0.0 or 1.0)
"""

from __future__ import annotations

import math
from typing import Any, Optional
import numpy as np
import structlog

log = structlog.get_logger(__name__)


class TrackingEnvironment:
    """Tactical Tracking Environment observation builder and reward evaluator."""

    def __init__(self, fov_width: float = 1280.0, fov_height: float = 720.0) -> None:
        self.fov_width = fov_width
        self.fov_height = fov_height
        self.center_x = fov_width / 2.0
        self.center_y = fov_height / 2.0
        self.max_fov_dist = math.hypot(self.center_x, self.center_y)

    def extract_observation(
        self,
        ukf_state: np.ndarray,
        bbox_center: tuple[float, float],
        is_breached: bool = False,
    ) -> np.ndarray:
        """
        Build normalized 12-element observation array from 10D UKF state and frame metadata.
        
        ukf_state: 10D array [x, y, z, vx, vy, vz, ax, ay, az, turn_rate]
        """
        obs = np.zeros(12, dtype=np.float32)
        if len(ukf_state) >= 10:
            # Position (scaled by 1000m)
            obs[0:3] = np.clip(ukf_state[0:3] / 1000.0, -5.0, 5.0)
            # Velocity (scaled by 100m/s)
            obs[3:6] = np.clip(ukf_state[3:6] / 100.0, -5.0, 5.0)
            # Acceleration (scaled by 20m/s^2)
            obs[6:9] = np.clip(ukf_state[6:9] / 20.0, -5.0, 5.0)
            # Turn rate
            obs[9] = np.clip(ukf_state[9] / math.pi, -2.0, 2.0)

        # FOV alignment distance
        cx, cy = bbox_center
        fov_dist = math.hypot(cx - self.center_x, cy - self.center_y)
        obs[10] = float(np.clip(fov_dist / self.max_fov_dist, 0.0, 1.0))

        # Geofence status flag
        obs[11] = 1.0 if is_breached else 0.0
        return obs

    def compute_reward(
        self,
        observation: np.ndarray,
        action: int,
        track_confirmed: bool = True,
    ) -> float:
        """
        Compute scalar tactical reward for current RL action step.
        
        Reward components:
        + 1.0 for maintaining active confirmed target lock
        + (1.0 - fov_offset) for keeping target centered in optical FOV
        - 5.0 penalty for geofence breach
        - 0.1 minor penalty for aggressive action switching
        """
        fov_offset = float(observation[10])
        is_breached = bool(observation[11] > 0.5)

        reward = 0.0

        if track_confirmed:
            reward += 1.0

        # FOV centering bonus (higher reward when target is near center)
        reward += (1.0 - fov_offset) * 2.0

        # High-Speed Velocity Compensation (60-120 km/h) bonus
        vx, vy, vz = observation[3:6]
        speed_norm = float(np.linalg.norm([vx, vy, vz]))
        if speed_norm > 0.15:  # Target moving fast (60-120 km/h range)
            reward += 1.5 * (1.0 - fov_offset)  # Reward keeping high-speed target locked

        # Geofence breach penalty
        if is_breached:
            reward -= 5.0

        # Action-specific rewards/penalties
        if action == 1:  # CENTER_GIMBAL
            reward += 0.5
        elif action == 2:  # HIGH_SPEED_INTERCEPT
            reward += 1.0 if speed_norm > 0.15 else -0.2

        return float(reward)
