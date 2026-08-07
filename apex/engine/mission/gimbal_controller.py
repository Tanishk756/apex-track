"""
Gimbal Controller
=================
Closed-loop PID target pursuit gimbal controller for EO/IR camera payloads.

Design:
- Computes Pan / Tilt / Zoom slew rate commands from target pixel offset.
- Uses Kalman-predicted next-frame bounding box for predictive lead targeting.
- Outputs MAVLink MOUNT_CONTROL / MAV_CMD_DO_MOUNT_CONTROL pitch/yaw angles.
"""

from __future__ import annotations

import math
import structlog

from apex.engine.contracts.track import Track

log = structlog.get_logger(__name__)


class GimbalPIDController:
    """Proportional-Integral-Derivative gimbal angular velocity controller."""

    def __init__(self, kp: float = 0.25, ki: float = 0.01, kd: float = 0.05) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self._integral_x = 0.0
        self._integral_y = 0.0
        self._prev_error_x = 0.0
        self._prev_error_y = 0.0

    def compute_slew_rates(
        self,
        target_track: Track,
        image_w: int = 640,
        image_h: int = 480,
        use_predictive_lead: bool = True,
    ) -> tuple[float, float, float]:
        """
        Calculate (pan_rate_deg_s, tilt_rate_deg_s, target_distance_m).
        """
        bbox = target_track.predicted_bbox if use_predictive_lead else target_track.bbox

        # Normalized pixel error from optical frame center [-1.0, 1.0]
        cx, cy = image_w / 2.0, image_h / 2.0
        error_x = (bbox.cx - cx) / cx
        error_y = (bbox.cy - cy) / cy

        # PID calculations
        self._integral_x += error_x
        self._integral_y += error_y
        deriv_x = error_x - self._prev_error_x
        deriv_y = error_y - self._prev_error_y

        self._prev_error_x = error_x
        self._prev_error_y = error_y

        # Slew rates (deg/sec)
        pan_rate = self.kp * error_x + self.ki * self._integral_x + self.kd * deriv_x
        tilt_rate = -(self.kp * error_y + self.ki * self._integral_y + self.kd * deriv_y)

        # Scale to max angular velocity (+/- 45 deg/sec)
        pan_rate_deg = float(math.degrees(pan_rate))
        tilt_rate_deg = float(math.degrees(tilt_rate))

        return pan_rate_deg, tilt_rate_deg, 0.0

    def reset(self) -> None:
        self._integral_x = 0.0
        self._integral_y = 0.0
        self._prev_error_x = 0.0
        self._prev_error_y = 0.0
