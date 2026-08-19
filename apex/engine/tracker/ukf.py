"""
10-State Unscented Kalman Filter (UKF) Motion State Estimator
============================================================
Provides non-linear state estimation decoupled from detection and data association.

State Vector (10D):
  [0] x          : Easting / X position (m or px)
  [1] y          : Northing / Y position (m or px)
  [2] z          : Altitude / Z position (m)
  [3] vx         : X velocity (m/s or px/s)
  [4] vy         : Y velocity (m/s or px/s)
  [5] vz         : Z velocity (m/s)
  [6] ax         : X acceleration (m/s^2 or px/s^2)
  [7] ay         : Y acceleration (m/s^2 or px/s^2)
  [8] az         : Z acceleration (m/s^2)
  [9] turn_rate  : Yaw / turn rate (rad/s)
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple
import numpy as np
import structlog

log = structlog.get_logger(__name__)


@dataclass(slots=True)
class TargetKinematicState:
    """Estimated target kinematic state."""

    track_id: int
    timestamp: float
    position_3d: Tuple[float, float, float]
    velocity_3d: Tuple[float, float, float]
    acceleration_3d: Tuple[float, float, float]
    turn_rate_rad_s: float
    covariance: np.ndarray
    speed_kmh: float


class UnscentedKalmanFilter10D:
    """
    10-State Unscented Kalman Filter for non-linear maneuvering targets.
    """

    def __init__(
        self,
        dt: float = 0.033,
        process_noise_std: float = 0.5,
        measurement_noise_std: float = 1.0,
    ) -> None:
        self.dt = dt
        self.dim_x = 10
        self.dim_z = 3  # 3D Position Measurement [x, y, z]

        # State vector
        self.x = np.zeros((10, 1), dtype=np.float64)

        # Covariance matrix P
        self.P = np.eye(10, dtype=np.float64) * 10.0

        # Process Noise Q
        self.Q = np.eye(10, dtype=np.float64) * (process_noise_std ** 2)

        # Measurement Noise R
        self.R = np.eye(3, dtype=np.float64) * (measurement_noise_std ** 2)

        # UKF Merwe Sigma Point Hyperparameters
        self.alpha = 1e-3
        self.beta = 2.0
        self.kappa = 0.0
        self._compute_weights()

    def _compute_weights(self) -> None:
        n = self.dim_x
        lambda_val = (self.alpha ** 2) * (n + self.kappa) - n
        num_sigmas = 2 * n + 1

        self.weights_mean = np.full(num_sigmas, 0.5 / (n + lambda_val))
        self.weights_cov = np.full(num_sigmas, 0.5 / (n + lambda_val))

        self.weights_mean[0] = lambda_val / (n + lambda_val)
        self.weights_cov[0] = lambda_val / (n + lambda_val) + (1 - self.alpha ** 2 + self.beta)
        self.lambda_val = lambda_val

    def initiate(self, position_3d: Tuple[float, float, float]) -> None:
        """Initialize state vector from initial 3D measurement."""
        self.x[0, 0] = position_3d[0]
        self.x[1, 0] = position_3d[1]
        self.x[2, 0] = position_3d[2]

    def fx(self, x: np.ndarray, dt: float) -> np.ndarray:
        """Non-linear state transition function."""
        x_out = x.copy()
        vx, vy, vz = x[3, 0], x[4, 0], x[5, 0]
        ax, ay, az = x[6, 0], x[7, 0], x[8, 0]
        w = x[9, 0]

        # 3D Constant Acceleration + Coordinated Turn Position Update
        if abs(w) > 1e-5:
            # Turn dynamics integration
            cos_wt = math.cos(w * dt)
            sin_wt = math.sin(w * dt)
            x_out[0, 0] += (vx * sin_wt + vy * (cos_wt - 1.0)) / w + 0.5 * ax * (dt ** 2)
            x_out[1, 0] += (vx * (1.0 - cos_wt) + vy * sin_wt) / w + 0.5 * ay * (dt ** 2)
        else:
            x_out[0, 0] += vx * dt + 0.5 * ax * (dt ** 2)
            x_out[1, 0] += vy * dt + 0.5 * ay * (dt ** 2)

        x_out[2, 0] += vz * dt + 0.5 * az * (dt ** 2)

        # Velocity Update
        x_out[3, 0] += ax * dt
        x_out[4, 0] += ay * dt
        x_out[5, 0] += az * dt

        return x_out

    def predict(self, dt: Optional[float] = None) -> None:
        """Execute UKF time update step."""
        dt_val = dt if dt is not None else self.dt

        # 1. Generate Sigma Points
        sigmas = self._generate_sigma_points()

        # 2. Propagate Sigma Points through non-linear dynamics fx
        sigmas_f = np.zeros_like(sigmas)
        for i in range(sigmas.shape[1]):
            sigmas_f[:, i : i + 1] = self.fx(sigmas[:, i : i + 1], dt_val)

        # 3. Compute Predicted Mean and Covariance
        x_pred = np.zeros((10, 1), dtype=np.float64)
        for i in range(sigmas_f.shape[1]):
            x_pred += self.weights_mean[i] * sigmas_f[:, i : i + 1]

        P_pred = self.Q.copy()
        for i in range(sigmas_f.shape[1]):
            y = sigmas_f[:, i : i + 1] - x_pred
            P_pred += self.weights_cov[i] * (y @ y.T)

        self.x = x_pred
        self.P = P_pred

    def update(self, measurement: Tuple[float, float, float]) -> TargetKinematicState:
        """Execute UKF measurement update step."""
        z = np.array(measurement, dtype=np.float64).reshape(3, 1)

        # H matrix projects 10D state to 3D position [x, y, z]
        H = np.zeros((3, 10), dtype=np.float64)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0

        y = z - (H @ self.x)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(10) - K @ H) @ self.P

        vx, vy, vz = self.x[3, 0], self.x[4, 0], self.x[5, 0]
        speed_mps = math.sqrt(vx**2 + vy**2 + vz**2)

        return TargetKinematicState(
            track_id=0,
            timestamp=0.0,
            position_3d=(float(self.x[0, 0]), float(self.x[1, 0]), float(self.x[2, 0])),
            velocity_3d=(float(vx), float(vy), float(vz)),
            acceleration_3d=(float(self.x[6, 0]), float(self.x[7, 0]), float(self.x[8, 0])),
            turn_rate_rad_s=float(self.x[9, 0]),
            covariance=self.P.copy(),
            speed_kmh=round(speed_mps * 3.6, 1),
        )

    def _generate_sigma_points(self) -> np.ndarray:
        n = self.dim_x
        num_sigmas = 2 * n + 1
        sigmas = np.zeros((n, num_sigmas), dtype=np.float64)
        sigmas[:, 0:1] = self.x

        scale = math.sqrt(n + self.lambda_val)
        try:
            L = np.linalg.cholesky(self.P)
        except np.linalg.LinAlgError:
            L = np.eye(n)

        for k in range(n):
            sigmas[:, k + 1 : k + 2] = self.x + scale * L[:, k : k + 1]
            sigmas[:, n + k + 1 : n + k + 2] = self.x - scale * L[:, k : k + 1]

        return sigmas
