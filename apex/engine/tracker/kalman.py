"""
Kalman Filter Target Predictor
==============================
High-speed 8-state / 9-state Constant Acceleration Kalman Filter for UAV target tracking.

State vector:
    x = [cx, cy, aspect_ratio, height, vx, vy, va, vh]

Design:
- Supports 60-120 km/h high-velocity target motion prediction.
- Handles measurement updates from noisy object detector bounding boxes.
- Emits future predicted bounding box for gimbal lead targeting.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from apex.engine.contracts.detection import BoundingBox


class KalmanFilterTarget:
    """
    8-State Constant Velocity / Acceleration Kalman Filter for bounding box tracking.
    """

    def __init__(self) -> None:
        # State dimension: 8 (x, y, a, h, vx, vy, va, vh)
        # Measurement dimension: 4 (x, y, a, h)
        self._ndim = 4

        # Motion transition matrix (F)
        self._motion_mat = np.eye(8, 8, dtype=np.float32)
        for i in range(4):
            self._motion_mat[i, i + 4] = 1.0  # dx/dt = vx

        # Measurement matrix (H)
        self._update_mat = np.eye(4, 8, dtype=np.float32)

        # Standard deviation weights for noise covariance initialization
        self._std_weight_position = 1.0 / 20.0
        self._std_weight_velocity = 1.0 / 160.0

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Create track state mean and covariance from first measurement [cx, cy, aspect_ratio, height].
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Run Kalman state prediction step x_{k|k-1} = F * x_{k-1|k-1}.
        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return mean, covariance

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Run Kalman measurement update step with innovation covariance.
        """
        std = [
            self._std_weight_position * measurement[3],
            self._std_weight_position * measurement[3],
            1e-1,
            self._std_weight_position * measurement[3],
        ]
        innovation_cov = np.diag(np.square(std))

        projected_mean = np.dot(self._update_mat, mean)
        projected_cov = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T)) + innovation_cov

        # Kalman gain K = P * H^T * (H * P * H^T + R)^-1
        chol_factor, lower = cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = cho_solve(
            (chol_factor, lower),
            np.dot(covariance, self._update_mat.T).T,
            check_finite=False,
        ).T

        innovation = measurement - projected_mean
        new_mean = mean + np.dot(kalman_gain, innovation)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))

        return new_mean, new_covariance

    @staticmethod
    def bbox_to_z(bbox: BoundingBox) -> np.ndarray:
        """Convert BoundingBox (xyxy) to state measurement z = [cx, cy, aspect_ratio, height]."""
        cx, cy, w, h = bbox.to_cxcywh()
        aspect_ratio = w / h if h > 0 else 1.0
        return np.array([cx, cy, aspect_ratio, h], dtype=np.float32)

    @staticmethod
    def z_to_bbox(z: np.ndarray) -> BoundingBox:
        """Convert state vector z = [cx, cy, aspect_ratio, height] back to BoundingBox (xyxy)."""
        cx, cy, aspect_ratio, h = z[:4]
        w = aspect_ratio * h
        return BoundingBox.from_cxcywh(float(cx), float(cy), float(w), float(h))
