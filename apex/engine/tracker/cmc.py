"""
Camera Motion Compensation (CMC)
================================
Estimates and compensates background camera movement (UAV translation, rotation, scale)
using Lucas-Kanade optical flow feature tracking across consecutive video frames.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import structlog

from apex.engine.contracts.detection import BoundingBox

log = structlog.get_logger(__name__)


class CameraMotionCompensator:
    """
    Lucas-Kanade optical flow based Camera Motion Compensator.
    """

    def __init__(self, max_features: int = 500) -> None:
        self.max_features = max_features
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_kp: Optional[np.ndarray] = None

    def apply(self, current_frame_bgr: np.ndarray, bboxes: list[BoundingBox]) -> list[BoundingBox]:
        """
        Estimate frame-to-frame affine transformation matrix and warp target bounding boxes.
        """
        gray = cv2.cvtColor(current_frame_bgr, cv2.COLOR_BGR2GRAY) if current_frame_bgr.ndim == 3 else current_frame_bgr

        if self._prev_gray is None:
            self._prev_gray = gray
            self._prev_kp = cv2.goodFeaturesToTrack(
                gray, maxCorners=self.max_features, qualityLevel=0.01, minDistance=10
            )
            return bboxes

        if self._prev_kp is None or len(self._prev_kp) < 10:
            self._prev_gray = gray
            self._prev_kp = cv2.goodFeaturesToTrack(
                gray, maxCorners=self.max_features, qualityLevel=0.01, minDistance=10
            )
            return bboxes

        # Track keypoints using Lucas-Kanade Pyramidal Optical Flow
        curr_kp, status, _ = cv2.calcOpticalFlowPyrLK(self._prev_gray, gray, self._prev_kp, None)

        good_prev = self._prev_kp[status == 1]
        good_curr = curr_kp[status == 1]

        if len(good_prev) < 10:
            self._prev_gray = gray
            self._prev_kp = cv2.goodFeaturesToTrack(
                gray, maxCorners=self.max_features, qualityLevel=0.01, minDistance=10
            )
            return bboxes

        # Estimate 2D Rigid / Affine Transformation Matrix (Translation + Rotation + Scale)
        affine_mat, inliers = cv2.estimateAffinePartial2D(good_prev, good_curr)

        self._prev_gray = gray
        self._prev_kp = cv2.goodFeaturesToTrack(
            gray, maxCorners=self.max_features, qualityLevel=0.01, minDistance=10
        )

        if affine_mat is None:
            return bboxes

        # Warp bounding box coordinates according to camera transformation
        warped_boxes: list[BoundingBox] = []
        for box in bboxes:
            pts = np.array(
                [[box.x1, box.y1, 1.0], [box.x2, box.y2, 1.0]],
                dtype=np.float32,
            )
            warped_pts = np.dot(affine_mat, pts.T).T
            w_x1, w_y1 = float(warped_pts[0, 0]), float(warped_pts[0, 1])
            w_x2, w_y2 = float(warped_pts[1, 0]), float(warped_pts[1, 1])
            warped_boxes.append(BoundingBox(w_x1, w_y1, w_x2, w_y2))

        return warped_boxes
