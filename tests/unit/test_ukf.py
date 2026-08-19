"""
Unit Tests — 10-State Unscented Kalman Filter (UKF) Motion Estimator
"""

import math
import numpy as np
import pytest

from apex.engine.tracker.ukf import UnscentedKalmanFilter10D, TargetKinematicState


class TestUnscentedKalmanFilter10D:

    def test_ukf_initialization(self):
        ukf = UnscentedKalmanFilter10D(dt=0.033)
        ukf.initiate((100.0, 200.0, 50.0))

        assert ukf.x[0, 0] == 100.0
        assert ukf.x[1, 0] == 200.0
        assert ukf.x[2, 0] == 50.0

    def test_ukf_prediction_and_update(self):
        ukf = UnscentedKalmanFilter10D(dt=0.1)
        ukf.initiate((0.0, 0.0, 10.0))

        # Predict step
        ukf.predict()
        assert ukf.P.shape == (10, 10)

        # Update with new 3D measurement moving along X axis
        state = ukf.update((1.0, 0.0, 10.0))
        assert isinstance(state, TargetKinematicState)
        assert state.position_3d[0] > 0.0
        assert state.speed_kmh >= 0.0

    def test_ukf_coordinated_turn_dynamics(self):
        ukf = UnscentedKalmanFilter10D(dt=0.1)
        ukf.initiate((0.0, 0.0, 0.0))
        ukf.x[3, 0] = 10.0  # vx = 10 m/s
        ukf.x[9, 0] = 0.5   # turn_rate = 0.5 rad/s

        ukf.predict()
        assert ukf.x[1, 0] != 0.0  # Turn rate causes Y displacement
