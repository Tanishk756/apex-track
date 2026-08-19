"""
Unit Tests — Reinforcement Learning Subsystem
"""

import numpy as np
import pytest

from apex.engine.rl.dqn_agent import DQNAgent
from apex.engine.rl.rl_env import TrackingEnvironment
from apex.engine.rl.rl_manager import RLManager


class TestReinforcementLearningSubsystem:

    def test_tracking_environment_observation_extraction(self):
        env = TrackingEnvironment(fov_width=1280.0, fov_height=720.0)
        ukf_state = np.array([100.0, 200.0, 50.0, 10.0, -5.0, 0.0, 1.0, 0.5, 0.0, 0.1], dtype=np.float32)

        obs = env.extract_observation(ukf_state, bbox_center=(640.0, 360.0), is_breached=False)
        assert len(obs) == 12
        assert obs[0] == pytest.approx(0.1, abs=1e-3)  # 100 / 1000
        assert obs[10] == pytest.approx(0.0, abs=1e-3) # Centered FOV offset
        assert obs[11] == 0.0

    def test_reward_computation(self):
        env = TrackingEnvironment(fov_width=1280.0, fov_height=720.0)
        ukf_state = np.array([100.0, 200.0, 50.0, 10.0, -5.0, 0.0, 1.0, 0.5, 0.0, 0.1], dtype=np.float32)
        obs = env.extract_observation(ukf_state, bbox_center=(640.0, 360.0), is_breached=False)

        reward = env.compute_reward(obs, action=1, track_confirmed=True)
        assert reward > 0.0

        # Breached state should penalize
        obs_breached = env.extract_observation(ukf_state, bbox_center=(640.0, 360.0), is_breached=True)
        reward_breached = env.compute_reward(obs_breached, action=1, track_confirmed=True)
        assert reward_breached < reward

    def test_dqn_agent_action_and_experience_replay(self):
        agent = DQNAgent(state_dim=12, action_dim=4, batch_size=4)
        obs = np.random.randn(12).astype(np.float32)

        action = agent.select_action(obs, eval_mode=True)
        assert 0 <= action < 4

        # Store transitions
        for i in range(10):
            agent.store_transition(obs, action, 1.0, obs, False)

        assert len(agent.replay_buffer) == 10

    def test_rl_manager_step(self):
        mgr = RLManager()
        ukf_state = np.array([100.0, 200.0, 50.0, 10.0, -5.0, 0.0, 1.0, 0.5, 0.0, 0.1], dtype=np.float32)

        res = mgr.evaluate_step(ukf_state, (640.0, 360.0))
        assert "action" in res
        assert "action_label" in res
        assert res["step"] > 0

        status = mgr.get_status()
        assert status["total_steps"] > 0
