"""
Reinforcement Learning Subsystem Manager
=========================================
Coordinates observation building, agent policy evaluation, and event publishing for real-time target tracking.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Optional
import numpy as np
import structlog

from apex.engine.bus.channels import Ch
from apex.engine.bus.message_bus import MessageBus
from apex.engine.rl.dqn_agent import DQNAgent
from apex.engine.rl.rl_env import TrackingEnvironment

log = structlog.get_logger(__name__)


class RLManager:
    """Singleton coordinator for RL perception policy evaluation and training."""

    _instance: Optional["RLManager"] = None

    def __init__(self, bus: Optional[MessageBus] = None) -> None:
        self.bus = bus or MessageBus.instance()
        self.env = TrackingEnvironment()
        self.agent = DQNAgent()
        self._last_obs: Optional[np.ndarray] = None
        self._last_action: Optional[int] = None
        self._total_steps = 0
        self._total_reward = 0.0

    @classmethod
    def instance(cls) -> "RLManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def evaluate_step(
        self,
        ukf_state: np.ndarray,
        bbox_center: tuple[float, float],
        is_breached: bool = False,
        track_confirmed: bool = True,
    ) -> dict[str, Any]:
        """
        Execute single RL policy evaluation step given UKF state and frame metadata.
        Returns dict containing selected action, reward, and state vector.
        """
        obs = self.env.extract_observation(ukf_state, bbox_center, is_breached)
        action = self.agent.select_action(obs)

        reward = 0.0
        if self._last_obs is not None and self._last_action is not None:
            reward = self.env.compute_reward(obs, action, track_confirmed)
            self.agent.store_transition(self._last_obs, self._last_action, reward, obs, False)
            self._total_reward += reward

        update_info = self.agent.update_policy()
        if self._total_steps % 50 == 0:
            self.agent.update_target_network()

        self._last_obs = obs
        self._last_action = action
        self._total_steps += 1

        action_names = {
            0: "MAINTAIN_TRACK",
            1: "CENTER_GIMBAL",
            2: "HIGH_SPEED_INTERCEPT",
            3: "COUNTERMEASURE_ENGAGE",
        }
        action_label = action_names.get(action, "UNKNOWN")

        result = {
            "step": self._total_steps,
            "action": action,
            "action_label": action_label,
            "reward": round(reward, 3),
            "total_reward": round(self._total_reward, 3),
            "epsilon": round(getattr(self.agent, "epsilon", 0.0), 3),
            "update_metrics": update_info,
        }

        # Publish RL action event to MessageBus
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.bus.publish(Ch.MISSION_COMMANDS, result))
        except RuntimeError:
            pass
        return result

    def get_status(self) -> dict[str, Any]:
        """Return RL subsystem telemetry status."""
        return {
            "total_steps": self._total_steps,
            "total_reward": round(self._total_reward, 3),
            "epsilon": round(getattr(self.agent, "epsilon", 0.0), 3),
            "replay_buffer_size": len(self.agent.replay_buffer),
        }
