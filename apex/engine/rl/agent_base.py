"""
Reinforcement Learning Abstract Agent Contract
==============================================
Abstract interface for RL policy agents operating in APEX-Track.
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Any, Optional
import numpy as np


class RLAgentBase(abc.ABC):
    """Abstract base class for all Reinforcement Learning policy agents."""

    def __init__(self, state_dim: int = 12, action_dim: int = 4) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim

    @abc.abstractmethod
    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """Select action given observation state vector."""

    @abc.abstractmethod
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store experience tuple into replay buffer."""

    @abc.abstractmethod
    def update_policy(self) -> Optional[dict[str, float]]:
        """Perform policy learning update step."""

    @abc.abstractmethod
    def save(self, filepath: str | Path) -> bool:
        """Save model checkpoint."""

    @abc.abstractmethod
    def load(self, filepath: str | Path) -> bool:
        """Load model checkpoint."""
