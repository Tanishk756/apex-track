"""
Deep Q-Network (DQN) Policy Agent Implementation
================================================
PyTorch-backed Deep Q-Network with Experience Replay Buffer and Epsilon-Greedy exploration policy.
Features pure-NumPy fallback mode if PyTorch is operating without CUDA or in minimal mode.
"""

from __future__ import annotations

import collections
import random
from pathlib import Path
from typing import Optional
import numpy as np
import structlog

from apex.engine.rl.agent_base import RLAgentBase

log = structlog.get_logger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    class DuelingQNetwork(nn.Module):
        """Dueling Neural Network Architecture separating State Value V(s) and Action Advantage A(s, a)."""

        def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
            super().__init__()
            self.feature_net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
            )
            # Value Stream V(s)
            self.value_stream = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            # Advantage Stream A(s, a)
            self.advantage_stream = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            feats = self.feature_net(x)
            val = self.value_stream(feats)
            adv = self.advantage_stream(feats)
            # Combine: Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
            q_vals = val + (adv - adv.mean(dim=1, keepdim=True))
            return q_vals


class DQNAgent(RLAgentBase):
    """Dueling Double Deep Q-Network (D3QN) RL Agent with Prioritized Experience Replay."""

    def __init__(
        self,
        state_dim: int = 12,
        action_dim: int = 4,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 10000,
        batch_size: int = 32,
        alpha: float = 0.6,
    ) -> None:
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.alpha = alpha
        self.replay_buffer: collections.deque = collections.deque(maxlen=buffer_capacity)
        self.priorities: collections.deque = collections.deque(maxlen=buffer_capacity)

        self._use_torch = HAS_TORCH
        if self._use_torch:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.q_net = DuelingQNetwork(state_dim, action_dim).to(self.device)
            self.target_net = DuelingQNetwork(state_dim, action_dim).to(self.device)
            self.target_net.load_state_dict(self.q_net.state_dict())
            self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
            self.loss_fn = nn.MSELoss()
            log.info("d3qn_agent_initialized_pytorch", device=str(self.device))
        else:
            log.info("dqn_agent_initialized_numpy_fallback")

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """Select action using epsilon-greedy strategy."""
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        if self._use_torch:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_net(state_t)
                action = int(torch.argmax(q_values, dim=1).item())
                return action
        else:
            fov_dist = state[10] if len(state) > 10 else 0.5
            return 1 if fov_dist < 0.3 else 0

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store experience in prioritized replay buffer."""
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.replay_buffer.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)

    def update_policy(self) -> Optional[dict[str, float]]:
        """Perform Double Q-learning update with Prioritized Experience Replay (PER)."""
        if len(self.replay_buffer) < self.batch_size or not self._use_torch:
            return None

        # Sample transitions based on priorities
        prios = np.array(self.priorities, dtype=np.float32) ** self.alpha
        probs = prios / prios.sum()
        indices = np.random.choice(len(self.replay_buffer), self.batch_size, p=probs, replace=False)

        batch = [self.replay_buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Double Q-Learning: action selected by main network, evaluated by target network
        curr_q = self.q_net(states_t).gather(1, actions_t)
        with torch.no_grad():
            best_next_actions = self.q_net(next_states_t).argmax(dim=1, keepdim=True)
            max_next_q = self.target_net(next_states_t).gather(1, best_next_actions)
            target_q = rewards_t + (1.0 - dones_t) * self.gamma * max_next_q

        td_errors = torch.abs(curr_q - target_q).detach().cpu().numpy().flatten()
        for i, idx in enumerate(indices):
            self.priorities[idx] = float(td_errors[i] + 1e-5)

        loss = self.loss_fn(curr_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return {"loss": float(loss.item()), "epsilon": self.epsilon, "avg_td_error": float(np.mean(td_errors))}

    def update_target_network(self) -> None:
        """Copy Q-network weights to target network."""
        if self._use_torch:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, filepath: str | Path) -> bool:
        """Save network weights."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._use_torch:
            torch.save(self.q_net.state_dict(), path)
            log.info("dqn_weights_saved", path=str(path))
            return True
        return False

    def load(self, filepath: str | Path) -> bool:
        """Load network weights."""
        path = Path(filepath)
        if not path.is_file() or not self._use_torch:
            return False
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.q_net.state_dict())
        log.info("dqn_weights_loaded", path=str(path))
        return True
