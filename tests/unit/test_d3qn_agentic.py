"""
Unit Tests — D3QN Policy Architecture, PER, Spatial-Visual Re-ID & ReAct Agentic AI Subsystems
"""

import numpy as np
import pytest

from apex.engine.agent.tactical_rag_agent import TacticalAgentRAG
from apex.engine.contracts.detection import BoundingBox
from apex.engine.rl.dqn_agent import DQNAgent
from apex.engine.tracker.reid_db import PersistentReIDDatabase
from apex.engine.tracker.reid_extractor import ReIDFeatureExtractor


class TestD3QNAgenticAndReID:

    def test_d3qn_architecture_and_per_buffer(self):
        agent = DQNAgent(state_dim=12, action_dim=4, batch_size=4)
        s1 = np.ones(12, dtype=np.float32)
        s2 = np.ones(12, dtype=np.float32) * 0.5

        # Store transitions in Prioritized Experience Replay buffer
        for i in range(10):
            agent.store_transition(s1, action=1, reward=1.0, next_state=s2, done=False)

        assert len(agent.replay_buffer) == 10
        assert len(agent.priorities) == 10

        # Perform D3QN policy update step
        metrics = agent.update_policy()
        if agent._use_torch:
            assert metrics is not None
            assert "loss" in metrics
            assert "avg_td_error" in metrics

    def test_spatial_visual_reid_fused_matching(self):
        db = PersistentReIDDatabase.instance()
        db.reset()

        bbox1 = BoundingBox(100, 100, 200, 200)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[100:200, 100:200] = 128  # Grey patch

        # Frame 1: Register new persistent target UID
        uid1, reid1 = db.match_or_create_uid(img, bbox1, "person")
        assert uid1 == 1
        assert reid1 is False

        # Frame 2: Move target slightly (spatial + visual continuity)
        bbox2 = BoundingBox(105, 105, 205, 205)
        uid2, reid2 = db.match_or_create_uid(img, bbox2, "person")
        assert uid2 == 1  # Should preserve same UID=1!
        assert reid2 is True

    def test_react_autonomous_tool_calling_agent(self):
        agent = TacticalAgentRAG.instance()

        # 1. Audit tool call
        res1 = agent.query("Audit historical log database")
        assert "Thought:" in res1["thought_process"][0]
        assert "`query_history_db`" in res1["data_sources"][0]

        # 2. Intercept calculation tool call
        res2 = agent.query("Calculate dynamic intercept for high speed target")
        assert "Thought:" in res2["thought_process"][0]
        assert "HIGH_SPEED_INTERCEPT" in res2["action_recommended"]

        # 3. Action execution tool call
        res3 = agent.query("Engage countermeasure action override")
        assert "Thought:" in res3["thought_process"][0]
        assert "ENGAGE_COUNTERMEASURE" in res3["action_recommended"]
