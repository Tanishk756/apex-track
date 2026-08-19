"""
Unit Tests — REMIND (RE-Identification with Memory for INDoor Navigation) Engine
"""

import numpy as np
import pytest

from apex.engine.contracts.detection import BoundingBox
from apex.engine.tracker.reid_db import PersistentReIDDatabase
from apex.engine.tracker.remind_reid import REMINDReIDTracker, TargetMemoryProfile


@pytest.fixture(autouse=True)
def reset_remind():
    REMINDReIDTracker.reset()
    PersistentReIDDatabase.instance().reset()
    yield
    REMINDReIDTracker.reset()
    PersistentReIDDatabase.instance().reset()


class TestREMINDMemoryEngine:

    def test_target_memory_profile_initialization(self):
        feat = np.ones(128, dtype=np.float32) / np.sqrt(128)
        profile = TargetMemoryProfile(
            uid=1,
            class_name="person",
            initial_feature=feat,
            initial_center=(100.0, 100.0),
            timestamp=1000.0,
            max_episodic=5,
        )
        assert profile.uid == 1
        assert profile.class_name == "person"
        assert len(profile.episodic_buffer) == 1
        assert np.isclose(np.linalg.norm(profile.semantic_centroid), 1.0)

    def test_episodic_buffer_eviction_and_semantic_ema(self):
        feat1 = np.ones(128, dtype=np.float32) / np.sqrt(128)
        profile = TargetMemoryProfile(
            uid=1,
            class_name="robot",
            initial_feature=feat1,
            initial_center=(100.0, 100.0),
            timestamp=1000.0,
            max_episodic=3,
        )

        for i in range(5):
            f = np.zeros(128, dtype=np.float32)
            f[i] = 1.0
            profile.update(f, (100.0 + i, 100.0 + i), 1000.0 + i)

        assert len(profile.episodic_buffer) == 3
        assert profile.total_observations == 6
        assert np.linalg.norm(profile.semantic_centroid) > 0.9

    def test_remind_match_or_register_new_target(self):
        tracker = REMINDReIDTracker.instance()
        bbox = BoundingBox(10.0, 10.0, 50.0, 100.0)
        img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)

        uid1, is_reid1, conf1 = tracker.match_or_register(img, bbox, "person", timestamp=10.0)
        assert uid1 == 1
        assert is_reid1 is False

        # Re-query same image crop & location (should re-identify target)
        uid2, is_reid2, conf2 = tracker.match_or_register(img, bbox, "person", timestamp=12.0)
        assert uid2 == 1
        assert is_reid2 is True
        assert conf2 >= tracker.sim_threshold

    def test_remind_status_and_target_memory(self):
        tracker = REMINDReIDTracker.instance()
        bbox = BoundingBox(20.0, 20.0, 80.0, 140.0)
        img = np.full((300, 300, 3), 128, dtype=np.uint8)

        uid, _, _ = tracker.match_or_register(img, bbox, "uav", timestamp=1.0)
        status = tracker.get_status()
        assert status["engine"] == "REMIND (Memory Re-ID)"
        assert status["active_profiles_in_memory"] == 1

        mem = tracker.get_target_memory(uid)
        assert mem is not None
        assert mem["uid"] == uid
        assert mem["class_name"] == "uav"
        assert mem["episodic_buffer_count"] == 1

    def test_persistent_reid_db_integration(self):
        db = PersistentReIDDatabase.instance()
        bbox = BoundingBox(30.0, 30.0, 90.0, 150.0)
        img = np.full((300, 300, 3), 200, dtype=np.uint8)

        uid1, is_reid1 = db.match_or_create_uid(img, bbox, "vehicle")
        assert uid1 >= 1
        assert is_reid1 is False

        uid2, is_reid2 = db.match_or_create_uid(img, bbox, "vehicle")
        assert uid2 == uid1
        assert is_reid2 is True

        remind_status = db.get_remind_status()
        assert remind_status["total_reid_hits"] >= 1
