"""
Unit Tests — Mission Replay Engine
"""

import json
import pytest

from apex.engine.replay.replay_engine import ReplayEngine


class TestReplayEngine:

    def test_replay_engine_load_and_step(self, tmp_path):
        # Create temp JSONL recording file
        log_file = tmp_path / "test_recording.jsonl"
        events = [
            {"timestamp": 100.0, "channel": "ch.event", "payload": "evt_1"},
            {"timestamp": 101.0, "channel": "ch.event", "payload": "evt_2"},
            {"timestamp": 102.0, "channel": "ch.event", "payload": "evt_3"},
        ]
        with open(log_file, "w") as f:
            for ev in events:
                f.write(json.dumps(ev) + "\n")

        engine = ReplayEngine()
        loaded_count = engine.load_recording(log_file)
        assert loaded_count == 3
        assert engine.total_records == 3

        # Test stepping through events
        ev1 = engine.step()
        assert ev1 is not None
        assert ev1["payload"] == "evt_1"
        assert engine.current_index == 1

        # Test seeking
        idx = engine.seek(102.0)
        assert idx == 2
        assert engine.current_index == 2

        ev3 = engine.step()
        assert ev3["payload"] == "evt_3"

        # End of recording
        assert engine.step() is None

    def test_replay_speed_controls(self, tmp_path):
        log_file = tmp_path / "test_speed.jsonl"
        with open(log_file, "w") as f:
            f.write(json.dumps({"timestamp": 1.0, "payload": "a"}) + "\n")

        engine = ReplayEngine()
        engine.load_recording(log_file)

        engine.set_speed(2.5)
        assert engine.playback_speed == 2.5

        engine.set_speed(15.0)  # Clamped to 10.0
        assert engine.playback_speed == 10.0
