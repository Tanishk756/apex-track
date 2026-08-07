"""
Unit Tests — System State Machine
"""

import pytest
from apex.engine.state.system_state import SystemState, SystemStateMachine


class TestSystemStateMachine:

    def test_initial_state(self):
        sm = SystemStateMachine()
        assert sm.state == SystemState.INITIALIZING

    def test_valid_transition(self):
        sm = SystemStateMachine()
        sm.transition(SystemState.LOADING_MODELS, reason="test")
        assert sm.state == SystemState.LOADING_MODELS

    def test_illegal_transition_raises(self):
        sm = SystemStateMachine()
        with pytest.raises(ValueError, match="Illegal state transition"):
            sm.transition(SystemState.LOCKED)  # can't jump from INIT to LOCKED

    def test_full_happy_path(self):
        sm = SystemStateMachine()
        path = [
            SystemState.LOADING_MODELS,
            SystemState.CAMERA_READY,
            SystemState.MISSION_READY,
            SystemState.RUNNING,
            SystemState.TRACKING,
            SystemState.LOCKED,
        ]
        for state in path:
            sm.transition(state)
        assert sm.state == SystemState.LOCKED

    def test_any_state_to_error(self):
        sm = SystemStateMachine()
        sm.transition(SystemState.LOADING_MODELS)
        sm.transition(SystemState.ERROR, reason="model load failed")
        assert sm.state == SystemState.ERROR

    def test_any_state_to_shutdown(self):
        sm = SystemStateMachine()
        sm.transition(SystemState.LOADING_MODELS)
        sm.transition(SystemState.SHUTDOWN, reason="operator shutdown")
        assert sm.state == SystemState.SHUTDOWN

    def test_shutdown_is_terminal(self):
        sm = SystemStateMachine()
        sm.transition(SystemState.LOADING_MODELS)
        sm.transition(SystemState.SHUTDOWN)
        with pytest.raises(ValueError):
            sm.transition(SystemState.LOADING_MODELS)

    def test_can_transition(self):
        sm = SystemStateMachine()
        assert sm.can_transition(SystemState.LOADING_MODELS) is True
        assert sm.can_transition(SystemState.LOCKED) is False

    def test_history_grows(self):
        sm = SystemStateMachine()
        sm.transition(SystemState.LOADING_MODELS)
        sm.transition(SystemState.CAMERA_READY)
        assert len(sm.history) == 3  # INIT + 2 transitions

    def test_listener_called(self):
        sm = SystemStateMachine()
        transitions = []
        sm.add_listener(lambda prev, new: transitions.append((prev, new)))
        sm.transition(SystemState.LOADING_MODELS)
        sm.transition(SystemState.CAMERA_READY)
        assert len(transitions) == 2
        assert transitions[0] == (SystemState.INITIALIZING, SystemState.LOADING_MODELS)

    def test_remove_listener(self):
        sm = SystemStateMachine()
        calls = []
        fn = lambda prev, new: calls.append(new)
        sm.add_listener(fn)
        sm.transition(SystemState.LOADING_MODELS)
        sm.remove_listener(fn)
        sm.transition(SystemState.CAMERA_READY)
        assert len(calls) == 1   # only first transition was captured

    def test_time_in_state(self):
        import time
        sm = SystemStateMachine()
        time.sleep(0.05)
        elapsed = sm.time_in_state()
        assert elapsed >= 0.04
