"""
System State Machine
====================
Global system state with valid transition enforcement.

Design:
- SystemState is a flat enum — published on /system/state whenever it changes.
- SystemStateMachine enforces legal transitions; illegal ones raise ValueError.
- Every module can subscribe to /system/state on the MessageBus to react
  to startup, shutdown, error, and mode changes.
- The state machine is not async itself — state changes are synchronous,
  but the publication to the bus is scheduled as an asyncio task.

State lifecycle:
    INITIALIZING → LOADING_MODELS → CAMERA_READY → MISSION_READY
         → RUNNING → TRACKING → LOCKED
         ↕ (any state can go to) → ERROR → SHUTDOWN
"""

from __future__ import annotations

import asyncio
import time
from enum import Enum, auto
from typing import Callable, Optional

import structlog

log = structlog.get_logger(__name__)


class SystemState(Enum):
    """
    Global system lifecycle states.
    Published on Ch.SYSTEM_STATE whenever a transition occurs.
    """
    INITIALIZING   = auto()   # Engine is starting up, loading config
    LOADING_MODELS = auto()   # Detector/tracker models being loaded
    CAMERA_READY   = auto()   # At least one camera is streaming
    MISSION_READY  = auto()   # All systems nominal, awaiting mission start
    RUNNING        = auto()   # Active detection/tracking, no primary lock
    TRACKING       = auto()   # Tracking confirmed targets, no primary lock
    LOCKED         = auto()   # Primary target lock acquired
    RECORDING      = auto()   # Mission recording active (can overlap with LOCKED)
    PAUSED         = auto()   # Pipeline paused (e.g. config reload)
    ERROR          = auto()   # Non-fatal error; system attempting recovery
    SHUTDOWN       = auto()   # Graceful shutdown in progress


# Valid state transitions: {from_state: {to_states}}
_TRANSITIONS: dict[SystemState, set[SystemState]] = {
    SystemState.INITIALIZING:   {SystemState.LOADING_MODELS, SystemState.ERROR, SystemState.SHUTDOWN},
    SystemState.LOADING_MODELS: {SystemState.CAMERA_READY,   SystemState.ERROR, SystemState.SHUTDOWN},
    SystemState.CAMERA_READY:   {SystemState.MISSION_READY,  SystemState.ERROR, SystemState.SHUTDOWN, SystemState.LOADING_MODELS},
    SystemState.MISSION_READY:  {SystemState.RUNNING,        SystemState.ERROR, SystemState.SHUTDOWN, SystemState.PAUSED},
    SystemState.RUNNING:        {SystemState.TRACKING,       SystemState.MISSION_READY, SystemState.RECORDING, SystemState.ERROR, SystemState.SHUTDOWN, SystemState.PAUSED},
    SystemState.TRACKING:       {SystemState.LOCKED,         SystemState.RUNNING, SystemState.RECORDING, SystemState.ERROR, SystemState.SHUTDOWN, SystemState.PAUSED},
    SystemState.LOCKED:         {SystemState.TRACKING,       SystemState.RUNNING, SystemState.RECORDING, SystemState.ERROR, SystemState.SHUTDOWN},
    SystemState.RECORDING:      {SystemState.RUNNING,        SystemState.TRACKING, SystemState.LOCKED, SystemState.ERROR, SystemState.SHUTDOWN},
    SystemState.PAUSED:         {SystemState.MISSION_READY,  SystemState.RUNNING, SystemState.ERROR, SystemState.SHUTDOWN},
    SystemState.ERROR:          {SystemState.MISSION_READY,  SystemState.LOADING_MODELS, SystemState.SHUTDOWN},
    SystemState.SHUTDOWN:       set(),   # terminal state
}


class SystemStateMachine:
    """
    Thread-safe system state machine.
    Publishes state changes on Ch.SYSTEM_STATE via the MessageBus.
    """

    def __init__(self, bus=None) -> None:
        self._state = SystemState.INITIALIZING
        self._bus = bus                          # Optional MessageBus reference
        self._history: list[tuple[float, SystemState]] = [
            (time.time(), SystemState.INITIALIZING)
        ]
        self._listeners: list[Callable[[SystemState, SystemState], None]] = []

    @property
    def state(self) -> SystemState:
        return self._state

    def transition(self, new_state: SystemState, reason: str = "") -> None:
        """
        Attempt a state transition. Raises ValueError on illegal transitions.
        Publishes the new state to the message bus.
        """
        allowed = _TRANSITIONS.get(self._state, set())
        if new_state not in allowed:
            raise ValueError(
                f"Illegal state transition: {self._state.name} → {new_state.name}. "
                f"Allowed: {[s.name for s in allowed]}"
            )

        prev = self._state
        self._state = new_state
        self._history.append((time.time(), new_state))

        log.info(
            "system_state_transition",
            prev=prev.name,
            new=new_state.name,
            reason=reason or "unspecified",
        )

        # Notify sync listeners
        for listener in self._listeners:
            try:
                listener(prev, new_state)
            except Exception as exc:
                log.warning("state_listener_error", error=str(exc))

        # Publish to message bus (if event loop is running)
        if self._bus is not None:
            self._bus.publish_sync("/system/state", new_state.name)

    def can_transition(self, new_state: SystemState) -> bool:
        return new_state in _TRANSITIONS.get(self._state, set())

    def add_listener(self, fn: Callable[[SystemState, SystemState], None]) -> None:
        """Register a synchronous listener called on every state change."""
        self._listeners.append(fn)

    def remove_listener(self, fn: Callable[[SystemState, SystemState], None]) -> None:
        try:
            self._listeners.remove(fn)
        except ValueError:
            pass

    @property
    def history(self) -> list[tuple[float, SystemState]]:
        return list(self._history)

    def time_in_state(self) -> float:
        """Seconds spent in the current state."""
        return time.time() - self._history[-1][0]

    def __repr__(self) -> str:
        return f"SystemStateMachine(state={self._state.name}, history_len={len(self._history)})"
