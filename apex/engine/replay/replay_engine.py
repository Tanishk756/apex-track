"""
Mission Replay Engine
=====================
Deterministic timestamp-synchronized mission log replay system.

Features:
- Parses blackbox mission event logs (JSONL).
- Playback controls: play(), pause(), step(), seek(timestamp), set_speed(speed).
- Real-time (1.0x), accelerated (2.0x, 5.0x), or slowed (0.25x, 0.5x) playback rate.
- Publishes replayed events to MessageBus channels.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import time
from typing import Any, Callable, Optional
import structlog

from apex.engine.bus.channels import Ch
from apex.engine.bus.message_bus import MessageBus

log = structlog.get_logger(__name__)


class ReplayEngine:
    """Deterministic timestamp-synchronized mission log replay engine."""

    def __init__(self, bus: Optional[MessageBus] = None) -> None:
        self.bus = bus or MessageBus.instance()
        self.log_file_path: Optional[Path] = None
        self._records: list[dict[str, Any]] = []
        self._current_index = 0
        self._playback_speed = 1.0
        self._is_playing = False
        self._is_paused = False
        self._replay_task: Optional[asyncio.Task] = None
        self._on_event_callback: Optional[Callable[[dict[str, Any]], None]] = None

    def load_recording(self, file_path: str | Path) -> int:
        """Load JSONL mission log recording."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Recording file not found: {file_path}")

        self.log_file_path = path
        self._records.clear()

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    try:
                        data = json.loads(line_str)
                        self._records.append(data)
                    except json.JSONDecodeError:
                        continue

        # Sort records chronologically by timestamp if available
        self._records.sort(key=lambda r: r.get("timestamp", 0.0))
        self._current_index = 0
        log.info("mission_recording_loaded", path=str(path), records=len(self._records))
        return len(self._records)

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    @property
    def is_paused(self) -> bool:
        return self._is_paused

    @property
    def playback_speed(self) -> float:
        return self._playback_speed

    @property
    def total_records(self) -> int:
        return len(self._records)

    @property
    def current_index(self) -> int:
        return self._current_index

    def set_speed(self, speed: float) -> None:
        """Set replay playback speed multiplier (0.1x to 10.0x)."""
        self._playback_speed = max(0.1, min(10.0, float(speed)))
        log.info("replay_speed_updated", speed=self._playback_speed)

    async def play(self) -> None:
        """Start async replay loop."""
        if not self._records:
            log.warning("cannot_play_empty_recording")
            return

        if self._is_playing:
            self._is_paused = False
            return

        self._is_playing = True
        self._is_paused = False
        self._replay_task = asyncio.create_task(self._replay_loop())
        log.info("replay_started", total_events=len(self._records))

    def pause(self) -> None:
        """Pause playback."""
        self._is_paused = True
        log.info("replay_paused", index=self._current_index)

    def resume(self) -> None:
        """Resume playback."""
        self._is_paused = False
        log.info("replay_resumed", index=self._current_index)

    async def stop(self) -> None:
        """Stop replay and reset playback index."""
        self._is_playing = False
        self._is_paused = False
        if self._replay_task:
            self._replay_task.cancel()
            try:
                await self._replay_task
            except asyncio.CancelledError:
                pass
            self._replay_task = None
        self._current_index = 0
        log.info("replay_stopped")

    def step(self) -> Optional[dict[str, Any]]:
        """Advance single event step forward in recording."""
        if self._current_index >= len(self._records):
            return None

        event = self._records[self._current_index]
        self._current_index += 1
        self._dispatch_event(event)
        return event

    def seek(self, timestamp: float) -> int:
        """Seek to record matching target timestamp."""
        for i, r in enumerate(self._records):
            if r.get("timestamp", 0.0) >= timestamp:
                self._current_index = i
                return i
        self._current_index = len(self._records)
        return self._current_index

    async def _replay_loop(self) -> None:
        """Internal async playback loop matching original timestamp deltas scaled by speed."""
        try:
            last_ts: Optional[float] = None
            while self._is_playing and self._current_index < len(self._records):
                if self._is_paused:
                    await asyncio.sleep(0.05)
                    continue

                event = self._records[self._current_index]
                ts = float(event.get("timestamp", time.time()))

                if last_ts is not None:
                    delta = (ts - last_ts) / self._playback_speed
                    if delta > 0:
                        await asyncio.sleep(min(delta, 1.0))  # Cap max sleep delta to 1.0s

                last_ts = ts
                self._dispatch_event(event)
                self._current_index += 1

            self._is_playing = False
            log.info("replay_completed", total_dispatched=self._current_index)
        except asyncio.CancelledError:
            self._is_playing = False

    def _dispatch_event(self, event: dict[str, Any]) -> None:
        """Publish replayed event to MessageBus and callback."""
        ch = event.get("channel", Ch.SYSTEM_EVENTS)
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.bus.publish(ch, event))
        except RuntimeError:
            pass

        if self._on_event_callback:
            self._on_event_callback(event)
