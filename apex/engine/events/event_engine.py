"""
Event Engine
============
High-level event pub/sub built on top of the MessageBus.

The EventEngine is a thin convenience layer that:
1. Accepts ApexEvent objects (not raw messages)
2. Routes them to the correct bus channel based on EventType category
3. Provides a synchronous emit() for non-async callers
4. Keeps a rolling event log for the health monitor and GUI

The MessageBus handles all the actual delivery mechanics.
The EventEngine handles domain semantics.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Callable, Optional

import structlog

from apex.engine.bus.channels import Ch
from apex.engine.bus.message_bus import MessageBus
from apex.engine.contracts.event import ApexEvent, EventSeverity, EventType

log = structlog.get_logger(__name__)

# Route EventTypes to their bus channel
_EVENT_CHANNEL_MAP: dict[str, str] = {
    # Camera events → system channel
    "CAMERA_CONNECTED":    Ch.SYSTEM_EVENTS,
    "CAMERA_DISCONNECTED": Ch.SYSTEM_EVENTS,
    "VIDEO_LOST":          Ch.SYSTEM_EVENTS,
    "VIDEO_RECOVERED":     Ch.SYSTEM_EVENTS,
    "FRAME_DROP":          Ch.SYSTEM_EVENTS,
    # Tracker events
    "TARGET_DETECTED":     Ch.TRACK_EVENTS,
    "TARGET_LOST":         Ch.TRACK_EVENTS,
    "TARGET_REACQUIRED":   Ch.TRACK_EVENTS,
    "TRACK_COASTING":      Ch.TRACK_EVENTS,
    "TRACK_DELETED":       Ch.TRACK_EVENTS,
    "LOW_CONFIDENCE":      Ch.TRACK_EVENTS,
    "OCCLUDED":            Ch.TRACK_EVENTS,
    # Mission events
    "ZONE_ENTERED":        Ch.MISSION_EVENTS,
    "ZONE_EXITED":         Ch.MISSION_EVENTS,
    "MISSION_STARTED":     Ch.MISSION_EVENTS,
    "MISSION_COMPLETE":    Ch.MISSION_EVENTS,
    "MISSION_ABORTED":     Ch.MISSION_EVENTS,
    "THREAT_ALERT":        Ch.MISSION_EVENTS,
    "LOCK_ACQUIRED":       Ch.MISSION_EVENTS,
    "LOCK_LOST":           Ch.MISSION_EVENTS,
    "LOCK_SWITCHED":       Ch.MISSION_EVENTS,
    # Default → system events
}


def _channel_for(event: ApexEvent) -> str:
    return _EVENT_CHANNEL_MAP.get(event.type.name, Ch.SYSTEM_EVENTS)


class EventEngine:
    """
    System-wide event hub. Wraps the MessageBus for ApexEvent semantics.

    Usage (async)::
        engine = EventEngine(bus)
        await engine.emit(ApexEvent(EventType.TARGET_DETECTED, source="bytetrack"))

    Usage (sync, from hardware callback)::
        engine.emit_sync(ApexEvent(EventType.VIDEO_LOST, source="usb_camera.0"))

    Subscribing::
        async for _, event in engine.subscribe(EventType.TARGET_DETECTED):
            handle(event)
    """

    def __init__(self, bus: Optional[MessageBus] = None, log_size: int = 500) -> None:
        self._bus = bus or MessageBus.instance()
        self._log: deque[ApexEvent] = deque(maxlen=log_size)
        self._listeners: dict[str, list[Callable[[ApexEvent], None]]] = {}

    async def emit(self, event: ApexEvent) -> None:
        """Publish an event asynchronously."""
        self._log.append(event)
        self._call_sync_listeners(event)

        channel = _channel_for(event)
        await self._bus.publish(channel, event)

        # Log at appropriate level
        if event.severity == EventSeverity.ERROR:
            log.error("apex_event", type=event.type.name, source=event.source, **event.payload)
        elif event.severity == EventSeverity.WARNING:
            log.warning("apex_event", type=event.type.name, source=event.source)
        else:
            log.debug("apex_event", type=event.type.name, source=event.source)

    def emit_sync(self, event: ApexEvent) -> None:
        """Emit from synchronous (non-async) code."""
        self._log.append(event)
        self._call_sync_listeners(event)
        self._bus.publish_sync(_channel_for(event), event)

    def subscribe(self, *event_types: EventType):
        """
        Async generator for specific event types across all channels.

        Usage::
            async for _, event in engine.subscribe(EventType.TARGET_DETECTED, EventType.TARGET_LOST):
                ...
        """
        wanted = {et.name for et in event_types}

        async def _gen():
            # Subscribe to all event channels
            async for channel, event in self._bus.subscribe("/*/events"):
                if isinstance(event, ApexEvent):
                    if not wanted or event.type.name in wanted:
                        yield channel, event

        return _gen()

    def on_sync(self, event_type: EventType, fn: Callable[[ApexEvent], None]) -> None:
        """Register a synchronous listener for a specific event type."""
        key = event_type.name
        if key not in self._listeners:
            self._listeners[key] = []
        self._listeners[key].append(fn)

    def _call_sync_listeners(self, event: ApexEvent) -> None:
        for fn in self._listeners.get(event.type.name, []):
            try:
                fn(event)
            except Exception as exc:
                log.warning("event_listener_error", error=str(exc))

    @property
    def recent_events(self) -> list[ApexEvent]:
        """Returns the rolling event log (newest last)."""
        return list(self._log)

    def events_since(self, since_ts: float) -> list[ApexEvent]:
        return [e for e in self._log if e.timestamp >= since_ts]

    def __repr__(self) -> str:
        return f"EventEngine(log_size={len(self._log)})"
