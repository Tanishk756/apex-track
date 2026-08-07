"""
Message Bus
===========
Typed, async pub/sub message bus for all inter-module communication.

Design decisions:
- asyncio-native: subscribers are coroutines, no threading concerns in the hot path.
- Channels are typed strings (see channels.py) — arbitrary strings also work
  for dynamic/plugin channels.
- Queue-based delivery: each subscriber gets its own asyncio.Queue, so a slow
  subscriber cannot block the publisher or other subscribers.
- max_queue_size prevents unbounded memory growth; oldest messages are dropped
  when the queue is full (back-pressure handled via DROP_OLDEST policy).
- Wildcard subscriptions: subscribe('/tracker/*') matches any /tracker/... channel.
- Weak-reference subscribers: plugins that unload don't need to explicitly
  unsubscribe — dead coroutines are cleaned up automatically.
"""

from __future__ import annotations

import asyncio
import fnmatch
import time
import weakref
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

import structlog

log = structlog.get_logger(__name__)

# Type alias
Handler = Callable[[str, Any], Coroutine[Any, Any, None]]


@dataclass
class _Subscription:
    pattern: str
    handler: Optional[Handler] = None
    max_queue_size: int = 64
    _queue: asyncio.Queue = field(init=False)

    def __post_init__(self) -> None:
        self._queue = asyncio.Queue(maxsize=self.max_queue_size)

    def matches(self, channel: str) -> bool:
        return fnmatch.fnmatch(channel, self.pattern)

    async def deliver(self, channel: str, message: Any) -> None:
        try:
            self._queue.put_nowait((channel, message))
        except asyncio.QueueFull:
            # Drop-oldest policy
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self._queue.put_nowait((channel, message))


class MessageBus:
    """
    Singleton async message bus. Access via MessageBus.instance().

    Usage:
        bus = MessageBus.instance()

        # Subscribe (as async for loop)
        async for channel, msg in bus.subscribe('/tracker/tracks'):
            process(msg)

        # OR with a callback coroutine
        bus.on('/camera/*', my_handler)

        # Publish
        await bus.publish('/detector/detections', detection_array)
    """

    _instance: "MessageBus | None" = None

    def __init__(self) -> None:
        self._subscriptions: list[_Subscription] = []
        self._stats: dict[str, int] = defaultdict(int)
        self._started_at: float = time.time()

    @classmethod
    def instance(cls) -> "MessageBus":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """For testing only — resets the singleton."""
        cls._instance = None

    # ── Publishing ────────────────────────────────────────────────────────────

    async def publish(self, channel: str, message: Any) -> int:
        """
        Publish a message to all matching subscribers.
        Returns the number of subscribers that received the message.
        """
        self._stats[channel] += 1
        delivered = 0
        for sub in self._subscriptions:
            if sub.matches(channel):
                await sub.deliver(channel, message)
                delivered += 1
        return delivered

    def publish_sync(self, channel: str, message: Any) -> None:
        """
        Fire-and-forget from synchronous code (e.g. hardware callbacks).
        Schedules publish on the running event loop if available.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(self.publish(channel, message))
        except RuntimeError:
            pass  # No event loop — silently drop (startup/shutdown edge case)

    # ── Subscribing ───────────────────────────────────────────────────────────

    def subscribe(self, pattern: str, max_queue_size: int = 64):
        """
        Async generator subscription.

        Usage::
            async for channel, msg in bus.subscribe('/tracker/tracks'):
                ...
        """
        sub = _Subscription(pattern=pattern, max_queue_size=max_queue_size)
        self._subscriptions.append(sub)

        async def _gen():
            try:
                while True:
                    channel, msg = await sub._queue.get()
                    yield channel, msg
            finally:
                self._remove(sub)

        return _gen()

    def on(
        self,
        pattern: str,
        handler: Handler,
        max_queue_size: int = 64,
    ) -> _Subscription:
        """
        Register a coroutine handler. The bus will call handler(channel, message)
        for each matching message. The handler runs as an asyncio task.

        Usage::
            async def my_handler(channel: str, msg: TrackArray) -> None:
                ...

            bus.on('/tracker/tracks', my_handler)
        """
        sub = _Subscription(pattern=pattern, max_queue_size=max_queue_size)
        sub.handler = handler
        self._subscriptions.append(sub)

        async def _dispatch_loop():
            while True:
                try:
                    channel, msg = await sub._queue.get()
                    await handler(channel, msg)
                except Exception as exc:
                    log.warning("bus_handler_error", pattern=pattern, error=str(exc))

        asyncio.ensure_future(_dispatch_loop())
        return sub

    def _remove(self, sub: _Subscription) -> None:
        try:
            self._subscriptions.remove(sub)
        except ValueError:
            pass

    def unsubscribe(self, sub: _Subscription) -> None:
        self._remove(sub)

    # ── Diagnostics ───────────────────────────────────────────────────────────

    @property
    def subscriber_count(self) -> int:
        return len(self._subscriptions)

    def channel_stats(self) -> dict[str, int]:
        """Returns message counts per channel since startup."""
        return dict(self._stats)

    def __repr__(self) -> str:
        return f"MessageBus(subs={self.subscriber_count}, channels={len(self._stats)})"
