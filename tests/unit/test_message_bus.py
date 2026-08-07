"""
Unit Tests — Message Bus
"""

import asyncio
import pytest

from apex.engine.bus.message_bus import MessageBus
from apex.engine.bus.channels import Ch


@pytest.fixture(autouse=True)
def reset_bus():
    MessageBus.reset()
    yield
    MessageBus.reset()


class TestMessageBus:

    @pytest.mark.asyncio
    async def test_singleton(self):
        a = MessageBus.instance()
        b = MessageBus.instance()
        assert a is b

    @pytest.mark.asyncio
    async def test_publish_and_subscribe(self):
        bus = MessageBus.instance()
        received = []

        async def collect():
            async for _, msg in bus.subscribe(Ch.DETECTIONS):
                received.append(msg)
                break  # stop after first message

        task = asyncio.create_task(collect())
        await asyncio.sleep(0)  # yield to let subscriber register
        await bus.publish(Ch.DETECTIONS, "test_message")
        await asyncio.wait_for(task, timeout=1.0)

        assert received == ["test_message"]

    @pytest.mark.asyncio
    async def test_wildcard_subscription(self):
        bus = MessageBus.instance()
        received = []

        async def collect():
            async for ch, msg in bus.subscribe("/camera/*"):
                received.append((ch, msg))
                if len(received) >= 2:
                    break

        task = asyncio.create_task(collect())
        await asyncio.sleep(0)
        await bus.publish("/camera/frame", "frame1")
        await bus.publish("/camera/usb0/frame", "frame2")
        await asyncio.wait_for(task, timeout=1.0)

        assert len(received) == 2
        channels = [r[0] for r in received]
        assert "/camera/frame" in channels
        assert "/camera/usb0/frame" in channels

    @pytest.mark.asyncio
    async def test_multiple_subscribers(self):
        bus = MessageBus.instance()
        results_a = []
        results_b = []

        async def sub_a():
            async for _, msg in bus.subscribe(Ch.TRACKS):
                results_a.append(msg)
                break

        async def sub_b():
            async for _, msg in bus.subscribe(Ch.TRACKS):
                results_b.append(msg)
                break

        tasks = [asyncio.create_task(sub_a()), asyncio.create_task(sub_b())]
        await asyncio.sleep(0)
        count = await bus.publish(Ch.TRACKS, "track_data")
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=1.0)

        assert count == 2
        assert results_a == ["track_data"]
        assert results_b == ["track_data"]

    @pytest.mark.asyncio
    async def test_no_subscribers_returns_zero(self):
        bus = MessageBus.instance()
        count = await bus.publish("/nonexistent/channel", "data")
        assert count == 0

    @pytest.mark.asyncio
    async def test_channel_stats(self):
        bus = MessageBus.instance()
        await bus.publish(Ch.HEALTH, "snap1")
        await bus.publish(Ch.HEALTH, "snap2")
        stats = bus.channel_stats()
        assert stats.get(Ch.HEALTH, 0) == 2

    @pytest.mark.asyncio
    async def test_handler_callback(self):
        bus = MessageBus.instance()
        received = []

        async def handler(channel: str, msg) -> None:
            received.append(msg)

        bus.on(Ch.SYSTEM_EVENTS, handler)
        await asyncio.sleep(0)
        await bus.publish(Ch.SYSTEM_EVENTS, "event1")
        await bus.publish(Ch.SYSTEM_EVENTS, "event2")
        await asyncio.sleep(0.1)

        assert "event1" in received
        assert "event2" in received

    @pytest.mark.asyncio
    async def test_ch_camera_frame_helper(self):
        ch = Ch.camera_frame("cam0")
        assert ch == "/camera/cam0/frame"
