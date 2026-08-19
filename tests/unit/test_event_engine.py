"""
Unit Tests — Event Engine
"""

import asyncio
import pytest

from apex.engine.bus.message_bus import MessageBus
from apex.engine.contracts.event import ApexEvent, EventSeverity, EventType
from apex.engine.events.event_engine import EventEngine


@pytest.fixture(autouse=True)
def reset_bus():
    MessageBus.reset()
    yield
    MessageBus.reset()


def make_event(etype: EventType, source: str = "test") -> ApexEvent:
    return ApexEvent(type=etype, source=source)


class TestEventEngine:

    @pytest.mark.asyncio
    async def test_emit_adds_to_log(self):
        engine = EventEngine()
        evt = make_event(EventType.TARGET_DETECTED)
        await engine.emit(evt)
        assert evt in engine.recent_events

    @pytest.mark.asyncio
    async def test_emit_sync_adds_to_log(self):
        engine = EventEngine()
        evt = make_event(EventType.VIDEO_LOST)
        engine.emit_sync(evt)
        assert evt in engine.recent_events

    @pytest.mark.asyncio
    async def test_rolling_log_limit(self):
        engine = EventEngine(log_size=5)
        for i in range(10):
            await engine.emit(make_event(EventType.FRAME_DROP))
        assert len(engine.recent_events) == 5

    @pytest.mark.asyncio
    async def test_on_sync_listener(self):
        engine = EventEngine()
        received = []
        engine.on_sync(EventType.LOCK_ACQUIRED, lambda e: received.append(e))
        evt = make_event(EventType.LOCK_ACQUIRED)
        await engine.emit(evt)
        assert evt in received

    @pytest.mark.asyncio
    async def test_on_sync_listener_not_called_for_other_types(self):
        engine = EventEngine()
        received = []
        engine.on_sync(EventType.LOCK_ACQUIRED, lambda e: received.append(e))
        await engine.emit(make_event(EventType.VIDEO_LOST))
        assert received == []

    @pytest.mark.asyncio
    async def test_events_since(self):
        import time
        engine = EventEngine()
        t0 = time.time()
        await engine.emit(make_event(EventType.CAMERA_CONNECTED))
        await asyncio.sleep(0.01)
        t1 = time.time()
        assert t0 < t1
        await engine.emit(make_event(EventType.TARGET_DETECTED))
        after_t1 = engine.events_since(t1)
        assert len(after_t1) == 1
        assert after_t1[0].type == EventType.TARGET_DETECTED

    @pytest.mark.asyncio
    async def test_error_severity_event(self):
        engine = EventEngine()
        err_evt = ApexEvent(
            type=EventType.ERROR,
            source="test",
            severity=EventSeverity.ERROR,
            payload={"msg": "catastrophic failure"},
        )
        # Should not raise
        await engine.emit(err_evt)
        assert err_evt in engine.recent_events


class TestContracts:
    """Test core contract dataclasses."""

    def test_bounding_box_iou_perfect_overlap(self):
        from apex.engine.contracts.detection import BoundingBox
        a = BoundingBox(0, 0, 100, 100)
        assert a.iou(a) == pytest.approx(1.0)

    def test_bounding_box_iou_no_overlap(self):
        from apex.engine.contracts.detection import BoundingBox
        a = BoundingBox(0, 0, 50, 50)
        b = BoundingBox(60, 60, 100, 100)
        assert a.iou(b) == pytest.approx(0.0)

    def test_bounding_box_from_xywh(self):
        from apex.engine.contracts.detection import BoundingBox
        b = BoundingBox.from_xywh(10, 20, 50, 60)
        assert b.x1 == 10 and b.y1 == 20
        assert b.x2 == 60 and b.y2 == 80

    def test_track_state_is_active(self):
        from apex.engine.contracts.detection import BoundingBox
        from apex.engine.contracts.track import Track, TrackState
        bbox = BoundingBox(0, 0, 100, 100)
        track = Track(
            track_id=1, state=TrackState.CONFIRMED,
            bbox=bbox, predicted_bbox=bbox,
            confidence=0.9, class_id=0, class_name="car",
            frame_timestamp=0.0,
        )
        assert track.is_active() is True

    def test_track_deleted_not_active(self):
        from apex.engine.contracts.detection import BoundingBox
        from apex.engine.contracts.track import Track, TrackState
        bbox = BoundingBox(0, 0, 100, 100)
        track = Track(
            track_id=2, state=TrackState.DELETED,
            bbox=bbox, predicted_bbox=bbox,
            confidence=0.0, class_id=0, class_name="car",
            frame_timestamp=0.0,
        )
        assert track.is_active() is False
