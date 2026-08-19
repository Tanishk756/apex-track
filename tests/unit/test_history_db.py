"""
Unit Tests — Persistent Historical Detection Storage & Inspection Engine
"""

import os
import tempfile
import pytest

from apex.engine.contracts.detection import BoundingBox
from apex.engine.contracts.track import Track, TrackState
from apex.engine.db.target_database import TargetDatabase


class TestPersistentHistoryDatabase:

    def test_sqlite_persistence_and_querying(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_file = os.path.join(tmp_dir, "test_tracks.db")
            db = TargetDatabase(db_path=db_file)

            tr1 = Track(
                track_id=1,
                state=TrackState.CONFIRMED,
                bbox=BoundingBox(x1=10, y1=20, x2=100, y2=150),
                predicted_bbox=BoundingBox(x1=12, y1=22, x2=102, y2=152),
                confidence=0.92,
                class_id=0,
                class_name="person",
                frame_timestamp=1700000000.0,
                camera_id="cam_0",
                speed_kmh=15.5,
            )

            tr2 = Track(
                track_id=2,
                state=TrackState.CONFIRMED,
                bbox=BoundingBox(x1=200, y1=300, x2=400, y2=500),
                predicted_bbox=BoundingBox(x1=205, y1=305, x2=405, y2=505),
                confidence=0.88,
                class_id=1,
                class_name="truck",
                frame_timestamp=1700000005.0,
                camera_id="cam_0",
                speed_kmh=45.0,
            )

            # Insert tracks into database
            db.update_tracks([tr1, tr2])

            # Query historical records
            records = db.get_historical_records(limit=10)
            assert len(records) == 2
            assert records[0]["track_id"] == 2
            assert records[1]["track_id"] == 1

            # Filter by track_id
            filtered_id = db.get_historical_records(track_id=1)
            assert len(filtered_id) == 1
            assert filtered_id[0]["class_name"] == "person"

            # Filter by class_name
            filtered_class = db.get_historical_records(class_name="truck")
            assert len(filtered_class) == 1
            assert filtered_class[0]["track_id"] == 2

            # Summary metrics
            summary = db.get_history_summary()
            assert summary["total_records"] == 2
            assert summary["unique_targets"] == 2
            assert summary["class_counts"]["person"] == 1
            assert summary["class_counts"]["truck"] == 1
