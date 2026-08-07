"""
Unit Tests — Target Database & Geolocation (Phase 7)
"""

import pytest
import numpy as np

from apex.engine.contracts.detection import BoundingBox
from apex.engine.contracts.track import Track, TrackState
from apex.engine.db.target_database import TargetDatabase
from apex.engine.spatial.geofence import GeofenceEngine, GeofencePolygon
from apex.engine.spatial.geolocation import WorldCoordinateSystem


class TestTargetDatabase:

    def test_db_upsert_and_history(self):
        db = TargetDatabase(db_path=":memory:")

        box = BoundingBox(10, 10, 50, 50)
        track = Track(
            track_id=1,
            state=TrackState.CONFIRMED,
            bbox=box,
            predicted_bbox=box,
            confidence=0.9,
            class_id=0,
            class_name="drone",
            frame_timestamp=100.0,
            speed_kmh=85.0,
        )

        db.update_tracks([track])

        active = db.get_active_targets()
        assert len(active) == 1
        assert active[0].track_id == 1

        history = db.get_track_history(1)
        assert len(history) == 1
        assert history[0][0] == box.cx
        assert history[0][1] == box.cy

        threat = db.compute_threat_level(track)
        assert 0.0 <= threat <= 1.0
        assert threat > 0.5  # High threat drone target at 85 km/h


class TestWorldCoordinateSystem:

    def test_pixel_to_world_projection(self):
        wcs = WorldCoordinateSystem(focal_length_mm=35.0, sensor_width_mm=36.0, sensor_height_mm=24.0)

        # Center pixel projection with drone at (37.7749, -122.4194) alt 100m looking down at -45 deg
        lat, lon, alt = wcs.pixel_to_world(
            pixel_x=320,
            pixel_y=240,
            image_w=640,
            image_h=480,
            uav_lat=37.7749,
            uav_lon=-122.4194,
            uav_alt_m=100.0,
            gimbal_pitch_deg=-45.0,
            gimbal_yaw_deg=0.0,
        )

        assert isinstance(lat, float)
        assert isinstance(lon, float)
        assert abs(lat - 37.7749) < 0.1
        assert abs(lon - (-122.4194)) < 0.1


class TestGeofenceEngine:

    def test_geofence_breach_detection(self):
        engine = GeofenceEngine()
        poly = GeofencePolygon(
            name="No-Fly-Zone Alpha",
            vertices_lat_lon=[
                (37.770, -122.420),
                (37.780, -122.420),
                (37.780, -122.410),
                (37.770, -122.410),
            ],
            is_restricted=True,
        )
        engine.add_geofence(poly)

        box = BoundingBox(10, 10, 50, 50)
        track = Track(
            track_id=1,
            state=TrackState.CONFIRMED,
            bbox=box,
            predicted_bbox=box,
            confidence=0.9,
            class_id=0,
            class_name="drone",
            frame_timestamp=100.0,
            world_point=(37.775, -122.415, 0.0),  # Inside polygon!
        )

        events = engine.evaluate([track])
        assert len(events) == 1
        assert events[0].source == "GeofenceEngine"
        assert "No-Fly-Zone Alpha" in events[0].payload["geofence_name"]
