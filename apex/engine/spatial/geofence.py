"""
Geofence Spatial Engine
=======================
Evaluates tactical geofences, restricted airspace zones, and polygon breaches.
Emits alert events on TargetDatabase updates when targets breach operational boundaries.
"""

from __future__ import annotations

from typing import Sequence
import numpy as np

from apex.engine.contracts.event import ApexEvent, EventSeverity, EventType
from apex.engine.contracts.track import Track


class GeofencePolygon:
    """Represents a 2D spherical/GPS polygon perimeter."""

    def __init__(self, name: str, vertices_lat_lon: Sequence[tuple[float, float]], is_restricted: bool = True) -> None:
        self.name = name
        self.vertices = list(vertices_lat_lon)
        self.is_restricted = is_restricted

    def contains_point(self, lat: float, lon: float) -> bool:
        """Ray-casting algorithm to test point inclusion in polygon."""
        n = len(self.vertices)
        inside = False
        p1lat, p1lon = self.vertices[0]

        for i in range(n + 1):
            p2lat, p2lon = self.vertices[i % n]
            if lon > min(p1lon, p2lon):
                if lon <= max(p1lon, p2lon):
                    if lat <= max(p1lat, p2lat):
                        if p1lon != p2lon:
                            xinters = (lon - p1lon) * (p2lat - p1lat) / (p2lon - p1lon) + p1lat
                        if p1lat == p2lat or lat <= xinters:
                            inside = not inside
            p1lat, p1lon = p2lat, p2lon

        return inside


class GeofenceEngine:
    """Evaluates spatial polygon rules across active tracks."""

    def __init__(self) -> None:
        self.geofences: list[GeofencePolygon] = []

    def add_geofence(self, geofence: GeofencePolygon) -> None:
        self.geofences.append(geofence)

    def evaluate(self, tracks: Sequence[Track]) -> list[ApexEvent]:
        """Check all active tracks against active geofences."""
        events: list[ApexEvent] = []

        for track in tracks:
            if not track.world_point:
                continue

            lat, lon, _ = track.world_point
            for gf in self.geofences:
                if gf.contains_point(lat, lon):
                    severity = EventSeverity.CRITICAL if gf.is_restricted else EventSeverity.WARNING
                    evt = ApexEvent(
                        type=EventType.THREAT_ALERT,
                        severity=severity,
                        source="GeofenceEngine",
                        payload={
                            "track_id": track.track_id,
                            "geofence_name": gf.name,
                            "lat": lat,
                            "lon": lon,
                            "message": f"Target #{track.track_id} ({track.class_name}) breached geofence: {gf.name}",
                        },
                    )
                    events.append(evt)

        return events
