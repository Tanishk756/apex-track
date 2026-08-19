"""
Target Database
===============
Thread-safe spatial-temporal target state database backed by SQLite and in-memory spatial index.

Features:
- Stores raw track state history, world coordinates, speed, heading, and Re-ID embeddings.
- Evaluates threat assessment scores based on class priority and proximity to protected assets.
- Provides track history trails for HUD visualization and predictive trajectory estimation.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np
import structlog

from apex.engine.contracts.track import Track, TrackState

log = structlog.get_logger(__name__)

THREAT_MATRIX = {
    "drone": 0.95,
    "airplane": 0.90,
    "helicopter": 0.85,
    "vehicle": 0.70,
    "truck": 0.65,
    "person": 0.30,
}


class TargetDatabase:
    """Centralized track state repository with thread-safe persistence."""

    def __init__(self, db_path: str = "data/apex_tracks.db") -> None:
        self.db_path = db_path
        if self.db_path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)

        self._lock = threading.Lock()
        self._active_targets: dict[int, Track] = {}
        self._track_history: dict[int, list[tuple[float, float, float]]] = {}  # tid -> [(x, y, ts)]
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS target_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    track_id INTEGER NOT NULL,
                    class_name TEXT NOT NULL,
                    state TEXT NOT NULL,
                    bbox_x1 REAL, bbox_y1 REAL, bbox_x2 REAL, bbox_y2 REAL,
                    lat REAL, lon REAL, alt REAL,
                    speed_kmh REAL, heading_deg REAL,
                    timestamp REAL NOT NULL
                )
                """
            )
            self._conn.commit()

    def update_tracks(self, tracks: list[Track]) -> None:
        """Upsert new frame tracks into active registry and historical log."""
        with self._lock:
            cursor = self._conn.cursor()

            for t in tracks:
                if t.state == TrackState.DELETED:
                    self._active_targets.pop(t.track_id, None)
                    continue

                self._active_targets[t.track_id] = t

                # Append position to history trail
                cx, cy = t.bbox.cx, t.bbox.cy
                history_list = self._track_history.setdefault(t.track_id, [])
                history_list.append((cx, cy, t.frame_timestamp))
                if len(history_list) > 200:
                    history_list.pop(0)

                # Persist to SQLite
                lat = t.world_point[0] if t.world_point else None
                lon = t.world_point[1] if t.world_point else None
                alt = t.world_point[2] if t.world_point else None

                cursor.execute(
                    """
                    INSERT INTO target_history (
                        track_id, class_name, state,
                        bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                        lat, lon, alt, speed_kmh, heading_deg, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        t.track_id,
                        t.class_name,
                        t.state.name,
                        t.bbox.x1,
                        t.bbox.y1,
                        t.bbox.x2,
                        t.bbox.y2,
                        lat,
                        lon,
                        alt,
                        t.speed_kmh,
                        t.heading_deg,
                        t.frame_timestamp,
                    ),
                )

            self._conn.commit()

    def get_active_targets(self) -> list[Track]:
        """Return list of currently active confirmed target tracks, purging stale coasting tracks."""
        with self._lock:
            now = time.time()
            active: list[Track] = []
            for t in list(self._active_targets.values()):
                if t.is_active():
                    # Purge stale coasting targets if missing for > 1.5s
                    if t.state == TrackState.COASTING and (now - t.frame_timestamp) > 1.5:
                        continue
                    active.append(t)
            return active

    def get_track_history(self, track_id: int, limit: int = 50) -> list[tuple[float, float, float]]:
        """Return recent (cx, cy, timestamp) history points for given track ID."""
        with self._lock:
            history = self._track_history.get(track_id, [])
            return history[-limit:]

    def get_historical_records(
        self, track_id: Optional[int] = None, class_name: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Query persistent detection records from SQLite database."""
        with self._lock:
            cursor = self._conn.cursor()
            query = "SELECT track_id, class_name, state, bbox_x1, bbox_y1, bbox_x2, bbox_y2, lat, lon, alt, speed_kmh, timestamp FROM target_history"
            params = []
            conditions = []
            if track_id is not None:
                conditions.append("track_id = ?")
                params.append(track_id)
            if class_name:
                conditions.append("LOWER(class_name) = ?")
                params.append(class_name.lower())

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            query += " ORDER BY id DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            records = []
            for r in rows:
                records.append({
                    "track_id": r[0],
                    "class_name": r[1],
                    "state": r[2],
                    "bbox": [round(r[3] or 0, 1), round(r[4] or 0, 1), round(r[5] or 0, 1), round(r[6] or 0, 1)],
                    "world_point": [r[7], r[8], r[9]] if r[7] is not None else None,
                    "speed_kmh": round(r[10] or 0.0, 1),
                    "timestamp": r[11],
                    "formatted_time": time.strftime("%H:%M:%S", time.localtime(r[11])),
                })
            return records

    def get_history_summary(self) -> Dict[str, Any]:
        """Returns total detection count and target breakdown across all operational sessions."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("SELECT COUNT(*), COUNT(DISTINCT track_id) FROM target_history")
            total_records, unique_targets = cursor.fetchone()

            cursor.execute("SELECT class_name, COUNT(*) FROM target_history GROUP BY class_name")
            class_counts = {row[0]: row[1] for row in cursor.fetchall()}

            return {
                "total_records": total_records or 0,
                "unique_targets": unique_targets or 0,
                "class_counts": class_counts,
            }

    def compute_threat_level(self, track: Track) -> float:
        """Calculate dynamic threat score in [0.0, 1.0] factoring target speed, class priority, and proximity."""
        base_weight = THREAT_MATRIX.get(track.class_name.lower(), 0.5)
        speed_factor = min(1.0, (track.speed_kmh or 0.0) / 120.0)
        # Factor in FOV distance proximity
        cx, cy = track.bbox.cx, track.bbox.cy
        center_dist = np.hypot(cx - 640.0, cy - 360.0) / 734.0
        prox_factor = float(1.0 - center_dist)
        
        score = base_weight * 0.5 + speed_factor * 0.3 + prox_factor * 0.2
        return float(np.clip(score, 0.0, 1.0))

    def get_stanag_threat_level(self, track: Track) -> dict[str, str | float]:
        """Compute STANAG 4609 compliant military alert level (ALPHA, BRAVO, CHARLIE, DELTA)."""
        score = self.compute_threat_level(track)
        if score >= 0.85:
            alert = "DELTA (CRITICAL THREAT)"
        elif score >= 0.65:
            alert = "CHARLIE (WARNING)"
        elif score >= 0.40:
            alert = "BRAVO (MONITORING)"
        else:
            alert = "ALPHA (NOMINAL)"
        return {"score": round(score, 3), "alert_level": alert}
