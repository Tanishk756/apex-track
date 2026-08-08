"""
Tactical Black Box Recording Engine
====================================
Logs telemetry data, bounding boxes, threat scores, and mission events to JSON mission logs.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Any
import structlog

log = structlog.get_logger(__name__)


class BlackboxRecorder:
    """Persists real-time mission telemetry and HUD metadata logs."""

    def __init__(self, log_dir: str = "recordings/blackbox") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_log_file = self.log_dir / f"mission_log_{int(time.time())}.jsonl"

    def record_frame_event(self, frame_id: int, tracks: List[Any], threat_data: Dict[str, Any], thermal_mode: str) -> None:
        """Appends structured frame telemetry to mission log file."""
        event_record = {
            "timestamp": time.time(),
            "frame_id": frame_id,
            "thermal_mode": thermal_mode,
            "target_count": len(tracks),
            "max_threat_score": threat_data.get("max_threat_score", 0.0),
            "primary_lock_id": threat_data.get("primary_lock_id", None),
            "targets": [
                {
                    "track_id": t.track_id,
                    "class_name": t.class_name,
                    "confidence": t.confidence,
                    "bbox": [t.bbox.x1, t.bbox.y1, t.bbox.x2, t.bbox.y2],
                    "speed_kmh": getattr(t, "speed_kmh", 0.0),
                }
                for t in tracks
            ],
        }

        try:
            with open(self.current_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event_record) + "\n")
        except Exception as e:
            log.error("blackbox_record_failed", error=str(e))
