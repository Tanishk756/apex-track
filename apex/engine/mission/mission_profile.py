"""
Mission Profile & Manager
=========================
Defines tactical mission profiles and runtime switching engine.

Mission Profiles specify:
- Primary & secondary detector plugin choice (RT-DETR, RTMDet, YOLO11)
- Tracker plugin choice (ByteTrack, BoT-SORT)
- Class filter lists and confidence thresholds
- Gimbal tracking control policies
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml
import structlog

log = structlog.get_logger(__name__)


@dataclass
class MissionProfile:
    """Tactical mission profile configuration."""

    name: str
    description: str
    detector_plugin: str = "rtdetr"
    tracker_plugin: str = "bytetrack"
    confidence_threshold: float = 0.45
    nms_threshold: float = 0.45
    target_classes: list[str] = field(default_factory=lambda: ["person", "car", "truck", "drone", "airplane"])
    use_ensemble: bool = False
    gimbal_lead_targeting: bool = True
    speed_mode_kmh: int = 100

    @classmethod
    def load_from_yaml(cls, yaml_path: str | Path) -> MissionProfile:
        path = Path(yaml_path)
        if not path.exists():
            log.warning("mission_yaml_not_found_using_default", path=str(path))
            return cls(name="default", description="Fallback Default Mission")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return cls(
            name=data.get("name", path.stem),
            description=data.get("description", ""),
            detector_plugin=data.get("detector_plugin", "rtdetr"),
            tracker_plugin=data.get("tracker_plugin", "bytetrack"),
            confidence_threshold=float(data.get("confidence_threshold", 0.45)),
            nms_threshold=float(data.get("nms_threshold", 0.45)),
            target_classes=list(data.get("target_classes", ["person", "car", "truck", "drone"])),
            use_ensemble=bool(data.get("use_ensemble", False)),
            gimbal_lead_targeting=bool(data.get("gimbal_lead_targeting", True)),
            speed_mode_kmh=int(data.get("speed_mode_kmh", 100)),
        )
