"""
Mission Manager
===============
Tactical mission orchestrator. Executes mission profiles, handles target locking,
and coordinates Gimbal tracking commands.
"""

from __future__ import annotations

from typing import Optional
import structlog

from apex.engine.contracts.event import ApexEvent, EventSeverity, EventType
from apex.engine.contracts.track import Track
from apex.engine.events.event_engine import EventEngine
from apex.engine.mission.mission_profile import MissionProfile

log = structlog.get_logger(__name__)


class MissionManager:
    """Tactical mission coordinator."""

    def __init__(self, event_engine: EventEngine | None = None) -> None:
        self.event_engine = event_engine or EventEngine()
        self.active_profile: MissionProfile = MissionProfile(name="default", description="Default")
        self.locked_track_id: Optional[int] = None

    def load_mission_profile(self, profile_yaml_path: str) -> MissionProfile:
        """Load and activate a mission profile at runtime."""
        profile = MissionProfile.load_from_yaml(profile_yaml_path)
        self.active_profile = profile

        log.info("mission_profile_activated", profile_name=profile.name)
        self.event_engine.emit_sync(
            ApexEvent(
                type=EventType.MISSION_STARTED,
                source="MissionManager",
                severity=EventSeverity.INFO,
                payload={"profile_name": profile.name},
            )
        )
        return profile

    def acquire_target_lock(self, track_id: int) -> bool:
        """Lock primary tracker gimbal onto target track ID."""
        self.locked_track_id = track_id
        log.info("target_lock_acquired", track_id=track_id)
        self.event_engine.emit_sync(
            ApexEvent(
                type=EventType.LOCK_ACQUIRED,
                source="MissionManager",
                severity=EventSeverity.INFO,
                payload={"track_id": track_id},
            )
        )
        return True

    def release_target_lock(self) -> None:
        """Release active target lock."""
        if self.locked_track_id is not None:
            old_id = self.locked_track_id
            self.locked_track_id = None
            log.info("target_lock_released", track_id=old_id)
            self.event_engine.emit_sync(
                ApexEvent(
                    type=EventType.LOCK_LOST,
                    source="MissionManager",
                    severity=EventSeverity.WARNING,
                    payload={"track_id": old_id},
                )
            )

    def get_locked_target(self, active_tracks: list[Track]) -> Optional[Track]:
        """Retrieve active Track contract object for currently locked target."""
        if self.locked_track_id is None:
            return None
        for t in active_tracks:
            if t.track_id == self.locked_track_id:
                return t

        # Lock lost if track no longer present in active list
        self.release_target_lock()
        return None
