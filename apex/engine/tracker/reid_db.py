"""
Persistent Target Re-Identification (Re-ID) Database
=====================================================
Stores target visual feature embeddings by persistent global Unique Identifier (UID).
Matches newly detected targets against historical embeddings using Cosine Similarity to assign
their original UID when re-entering the camera frame.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import time
import numpy as np
import structlog

from apex.engine.contracts.detection import BoundingBox
from apex.engine.tracker.reid_extractor import ReIDFeatureExtractor
from apex.engine.tracker.remind_reid import REMINDReIDTracker

log = structlog.get_logger(__name__)


class PersistentReIDDatabase:
    """Persistent database of historical target visual feature signatures using REMIND Memory Engine."""

    _instance: Optional["PersistentReIDDatabase"] = None

    def __init__(self, sim_threshold: float = 0.68, max_history_per_uid: int = 10) -> None:
        self.sim_threshold = sim_threshold
        self.max_history = max_history_per_uid
        self.remind_tracker = REMINDReIDTracker.instance()
        self.extractor = ReIDFeatureExtractor()
        self._db: Dict[int, Dict[str, Any]] = {}
        self._next_uid = 1

    @classmethod
    def instance(cls) -> "PersistentReIDDatabase":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def match_or_create_uid(
        self,
        frame_img: Optional[np.ndarray],
        bbox: BoundingBox,
        class_name: str,
    ) -> Tuple[int, bool]:
        """
        Query persistent REMIND memory database using target image crop feature.
        
        Returns tuple: (persistent_uid, is_reidentified_boolean)
        """
        uid, is_reid, _ = self.remind_tracker.match_or_register(frame_img, bbox, class_name)
        return uid, is_reid

    def update_feature_for_uid(self, uid: int, frame_img: Optional[np.ndarray], bbox: BoundingBox) -> None:
        """Update existing UID feature memory during active tracking."""
        if frame_img is None:
            return
        feat = self.extractor.extract_crop_feature(frame_img, bbox)
        if not np.all(feat == 0) and uid in self.remind_tracker.memory_bank:
            self.remind_tracker.memory_bank[uid].update(feat, (bbox.cx, bbox.cy), time.time())

    def get_remind_status(self) -> Dict[str, Any]:
        """Return REMIND memory bank status metrics."""
        return self.remind_tracker.get_status()

    def reset(self) -> None:
        """Clear database memory."""
        self._db.clear()
        self._next_uid = 1
        REMINDReIDTracker.reset()
        self.remind_tracker = REMINDReIDTracker.instance()
