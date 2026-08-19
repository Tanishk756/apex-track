"""
REMIND (RE-Identification with Memory for INDoor & Spatial Navigation) Engine
=============================================================================
State-of-the-art appearance-based multi-object re-identification tracking system
designed for long-term target persistence in complex indoor and outdoor Environments.

Key Architecture:
1. Dual-Tier Memory Structure:
   - Episodic Memory: Short-term temporal buffer of raw normalized visual appearance embeddings.
   - Semantic Memory: Drift-resistant global appearance centroid updated via EMA (Exponential Moving Average).
2. Spatial-Kinematic Decay Gating:
   - Fuses visual cosine similarity with exponential spatial distance & temporal decay factors.
3. Re-Entry Target Matcher:
   - Enables persistent target unique ID (UID) matching across long occlusions, frame exits, and re-entry.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import structlog

from apex.engine.contracts.detection import BoundingBox
from apex.engine.tracker.reid_extractor import ReIDFeatureExtractor

log = structlog.get_logger(__name__)


class TargetMemoryProfile:
    """Individual target REMIND dual-tier memory container."""

    def __init__(self, uid: int, class_name: str, initial_feature: np.ndarray, initial_center: Tuple[float, float], timestamp: float, max_episodic: int = 20) -> None:
        self.uid = uid
        self.class_name = class_name
        self.max_episodic = max_episodic
        
        # Dual-tier memory banks
        self.episodic_buffer: List[Dict[str, Any]] = [
            {"timestamp": timestamp, "feature": initial_feature.copy(), "center": initial_center}
        ]
        self.semantic_centroid: np.ndarray = initial_feature.copy()
        
        # Motion and status tracking
        self.last_center: Tuple[float, float] = initial_center
        self.last_seen: float = timestamp
        self.reid_hits: int = 0
        self.total_observations: int = 1

    def update(self, feature: np.ndarray, center: Tuple[float, float], timestamp: float, ema_alpha: float = 0.15) -> None:
        """Update episodic memory buffer and semantic centroid via EMA."""
        self.last_center = center
        self.last_seen = timestamp
        self.total_observations += 1

        # 1. Update Episodic Buffer
        self.episodic_buffer.append({"timestamp": timestamp, "feature": feature.copy(), "center": center})
        if len(self.episodic_buffer) > self.max_episodic:
            self.episodic_buffer.pop(0)

        # 2. Update Semantic Centroid via EMA
        fused = (1.0 - ema_alpha) * self.semantic_centroid + ema_alpha * feature
        norm = np.linalg.norm(fused)
        if norm > 1e-6:
            self.semantic_centroid = fused / norm
        else:
            self.semantic_centroid = fused

    def compute_similarity(self, query_feat: np.ndarray, query_center: Tuple[float, float], timestamp: float) -> Tuple[float, float, float]:
        """
        Compute REMIND multi-tiered similarity score.
        Returns: (fused_score, visual_sim, spatial_sim)
        """
        # 1. Semantic Centroid Similarity
        norm_q = np.linalg.norm(query_feat)
        norm_s = np.linalg.norm(self.semantic_centroid)
        if norm_q < 1e-6 or norm_s < 1e-6:
            semantic_sim = 0.0
        else:
            semantic_sim = float(np.clip(np.dot(query_feat, self.semantic_centroid) / (norm_q * norm_s), -1.0, 1.0))

        # 2. Episodic Max Similarity (best match in recent memory buffer)
        episodic_sims = []
        for item in self.episodic_buffer:
            ef = item["feature"]
            norm_e = np.linalg.norm(ef)
            if norm_q > 1e-6 and norm_e > 1e-6:
                episodic_sims.append(float(np.dot(query_feat, ef) / (norm_q * norm_e)))
        max_episodic_sim = max(episodic_sims) if episodic_sims else semantic_sim

        # Fused Visual Similarity
        visual_sim = 0.60 * semantic_sim + 0.40 * max_episodic_sim

        # 3. Spatial-Temporal Kinematic Proximity Decay
        dx = query_center[0] - self.last_center[0]
        dy = query_center[1] - self.last_center[1]
        dist_px = np.hypot(dx, dy)
        dt = max(0.001, timestamp - self.last_seen)

        # Spatial decay scale increases over time to allow re-entry anywhere after long absence
        spatial_scale = 300.0 + min(1200.0, dt * 50.0)
        spatial_sim = float(np.exp(-dist_px / spatial_scale))

        # 4. Total REMIND Fused Similarity
        fused_score = 0.75 * visual_sim + 0.25 * spatial_sim
        return float(fused_score), float(visual_sim), float(spatial_sim)


class REMINDReIDTracker:
    """Master REMIND Persistent Target Re-ID Engine."""

    _instance: Optional["REMINDReIDTracker"] = None

    def __init__(self, sim_threshold: float = 0.68, max_memory_targets: int = 500) -> None:
        self.sim_threshold = sim_threshold
        self.max_memory_targets = max_memory_targets
        self.extractor = ReIDFeatureExtractor()
        self.memory_bank: Dict[int, TargetMemoryProfile] = {}
        self._next_uid = 1
        self.total_reid_hits = 0

    @classmethod
    def instance(cls) -> "REMINDReIDTracker":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    def match_or_register(
        self,
        frame_img: Optional[np.ndarray],
        bbox: BoundingBox,
        class_name: str,
        timestamp: Optional[float] = None,
    ) -> Tuple[int, bool, float]:
        """
        Query REMIND dual-tier memory bank to match existing target or register new persistent UID.
        Returns: (persistent_uid, is_reidentified, match_confidence)
        """
        ts = timestamp if timestamp is not None else time.time()
        cx, cy = bbox.cx, bbox.cy

        if frame_img is None or frame_img.size == 0:
            uid = self._allocate_uid()
            return uid, False, 0.0

        feat = self.extractor.extract_crop_feature(frame_img, bbox)
        if np.all(feat == 0):
            uid = self._allocate_uid()
            return uid, False, 0.0

        best_uid: Optional[int] = None
        best_fused_score: float = -1.0

        # Query all targets in memory bank matching class_name
        for uid, profile in self.memory_bank.items():
            if profile.class_name.lower() != class_name.lower():
                continue

            fused_score, vis_sim, _ = profile.compute_similarity(feat, (cx, cy), ts)
            if fused_score > best_fused_score:
                best_fused_score = fused_score
                best_uid = uid

        # Check if match meets confidence threshold
        if best_uid is not None and best_fused_score >= self.sim_threshold:
            profile = self.memory_bank[best_uid]
            profile.update(feat, (cx, cy), ts)
            profile.reid_hits += 1
            self.total_reid_hits += 1

            log.info(
                "remind_reid_match_success",
                uid=best_uid,
                class_name=class_name,
                confidence=round(best_fused_score, 3),
            )
            return best_uid, True, best_fused_score

        # No persistent match: register new target profile in REMIND memory bank
        new_uid = self._allocate_uid()
        new_profile = TargetMemoryProfile(
            uid=new_uid,
            class_name=class_name,
            initial_feature=feat,
            initial_center=(cx, cy),
            timestamp=ts,
        )
        self.memory_bank[new_uid] = new_profile

        # Evict oldest inactive profile if bank capacity exceeded
        if len(self.memory_bank) > self.max_memory_targets:
            oldest_uid = min(self.memory_bank.keys(), key=lambda k: self.memory_bank[k].last_seen)
            del self.memory_bank[oldest_uid]

        return new_uid, False, 1.0

    def _allocate_uid(self) -> int:
        uid = self._next_uid
        self._next_uid += 1
        return uid

    def get_status(self) -> Dict[str, Any]:
        """Returns REMIND subsystem status diagnostics."""
        active_count = len(self.memory_bank)
        total_hits = self.total_reid_hits
        return {
            "engine": "REMIND (Memory Re-ID)",
            "version": "2.0.0",
            "active_profiles_in_memory": active_count,
            "max_memory_capacity": self.max_memory_targets,
            "total_reid_hits": total_hits,
            "similarity_threshold": self.sim_threshold,
        }

    def get_target_memory(self, uid: int) -> Optional[Dict[str, Any]]:
        """Returns detailed episodic & semantic memory diagnostics for a specific target UID."""
        profile = self.memory_bank.get(uid)
        if profile is None:
            return None
        return {
            "uid": profile.uid,
            "class_name": profile.class_name,
            "episodic_buffer_count": len(profile.episodic_buffer),
            "total_observations": profile.total_observations,
            "reid_hits": profile.reid_hits,
            "last_center": profile.last_center,
            "last_seen": profile.last_seen,
            "semantic_centroid_norm": float(np.linalg.norm(profile.semantic_centroid)),
        }
