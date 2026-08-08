"""
Autonomous Threat Priority Matrix & Auto-Lock Engine
====================================================
Evaluates active targets against tactical risk metrics:
- Approach velocity & closing speed
- Target class severity (UAV/Drone = 1.0, Vehicle = 0.7, Person = 0.4)
- Proximity & Time-To-Collision (TTC)

Ranks targets into ALPHA (Critical), BRAVO (Elevated), and CHARLIE (Low).
Automatically designates the highest ALPHA threat for Optical Tracking Lock.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any
import structlog

from apex.engine.contracts.track import Track

log = structlog.get_logger(__name__)

CLASS_SEVERITY_WEIGHTS: Dict[str, float] = {
    "drone": 1.0,
    "uav": 1.0,
    "airplane": 0.9,
    "vehicle": 0.7,
    "car": 0.7,
    "truck": 0.8,
    "bus": 0.8,
    "motorcycle": 0.6,
    "person": 0.4,
    "keyboard": 0.2,
    "laptop": 0.2,
    "monitor": 0.2,
}


class ThreatMatrixEngine:
    """Dynamic Threat Matrix Evaluator."""

    def __init__(self) -> None:
        self.primary_lock_id: Optional[int] = None

    def evaluate_threats(
        self, tracks: List[Track], trajectory_data: Dict[int, Dict[str, Any]], frame_w: int = 1280, frame_h: int = 720
    ) -> Dict[str, Any]:
        """
        Evaluates 0-100 threat score for all tracks and determines priority lock target.
        """
        threat_scores: Dict[int, Dict[str, Any]] = {}
        highest_score = -1.0
        alpha_target_id: Optional[int] = None

        cx_center = frame_w / 2.0
        cy_center = frame_h / 2.0

        for tr in tracks:
            tid = tr.track_id
            cls = tr.class_name.lower().strip()
            class_w = CLASS_SEVERITY_WEIGHTS.get(cls, 0.5)

            # Proximity metric (0.0 to 1.0)
            dx = tr.bbox.cx - cx_center
            dy = tr.bbox.cy - cy_center
            dist = (dx**2 + dy**2) ** 0.5
            max_dist = (cx_center**2 + cy_center**2) ** 0.5
            prox_score = max(0.0, 1.0 - (dist / max_dist))

            # Velocity & trajectory metrics
            traj = trajectory_data.get(tid, {})
            speed = traj.get("speed_px", 0.0)
            is_closing = traj.get("is_closing", False)
            ttc = traj.get("ttc_seconds", 999.0)

            speed_w = min(1.0, speed / 30.0)
            closing_w = 1.5 if is_closing else 1.0

            # Calculate composite Threat Score (0.0 to 100.0)
            raw_score = (class_w * 40.0 + prox_score * 35.0 + speed_w * 25.0) * closing_w
            threat_score = min(100.0, round(raw_score, 1))

            # Categorize threat tier
            if threat_score >= 70.0:
                level = "ALPHA"
            elif threat_score >= 40.0:
                level = "BRAVO"
            else:
                level = "CHARLIE"

            threat_scores[tid] = {
                "score": threat_score,
                "level": level,
                "class": tr.class_name,
                "is_closing": is_closing,
                "ttc": ttc,
            }

            if threat_score > highest_score:
                highest_score = threat_score
                alpha_target_id = tid

        # Auto-lock onto ALPHA target if no manual lock override
        if alpha_target_id is not None and highest_score >= 60.0:
            self.primary_lock_id = alpha_target_id

        return {
            "threat_matrix": threat_scores,
            "primary_lock_id": self.primary_lock_id,
            "alpha_target_id": alpha_target_id,
            "max_threat_score": max(0.0, highest_score),
        }
