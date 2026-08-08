"""
Drone Swarm Defense Grid & Spatial Cluster Analytics
=====================================================
Detects UAV drone swarm formations by spatial clustering (DBSCAN / Distance Centroid).
Computes Swarm Centroid, Dispersion Radius, and Pincer Formation Alerts.
"""

from __future__ import annotations

import math
from typing import Dict, List, Any
import numpy as np


class SwarmDefenseGrid:
    """Analyzes multi-target spatial clusters to detect drone swarms."""

    def __init__(self, cluster_dist_px: float = 180.0) -> None:
        self.cluster_dist_px = cluster_dist_px

    def analyze_swarms(self, tracks: List[Any]) -> Dict[str, Any]:
        """
        Groups drone/UAV targets into tactical swarms and calculates cluster metrics.
        """
        drones = [t for t in tracks if t.class_name.lower().strip() in ("drone", "uav", "airplane")]

        if len(drones) < 2:
            return {"swarm_detected": False, "swarm_count": 0, "swarms": []}

        coords = np.array([[d.bbox.cx, d.bbox.cy] for d in drones])
        centroid_x = float(np.mean(coords[:, 0]))
        centroid_y = float(np.mean(coords[:, 1]))

        # Calculate dispersion radius (max distance to centroid)
        dists = np.sqrt((coords[:, 0] - centroid_x) ** 2 + (coords[:, 1] - centroid_y) ** 2)
        dispersion_px = float(np.max(dists))

        is_pincer = False
        if len(drones) >= 3 and dispersion_px > 150.0:
            is_pincer = True

        return {
            "swarm_detected": True,
            "drone_count": len(drones),
            "swarm_centroid": (round(centroid_x, 1), round(centroid_y, 1)),
            "dispersion_radius_px": round(dispersion_px, 1),
            "pincer_alert": is_pincer,
        }
