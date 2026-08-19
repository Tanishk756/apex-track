"""
APEX-Track Roboflow API Engine
==============================
Integrates Roboflow Hosted Inference and Dataset Harvester APIs
using the operator's active Roboflow Private API Key.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional
import numpy as np
import structlog

from apex.engine.config.security import SecurityManager, mask_key

log = structlog.get_logger(__name__)


class RoboflowEngine:
    """Roboflow API Client for Hosted Inference & Dataset Harvesting."""

    _instance: Optional[RoboflowEngine] = None

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or SecurityManager.instance().roboflow_key
        self._rf_client: Any = None
        self._active_model: Any = None
        self.active_model_id: str = "uav-detection/3"
        self._init_client()

    @classmethod
    def instance(cls) -> RoboflowEngine:
        if cls._instance is None:
            cls._instance = RoboflowEngine()
        return cls._instance

    def _init_client(self) -> None:
        try:
            import roboflow
            self._rf_client = roboflow.Roboflow(api_key=self.api_key)
            log.info("roboflow_client_initialized", api_key_masked=self.api_key[:6] + "...")
        except Exception as exc:
            log.warning("roboflow_init_failed", error=str(exc))
            self._rf_client = None

    def infer_image(self, image_np: np.ndarray, model_id: Optional[str] = None, conf: float = 0.25) -> List[Dict[str, Any]]:
        """
        Runs neural object detection via Roboflow Hosted Inference API.
        """
        target_model = model_id or self.active_model_id
        if not self._rf_client:
            return []

        try:
            # Parse workspace/project/version e.g. "uav-detection/3" or "workspace/project/3"
            parts = target_model.strip().split("/")
            if len(parts) == 2:
                proj_name, ver = parts[0], int(parts[1])
                model_obj = self._rf_client.workspace().project(proj_name).version(ver).model
            elif len(parts) == 3:
                ws, proj_name, ver = parts[0], parts[1], int(parts[2])
                model_obj = self._rf_client.workspace(ws).project(proj_name).version(ver).model
            else:
                return []

            # Save temporary image for API inference call
            import cv2
            tmp_path = "/tmp/roboflow_infer_input.jpg"
            cv2.imwrite(tmp_path, image_np)

            res = model_obj.predict(tmp_path, confidence=int(conf * 100)).json()
            predictions = res.get("predictions", [])
            log.info("roboflow_inference_success", model=target_model, count=len(predictions))
            return predictions
        except Exception as exc:
            log.warning("roboflow_inference_failed", error=str(exc))
            return []

    def get_status(self) -> Dict[str, Any]:
        return {
            "status": "ONLINE" if self._rf_client is not None else "OFFLINE",
            "api_key_configured": bool(self.api_key),
            "api_key_masked": mask_key(self.api_key),
            "active_model_id": self.active_model_id,
        }
