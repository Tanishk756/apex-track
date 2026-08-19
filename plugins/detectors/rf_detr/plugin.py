"""
RF-DETR 2XL High-Precision Transformer Neural Detector Plugin
===============================================================
Leverages high-capacity Real-Time DEtection TRansformer (RF-DETR 2XL) backbone
for ultra-high mAP detection of small aerial drones and high-speed targets.

Licensing:
- Codebase & Adapter: Apache-2.0
- RF-DETR 2XL Pretrained Weights / Plus Package: PML-1.0 (Permissive Model License 1.0)
- Requires explicit user configuration `accept_pml1_license: true` to load restricted weights.
"""

from __future__ import annotations

import time
from typing import Any, Optional, Tuple
import numpy as np
import structlog

from apex.engine.contracts.detection import BoundingBox, Detection
from apex.engine.contracts.frame import Frame
from apex.engine.detector.detector_base import DetectorBase, IGNORED_CLUTTER_CLASSES
from apex.engine.plugins.plugin_base import PluginMetadata, PluginType, PluginStatus

log = structlog.get_logger(__name__)


class RFDetr2XLPlugin(DetectorBase):
    """RF-DETR 2XL Transformer Neural Object Detector Plugin."""

    metadata = PluginMetadata(
        name="rf_detr_2xl",
        version="2.1.0",
        plugin_type=PluginType.DETECTOR,
        license="PML-1.0 (Model) / Apache-2.0 (Adapter)",
        author="APEX-Track",
        description="Official RF-DETR 2XL Transformer High-Resolution Object Detector Engine",
    )

    def __init__(self) -> None:
        super().__init__()
        self.model_name = "rfdetr-2xl"
        self.conf_threshold = 0.25
        self.pml1_accepted = False
        self._rf_detr_model: Any = None

    async def load(self, config: dict[str, Any], hw_profile: Any) -> None:
        """
        Initialize RF-DETR 2XL transformer model with PML-1.0 license compliance check.
        """
        await super().load(config, hw_profile)
        self.conf_threshold = float(config.get("confidence_threshold", 0.25))

        # Check license acceptance flag
        self.pml1_accepted = bool(
            config.get("accept_pml1_license", False)
            or config.get("accept_restricted_licenses", False)
        )

        if not self.pml1_accepted:
            log.warning(
                "rf_detr_2xl_license_not_accepted",
                message=(
                    "RF-DETR 2XL model weights require accepting the PML-1.0 license terms. "
                    "Set 'accept_pml1_license: true' in config to enable official 2XL weights. "
                    "Falling back to standard RT-DETR / base transformer inference."
                ),
            )

        # Attempt loading official rfdetr package if available
        try:
            if self.pml1_accepted:
                import rfdetr  # type: ignore
                log.info("loading_official_rf_detr_2xl", model=self.model_name)
                self._rf_detr_model = rfdetr.load_model("rfdetr-2xl", device=getattr(hw_profile, "device", "cpu"))
            else:
                self._rf_detr_model = None
        except ImportError:
            log.info("official_rfdetr_package_not_installed_using_rtdetr_fallback")
            self._rf_detr_model = None
        except Exception as exc:
            log.warning("rf_detr_2xl_load_failed", error=str(exc))
            self._rf_detr_model = None

    def _preprocess(self, frame: Frame) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        return frame.data, 1.0, (0, 0)

    def _postprocess(self, raw_outputs: Any, frame: Frame, scale: float, pad: Tuple[int, int]) -> list[Detection]:
        return []

    def detect(self, frame: Frame) -> list[Detection]:
        """Runs RF-DETR 2XL Transformer inference with airborne target mapping."""
        if self._rf_detr_model is not None:
            # Delegate to official rfdetr model API
            try:
                preds = self._rf_detr_model.predict(frame.data, conf=self.conf_threshold)
                detections: list[Detection] = []
                for p in preds:
                    x1, y1, x2, y2 = p["bbox"]
                    score = float(p["confidence"])
                    cls_name = str(p.get("class_name", "target"))
                    det = Detection(
                        bbox=BoundingBox(x1, y1, x2, y2),
                        confidence=score,
                        class_id=int(p.get("class_id", 0)),
                        class_name=cls_name,
                        camera_id=frame.metadata.camera_id,
                        detector_id=self.metadata.name,
                        frame_timestamp=frame.timestamp,
                    )
                    detections.append(det)
                return detections
            except Exception as exc:
                log.warning("rf_detr_official_predict_failed", error=str(exc))

        # Base fallback path
        return super().detect(frame)
