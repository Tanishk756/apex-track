"""
Defense-Grade Image Enhancement Engine
========================================
Applies real-time image pre-processing (Adaptive CLAHE, contrast normalization, fast-motion sharpening)
to boost low-light, low-contrast, and small-target detection accuracy for 60-120+ km/h dynamic targets.
"""

from __future__ import annotations

import cv2
import numpy as np
import structlog

log = structlog.get_logger(__name__)


class DefenseImageEnhancer:
    """Real-time optical pre-processing pipeline for tactical perception enhancement."""

    def __init__(self, clip_limit: float = 2.5, tile_grid_size: tuple[int, int] = (8, 8)) -> None:
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def enhance_frame(self, frame_bgr: np.ndarray, motion_sharpen: bool = True) -> np.ndarray:
        """
        Enhance BGR frame using Adaptive CLAHE on L-channel (LAB color space)
        and optional unsharp masking to counteract high-speed motion blur.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return frame_bgr

        try:
            # 1. Convert to LAB color space and apply CLAHE to Luminance (L) channel
            lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
            l_chan, a_chan, b_chan = cv2.split(lab)
            enhanced_l = self.clahe.apply(l_chan)
            enhanced_lab = cv2.merge((enhanced_l, a_chan, b_chan))
            enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

            # 2. Unsharp Masking for fast-motion edge sharpening if high velocity detected
            if motion_sharpen:
                gaussian = cv2.GaussianBlur(enhanced_bgr, (0, 0), sigmaX=3.0)
                enhanced_bgr = cv2.addWeighted(enhanced_bgr, 1.3, gaussian, -0.3, 0)

            return enhanced_bgr
        except Exception as e:
            log.warning("frame_enhancement_failed_using_raw", error=str(e))
            return frame_bgr
