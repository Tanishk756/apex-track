"""
Multi-Spectral EO/IR Thermal Vision Fusion Shader
==================================================
Simulates thermal FLIR imagery palettes in real-time over standard EO optical feeds:
- EO (Standard Daylight Optical Feed)
- FLIR_IRONBOW (Military FLIR Thermal Heat Palette)
- FLIR_WHITE_HOT (Infrared White-Hot Highlight Shader)
- FLIR_BLACK_HOT (Infrared Black-Hot Highlight Shader)
- NVG_GREEN (Tactical Night Vision Phosphor Green Shader)
"""

from __future__ import annotations

import cv2
import numpy as np


class ThermalVisionMode:
    EO = "EO"
    FLIR_IRONBOW = "FLIR_IRONBOW"
    FLIR_WHITE_HOT = "FLIR_WHITE_HOT"
    FLIR_BLACK_HOT = "FLIR_BLACK_HOT"
    NVG_GREEN = "NVG_GREEN"


class ThermalFusionShader:
    """Real-time Multi-Spectral Optical/Thermal Vision Palette Converter."""

    def __init__(self, mode: str = ThermalVisionMode.EO) -> None:
        self.mode = mode

    def apply_fusion(self, frame: np.ndarray) -> np.ndarray:
        """Applies chosen thermal vision color map palette to input frame."""
        if self.mode == ThermalVisionMode.EO:
            return frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.mode == ThermalVisionMode.FLIR_IRONBOW:
            # Military FLIR Ironbow Heat Signature
            thermal = cv2.applyColorMap(gray, cv2.COLORMAP_COLORMAP_IRONBOW if hasattr(cv2, "COLORMAP_COLORMAP_IRONBOW") else cv2.COLORMAP_JET)
            return thermal

        elif self.mode == ThermalVisionMode.FLIR_WHITE_HOT:
            # FLIR White-Hot Heat Signature
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thermal = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
            return thermal

        elif self.mode == ThermalVisionMode.FLIR_BLACK_HOT:
            # FLIR Black-Hot Inverted Signature
            inverted = 255 - gray
            thermal = cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR)
            return thermal

        elif self.mode == ThermalVisionMode.NVG_GREEN:
            # Tactical Night Vision Phosphor Green
            nvg = np.zeros_like(frame)
            nvg[:, :, 1] = cv2.equalizeHist(gray)  # Strong green channel
            nvg[:, :, 0] = (gray * 0.1).astype(np.uint8)
            nvg[:, :, 2] = (gray * 0.1).astype(np.uint8)
            return nvg

        return frame
