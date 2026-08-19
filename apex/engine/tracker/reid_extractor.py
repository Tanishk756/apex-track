"""
Visual Target Re-Identification (Re-ID) Feature Extractor
===========================================================
Extracts normalized compact visual feature embedding vectors ($f \\in \\mathbb{R}^{D}$) from target bounding box image crops.
Combines L2-normalized multi-channel color structure histograms and spatial aspect features.
"""

from __future__ import annotations

import cv2
import numpy as np
import structlog

from apex.engine.contracts.detection import BoundingBox

log = structlog.get_logger(__name__)


class ReIDFeatureExtractor:
    """Extracts compact, invariant visual feature vectors for target re-identification."""

    def __init__(self, feature_dim: int = 128) -> None:
        self.feature_dim = feature_dim

    def extract_crop_feature(self, frame_img: np.ndarray, bbox: BoundingBox) -> np.ndarray:
        """
        Extract L2-normalized feature vector from image crop defined by bounding box.
        """
        if frame_img is None or frame_img.size == 0:
            return np.zeros(self.feature_dim, dtype=np.float32)

        h_img, w_img = frame_img.shape[:2]
        x1 = max(0, min(w_img - 1, int(bbox.x1)))
        y1 = max(0, min(h_img - 1, int(bbox.y1)))
        x2 = max(x1 + 1, min(w_img, int(bbox.x2)))
        y2 = max(y1 + 1, min(h_img, int(bbox.y2)))

        crop = frame_img[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros(self.feature_dim, dtype=np.float32)

        # Resize crop to standard canonical resolution
        resized = cv2.resize(crop, (64, 128))
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)

        # Multi-channel color histogram features (normalized per channel)
        hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
        hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
        hist_l = cv2.calcHist([lab], [0], None, [32], [0, 256]).flatten()
        hist_a = cv2.calcHist([lab], [1], None, [32], [0, 256]).flatten()
        hist_b = cv2.calcHist([lab], [2], None, [32], [0, 256]).flatten()

        def norm_vector(v):
            n = np.linalg.norm(v)
            return v / n if n > 1e-6 else v

        hist_h = norm_vector(hist_h)
        hist_s = norm_vector(hist_s)
        hist_l = norm_vector(hist_l)
        hist_a = norm_vector(hist_a)
        hist_b = norm_vector(hist_b)

        # Spatial structure & aspect ratio feature
        aspect_ratio = float((bbox.height + 1e-5) / (bbox.width + 1e-5))
        rel_area = float((bbox.width * bbox.height) / (w_img * h_img + 1e-5))

        feat_vector = np.concatenate([
            hist_h, hist_s, hist_l, hist_a, hist_b,
            np.array([aspect_ratio, rel_area], dtype=np.float32)
        ])

        # Truncate or pad to requested feature dimension
        if len(feat_vector) < self.feature_dim:
            feat_vector = np.pad(feat_vector, (0, self.feature_dim - len(feat_vector)))
        else:
            feat_vector = feat_vector[:self.feature_dim]

        # L2 Normalization
        norm = np.linalg.norm(feat_vector)
        if norm > 1e-6:
            feat_vector = feat_vector / norm

        return feat_vector.astype(np.float32)

    @staticmethod
    def compute_cosine_similarity(feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Compute Cosine Similarity between two L2-normalized feature vectors."""
        norm1 = np.linalg.norm(feat1)
        norm2 = np.linalg.norm(feat2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0
        dot_product = float(np.dot(feat1, feat2))
        return float(np.clip(dot_product / (norm1 * norm2), -1.0, 1.0))
