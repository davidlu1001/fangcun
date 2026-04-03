"""
Character glyph extraction: raw calligraphy image → clean binary mask.

Pipeline: grayscale → (denoise if 真迹) → adaptive binarization → denoise → bbox crop → quality check
Output:   mode "L" image, 255 = stroke pixel, 0 = background

Source-aware preprocessing:
  字典 — high-contrast dictionary scans, standard threshold params
  真迹 — original calligraphy with paper noise/yellowing, enhanced denoising
  本地 — local font rendering, clean digital, simple Otsu threshold
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Quality bounds: stroke coverage ratio
_MIN_COVERAGE = 0.05
_MAX_COVERAGE = 0.95


class CharExtractor:
    """Extract a clean stroke mask from a raw calligraphy image."""

    def extract(self, img: Image.Image, source: str = "字典") -> Image.Image:
        """
        Args:
            img:    raw calligraphy image (any mode)
            source: '字典' | '真迹' | '本地' — controls preprocessing intensity

        Returns:
            mode "L" mask cropped to bounding box. 255 = stroke, 0 = background.
        """
        gray = self._to_grayscale(img)

        if source == "真迹":
            # Enhanced preprocessing for noisy originals:
            # bilateral filter preserves edges while smoothing paper texture
            gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
            # Adaptive threshold with block_size proportional to image size —
            # must be large enough to cover full stroke width (50-100+ px).
            short_side = min(gray.shape)
            block = max(short_side // 4, 31) | 1  # ensure odd, ≥31
            binary = self._binarize(gray, block_size=block, c=10)
        else:
            # 字典 and 本地: clean uniform backgrounds → global Otsu is optimal.
            # Adaptive threshold CANNOT be used here: block_size=15 is tiny vs
            # stroke width (50-100+ px), causing only edges to be detected
            # (hollow outlines instead of solid fills).
            binary = self._binarize_otsu(gray)

        # Quality check — retry with Otsu if adaptive produced bad result
        coverage = np.count_nonzero(binary) / binary.size
        if coverage < _MIN_COVERAGE or coverage > _MAX_COVERAGE:
            logger.info(
                "Coverage %.1f%% out of range (source=%s), retrying with Otsu",
                coverage * 100,
                source,
            )
            binary = self._binarize_otsu(gray)

        binary = self._denoise(binary, strong=(source == "真迹"))
        cropped = self._crop_bbox(binary)

        return Image.fromarray(cropped, mode="L")

    # ── internal steps ───────────────────────────────────────

    @staticmethod
    def _to_grayscale(img: Image.Image) -> np.ndarray:
        """Convert any PIL image to uint8 grayscale numpy array."""
        if img.mode == "L":
            return np.array(img)
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            return np.array(bg.convert("L"))
        return np.array(img.convert("L"))

    @staticmethod
    def _binarize(
        gray: np.ndarray,
        block_size: int = 15,
        c: int = 8,
    ) -> np.ndarray:
        """Adaptive Gaussian threshold → binary inverse (stroke = 255)."""
        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            c,
        )

    @staticmethod
    def _binarize_otsu(gray: np.ndarray) -> np.ndarray:
        """Otsu threshold fallback for poor adaptive results."""
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        return binary

    @staticmethod
    def _denoise(binary: np.ndarray, strong: bool = False) -> np.ndarray:
        """Remove noise blobs. strong=True uses larger kernel + extra iteration for 真迹."""
        if strong:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    @staticmethod
    def _crop_bbox(binary: np.ndarray) -> np.ndarray:
        """Crop to tight bounding box around non-zero pixels."""
        coords = cv2.findNonZero(binary)
        if coords is None:
            return binary
        x, y, w, h = cv2.boundingRect(coords)
        return binary[y : y + h, x : x + w]
