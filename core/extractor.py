"""
Character glyph extraction: raw calligraphy image → clean binary mask.

Pipeline: grayscale → adaptive binarization → denoise → bbox crop → quality check
Output:   mode "L" image, 255 = stroke pixel, 0 = background
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Adaptive threshold starting parameters
_DEFAULT_BLOCK_SIZE = 15
_DEFAULT_C = 8

# Quality bounds: stroke coverage ratio
_MIN_COVERAGE = 0.05
_MAX_COVERAGE = 0.95


class CharExtractor:
    """Extract a clean stroke mask from a raw calligraphy image."""

    def extract(self, img: Image.Image) -> Image.Image:
        """
        Input:  raw calligraphy image (any mode — RGB, L, RGBA, etc.)
        Output: mode "L" mask cropped to bounding box.
                255 = stroke, 0 = background.
        """
        gray = self._to_grayscale(img)
        binary = self._binarize(gray)

        # Quality check — retry with adjusted params if needed
        coverage = np.count_nonzero(binary) / binary.size
        if coverage < _MIN_COVERAGE or coverage > _MAX_COVERAGE:
            logger.info(
                "Coverage %.1f%% out of range, retrying with Otsu", coverage * 100
            )
            binary = self._binarize_otsu(gray)

        binary = self._denoise(binary)
        cropped = self._crop_bbox(binary)

        return Image.fromarray(cropped, mode="L")

    # ── internal steps ───────────────────────────────────────

    @staticmethod
    def _to_grayscale(img: Image.Image) -> np.ndarray:
        """Convert any PIL image to uint8 grayscale numpy array."""
        if img.mode == "L":
            return np.array(img)
        if img.mode == "RGBA":
            # Composite onto white, then grayscale
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            return np.array(bg.convert("L"))
        return np.array(img.convert("L"))

    @staticmethod
    def _binarize(
        gray: np.ndarray,
        block_size: int = _DEFAULT_BLOCK_SIZE,
        c: int = _DEFAULT_C,
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
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary

    @staticmethod
    def _denoise(binary: np.ndarray) -> np.ndarray:
        """Remove small isolated noise blobs via morphological opening."""
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
