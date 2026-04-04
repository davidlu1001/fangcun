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
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

# Validity: absolute pixel count threshold (not percentage — percentage
# fails for characters like "一" where bbox is extremely thin after crop).
_MIN_STROKE_PIXELS = 50


class CharExtractor:
    """Extract a clean stroke mask from a raw calligraphy image."""

    def extract(self, img: Image.Image, source: str = "字典") -> Image.Image:
        """
        Args:
            img:    raw calligraphy image (any mode, any polarity)
            source: '字典' | '真迹' | '本地' — controls preprocessing intensity

        Returns:
            mode "L" mask cropped to bounding box. 255 = stroke, 0 = background.
        """
        # Step 0: normalize to black-on-white regardless of original polarity.
        # This is done HERE (not in scraper) so all characters in a seal
        # go through identical normalization — no per-image inversion drift.
        gray = self._normalize_to_black_on_white(img)

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

        # Quality check — use absolute pixel count, not percentage.
        # Percentage fails for "一" etc. where bbox is extremely thin.
        stroke_pixels = int(np.count_nonzero(binary))
        if stroke_pixels < _MIN_STROKE_PIXELS:
            logger.info(
                "Only %d stroke pixels (source=%s), retrying with Otsu",
                stroke_pixels,
                source,
            )
            binary = self._binarize_otsu(gray)

        binary = self._denoise(binary, strong=(source == "真迹"))
        cropped = self._crop_bbox(binary)

        return Image.fromarray(cropped, mode="L")

    # ── internal steps ───────────────────────────────────────

    @staticmethod
    def _normalize_to_black_on_white(img: Image.Image) -> np.ndarray:
        """
        Convert any image to uint8 grayscale with guaranteed black-on-white
        polarity. Handles:
          - RGBA transparent PNGs (composite onto white)
          - White-on-black / dark-background scans (auto-invert)
          - Normal black-on-white (pass through)

        Detection: center-region average brightness.
          < 128 → dark background → invert
          >= 128 → light background → keep
        """
        # Composite RGBA onto white so transparent pixels → white
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            gray = np.array(bg.convert("L"))
        elif img.mode == "L":
            gray = np.array(img)
        else:
            gray = np.array(img.convert("L"))

        # Detect dark-background images by checking BORDER brightness.
        # Border is mostly background (strokes rarely touch all 4 edges),
        # so it's a more reliable indicator than center (which may be
        # filled by thick strokes like 禅, 蘇, causing false positives).
        h, w = gray.shape
        border = np.concatenate([
            gray[0, :], gray[-1, :],   # top & bottom rows
            gray[:, 0], gray[:, -1],   # left & right columns
        ])

        if border.size > 0 and float(border.mean()) < 128:
            logger.info(
                "Dark background (border=%.0f), inverting to black-on-white",
                border.mean(),
            )
            gray = 255 - gray

        return gray

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
