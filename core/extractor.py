"""
Character glyph extraction: raw calligraphy image → clean binary mask.

Pipeline:
  1. _normalize_to_black_on_white()  — three-tier polarity defense
  2. Otsu / adaptive binarization (source-aware)
  3. Morphological denoising
  4. Bounding box crop
  5. Quality validation (absolute pixel count for simple chars like 一)

Output: mode "L" image, 255 = stroke pixel, 0 = background
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_MIN_STROKE_PIXELS = 50

# ── Known 印谱 sources (Tier 1 whitelist) ────────────────────
# O(1) lookup. Only add sources after empirical confirmation.
# 「中国篆刻大字典」is normal (阳文), NOT in this set.
KNOWN_YINPU_SOURCES = frozenset({
    "鸟虫篆全书",
    "汉印分韵续集",
    "汉印分韵",
    "金文名品",
    "金文书法集萃",
    "广金石韵府",
    "汉印文字征",
    "六书通",
})


class CharExtractor:
    """Extract a clean stroke mask from a raw calligraphy image."""

    def extract(
        self, img: Image.Image, source: str = "字典", source_name: str = ""
    ) -> Image.Image:
        """
        Args:
            img:         raw calligraphy image (any mode, any polarity)
            source:      '字典' | '真迹' | '本地' — controls binarization params
            source_name: e.g. '鸟虫篆全书' — used for Tier 1 whitelist

        Returns:
            mode "L" mask cropped to bounding box. 255 = stroke, 0 = background.
        """
        # Step 1: normalize polarity (three-tier defense)
        gray = self._normalize_to_black_on_white(img, source_name)

        # Step 2: binarize
        if source == "真迹":
            gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
            short_side = min(gray.shape)
            block = max(short_side // 4, 31) | 1
            binary = self._binarize_adaptive(gray, block_size=block, c=10)
        else:
            binary = self._binarize_otsu(gray)

        # Step 3: quality check (absolute pixel count, not percentage)
        stroke_pixels = int(np.count_nonzero(binary))
        if stroke_pixels < _MIN_STROKE_PIXELS:
            logger.info(
                "Only %d stroke pixels (source=%s), retrying with Otsu",
                stroke_pixels,
                source,
            )
            binary = self._binarize_otsu(gray)

        # Step 4: denoise
        binary = self._denoise(binary, strong=(source == "真迹"))

        # Step 5: crop to bounding box
        cropped = self._crop_bbox(binary)

        return Image.fromarray(cropped, mode="L")

    # ── Three-tier polarity normalization ─────────────────────

    def _normalize_to_black_on_white(
        self, img: Image.Image, source_name: str = ""
    ) -> np.ndarray:
        """
        Guarantee black-on-white output regardless of input polarity.

        Three-tier defense (performance: O(1) → O(N) → O(N)):

        Tier 1: Known 印谱 source whitelist — O(1), direct invert
        Tier 2: Alpha semantic detection — check bright-pixel ratio
                 in opaque region (印谱 has white stroke slots in black block)
        Tier 3: Morphological erosion fallback — erode black areas;
                 large surviving blocks = dark background, thin strokes vanish
        """
        # ── Tier 1: whitelist ────────────────────────────────
        if source_name in KNOWN_YINPU_SOURCES:
            logger.info("Tier 1: known 印谱 source '%s', inverting", source_name)
            return self._composite_and_invert(img)

        gray = self._composite_to_gray(img)

        # ── Tier 2: alpha semantic detection ─────────────────
        if img.mode in ("RGBA", "LA"):
            alpha = np.array(img.split()[-1])

            # Guard against fake transparency (e.g. convert('RGBA') on opaque)
            fully_transparent_ratio = float((alpha < 10).sum()) / alpha.size

            if fully_transparent_ratio > 0.08:
                opaque_mask = alpha > 128
                opaque_count = int(opaque_mask.sum())

                if opaque_count > 200:
                    opaque_gray = gray[opaque_mask]
                    light_ratio = float((opaque_gray > 200).sum()) / len(opaque_gray)

                    if light_ratio > 0.15:
                        logger.info(
                            "Tier 2: opaque region has %.1f%% bright pixels → 印谱, inverting",
                            light_ratio * 100,
                        )
                        return 255 - gray

                    if light_ratio < 0.05:
                        logger.debug(
                            "Tier 2: opaque region %.1f%% bright → normal strokes",
                            light_ratio * 100,
                        )
                        return gray

                    # 0.05–0.15 ambiguous → fall through to Tier 3

        # ── Tier 3: morphological erosion fallback ───────────
        content = gray < 250
        if not content.any():
            return gray

        coords = cv2.findNonZero(content.astype(np.uint8))
        if coords is None:
            return gray
        x, y, w, h = cv2.boundingRect(coords)
        cropped = gray[y : y + h, x : x + w]

        _, binary_inv = cv2.threshold(
            cropped, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Use max(w,h) so extreme aspect ratios (一: 100×20) still work:
        # min=20 → k=3 can't erode 20px stroke; max=100 → k=12 can.
        k = max(5, int(max(w, h) * 0.12))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        eroded = cv2.erode(binary_inv, kernel)

        surviving = cv2.countNonZero(eroded)
        min_block = max(10, int(w * h * 0.01))

        if surviving > min_block:
            logger.info(
                "Tier 3: %d px survived erosion (threshold=%d) → dark block, inverting",
                surviving,
                min_block,
            )
            return 255 - gray

        return gray

    # ── Compositing helpers ──────────────────────────────────

    @staticmethod
    def _composite_to_gray(img: Image.Image) -> np.ndarray:
        """Composite onto white background → uint8 grayscale array."""
        bg = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode in ("RGBA", "LA"):
            bg.paste(img, mask=img.split()[-1])
        else:
            bg.paste(img)
        return np.array(bg.convert("L"))

    @staticmethod
    def _composite_and_invert(img: Image.Image) -> np.ndarray:
        """Composite onto white → grayscale → invert."""
        bg = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode in ("RGBA", "LA"):
            bg.paste(img, mask=img.split()[-1])
        else:
            bg.paste(img)
        return 255 - np.array(bg.convert("L"))

    # ── Binarization ─────────────────────────────────────────

    @staticmethod
    def _binarize_adaptive(
        gray: np.ndarray, block_size: int = 15, c: int = 8
    ) -> np.ndarray:
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block_size, c,
        )

    @staticmethod
    def _binarize_otsu(gray: np.ndarray) -> np.ndarray:
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        return binary

    # ── Post-processing ──────────────────────────────────────

    @staticmethod
    def _denoise(binary: np.ndarray, strong: bool = False) -> np.ndarray:
        if strong:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    @staticmethod
    def _crop_bbox(binary: np.ndarray) -> np.ndarray:
        coords = cv2.findNonZero(binary)
        if coords is None:
            return binary
        x, y, w, h = cv2.boundingRect(coords)
        return binary[y : y + h, x : x + w]
