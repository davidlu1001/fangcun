"""
Character glyph extraction: raw calligraphy image → clean binary mask.

Pipeline:
  1. _normalize_to_black_on_white()  — three-tier polarity defense
  2. Otsu / adaptive binarization (source-aware)
  2.5 _remove_seal_frame()          — for 印谱 sources only
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

    def __init__(self) -> None:
        self._detected_as_yinpu = False

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
        self._detected_as_yinpu = False

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

        # Step 2.5: remove seal frame lines for 印谱 sources
        is_yinpu = source_name in KNOWN_YINPU_SOURCES or self._detected_as_yinpu
        if is_yinpu:
            binary = self._remove_seal_frame(binary)

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

        Tier 1: Known 印谱 source whitelist — O(1), alpha hole extraction
        Tier 2: Alpha semantic detection — check bright-pixel ratio
                 in opaque region (印谱 has white stroke slots in black block)
        Tier 3: Morphological erosion fallback — erode black areas;
                 large surviving blocks = dark background, thin strokes vanish
        """
        # ── Tier 1: whitelist ────────────────────────────────
        if source_name in KNOWN_YINPU_SOURCES:
            logger.info("Tier 1: 印谱白名单 '%s', alpha孔洞提取", source_name)
            self._detected_as_yinpu = True
            return self._extract_yinpu_strokes(img)

        gray = self._composite_to_gray(img)

        # ── Tier 2: alpha semantic detection ─────────────────
        if img.mode in ("RGBA", "LA"):
            alpha = np.array(img.split()[-1])

            fully_transparent_ratio = float((alpha < 10).sum()) / alpha.size

            if fully_transparent_ratio > 0.08:
                opaque_mask = alpha > 128
                opaque_count = int(opaque_mask.sum())

                if opaque_count > 200:
                    opaque_gray = gray[opaque_mask]
                    light_ratio = float((opaque_gray > 200).sum()) / len(opaque_gray)

                    if light_ratio > 0.15:
                        logger.info(
                            "Tier 2: 印谱结构检测 light_ratio=%.2f, alpha孔洞提取",
                            light_ratio,
                        )
                        self._detected_as_yinpu = True
                        return self._extract_yinpu_strokes(img)

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

    # ── 印谱 extraction ────────────────────────────────────

    @staticmethod
    def _extract_yinpu_strokes(img: Image.Image) -> np.ndarray:
        """
        印谱专用提取：笔画 = 不透明 bbox 内的透明孔洞。

        1. 找到所有不透明像素 (α>128) 的 bounding box
        2. bbox 内：α<128 → 笔画槽 → 黑 (0)
        3. bbox 内：α>128 → 印面 → 白 (255)
        4. bbox 外（透明外边距）→ 白 (255)，不参与笔画识别
        """
        if img.mode not in ("RGBA", "LA"):
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img)
            return 255 - np.array(bg.convert("L"))

        alpha = np.array(img.split()[-1])
        opaque = alpha > 128

        coords = np.argwhere(opaque)
        if coords.size == 0:
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[-1])
            return np.array(bg.convert("L"))

        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)

        gray = np.full(alpha.shape, 255, dtype=np.uint8)

        bbox_alpha = alpha[y0 : y1 + 1, x0 : x1 + 1]
        gray[y0 : y1 + 1, x0 : x1 + 1] = np.where(
            bbox_alpha < 128, 0, 255
        ).astype(np.uint8)

        return gray

    # ── Seal frame removal ───────────────────────────────────

    @staticmethod
    def _remove_seal_frame(binary: np.ndarray) -> np.ndarray:
        """
        Remove rectangular seal frame lines from 印谱 binary mask.

        Uses 1D pixel projection instead of connected-component analysis.
        CCA fails when frame pixels and stroke pixels are physically connected
        (粘连), merging into one large component where fill_ratio/aspect are
        useless. Projection-based detection is immune to connectivity.

        Algorithm:
          1. Project rows/cols: count white pixels per row and per column
          2. Frame lines span full width/height → pixel count > 50% threshold
             (no calligraphy stroke can reach 50% of a full row in a seal)
          3. Only scan outer 15% (top/bottom for rows, left/right for cols)
             — center zone is absolutely protected
          4. Convolve FIRST to dilate thick/double frames, THEN hard-truncate
             center zone (prevents dilation from leaking inward)

        Protection for 「一」: its pixels are in the center zone (>15% from
        edges), so the scan region never includes them.
        """
        h, w = binary.shape
        result = binary.copy()

        threshold_w = w * 0.50
        threshold_h = h * 0.50

        scan_y = max(5, int(h * 0.15))
        scan_x = max(5, int(w * 0.15))

        erase_r = max(3, int(min(w, h) * 0.015))
        kernel = np.ones(2 * erase_r + 1, dtype=int)

        # ── Y-axis projection: top/bottom frame lines ────────
        row_sums = (binary > 0).sum(axis=1).astype(float)

        frame_rows = np.zeros(h, dtype=bool)
        frame_rows[:scan_y] = row_sums[:scan_y] > threshold_w
        frame_rows[h - scan_y :] = row_sums[h - scan_y :] > threshold_w

        if frame_rows.any():
            # Convolve FIRST (dilate each hit independently)
            dilated_rows = np.convolve(frame_rows.astype(int), kernel, mode="same") > 0
            # THEN hard-truncate center — prevents edge dilation from leaking in
            dilated_rows[scan_y : h - scan_y] = False
            result[dilated_rows, :] = 0
            logger.info(
                "移除上下印框: %d rows erased (scan_y=%d, erase_r=%d)",
                int(dilated_rows.sum()),
                scan_y,
                erase_r,
            )

        # ── X-axis projection: left/right frame lines ────────
        col_sums = (binary > 0).sum(axis=0).astype(float)

        frame_cols = np.zeros(w, dtype=bool)
        frame_cols[:scan_x] = col_sums[:scan_x] > threshold_h
        frame_cols[w - scan_x :] = col_sums[w - scan_x :] > threshold_h

        if frame_cols.any():
            dilated_cols = np.convolve(frame_cols.astype(int), kernel, mode="same") > 0
            dilated_cols[scan_x : w - scan_x] = False
            result[:, dilated_cols] = 0
            logger.info(
                "移除左右印框: %d cols erased (scan_x=%d, erase_r=%d)",
                int(dilated_cols.sum()),
                scan_x,
                erase_r,
            )

        return result

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
