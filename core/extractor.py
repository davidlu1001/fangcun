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

    def _remove_seal_frame(self, binary: np.ndarray) -> np.ndarray:
        """
        Remove seal frame lines via 1D projection, anchored to content bbox.

        Key fix: old version used full-image dimensions (W×0.50) as threshold.
        When the image has large transparent padding (bbox_w << W), the frame's
        actual row_sum is far below W×0.50, so frames were never detected.

        New version finds the content bounding box first, then uses bbox
        dimensions for threshold (70%) and scan region (15%). Padding-immune.

        Threshold 70% (not 50%): frame lines span ~100% of bbox width,
        while calligraphy strokes rarely exceed 70%. Safe margin.

        Execution order: convolve FIRST, then hard-truncate center zone.
        Reverse order lets dilation leak past the protection boundary.
        """
        h, w = binary.shape

        # ── Step 0: find content bbox ────────────────────────
        content = binary > 0
        if not content.any():
            return binary.copy()

        rows_any = np.where(content.any(axis=1))[0]
        cols_any = np.where(content.any(axis=0))[0]
        by0, by1 = int(rows_any[0]), int(rows_any[-1])
        bx0, bx1 = int(cols_any[0]), int(cols_any[-1])
        bbox_h = by1 - by0 + 1
        bbox_w = bx1 - bx0 + 1

        # ── Step 1: bbox-based thresholds ────────────────────
        threshold_w = bbox_w * 0.70
        threshold_h = bbox_h * 0.70
        scan_rows = max(3, int(bbox_h * 0.15))
        scan_cols = max(3, int(bbox_w * 0.15))
        erase_r = max(2, int(min(bbox_w, bbox_h) * 0.015))
        kernel = np.ones(2 * erase_r + 1, dtype=int)

        result = binary.copy()

        # ── Step 2: Y-axis projection (top/bottom frames) ───
        row_sums = content.sum(axis=1).astype(float)

        frame_rows = np.zeros(h, dtype=bool)
        top_end = by0 + scan_rows
        bot_start = by1 - scan_rows + 1

        frame_rows[by0:top_end] = row_sums[by0:top_end] > threshold_w
        frame_rows[bot_start : by1 + 1] = row_sums[bot_start : by1 + 1] > threshold_w

        if frame_rows.any():
            dilated = np.convolve(frame_rows.astype(int), kernel, mode="same") > 0
            # Hard-truncate: outside bbox and center zone
            dilated[:by0] = False
            dilated[top_end:bot_start] = False
            dilated[by1 + 1 :] = False
            result[dilated, :] = 0
            logger.info(
                "移除上下印框: %d rows (bbox_h=%d, scan=%d, thr=%.0f, erase_r=%d)",
                int(dilated.sum()), bbox_h, scan_rows, threshold_w, erase_r,
            )

        # ── Step 3: X-axis projection (left/right frames) ───
        col_sums = content.sum(axis=0).astype(float)

        frame_cols = np.zeros(w, dtype=bool)
        left_end = bx0 + scan_cols
        right_start = bx1 - scan_cols + 1

        frame_cols[bx0:left_end] = col_sums[bx0:left_end] > threshold_h
        frame_cols[right_start : bx1 + 1] = col_sums[right_start : bx1 + 1] > threshold_h

        if frame_cols.any():
            dilated = np.convolve(frame_cols.astype(int), kernel, mode="same") > 0
            dilated[:bx0] = False
            dilated[left_end:right_start] = False
            dilated[bx1 + 1 :] = False
            result[:, dilated] = 0
            logger.info(
                "移除左右印框: %d cols (bbox_w=%d, scan=%d, thr=%.0f, erase_r=%d)",
                int(dilated.sum()), bbox_w, scan_cols, threshold_h, erase_r,
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
