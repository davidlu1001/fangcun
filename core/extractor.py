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

        # (印框剥离已在 _extract_yinpu_strokes 阶段2 完成，无需额外步骤)

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
        印谱字形提取：两阶段终极算法。

        阶段1 — 聚合 Alpha CCA：
          找到最大不透明连通块（石面主体），聚合周围大型碎块，
          得到印章真实石面的联合 Bounding Box。
          - 面积 > 最大块 5%（过滤噪点）
          - 中心距 < 主块短边 2 倍（过滤远处页面边框）

        阶段2 — CCA bbox 内边缘内缩：
          从联合 bbox 各边缘向内逐行扫描 alpha，
          找到首个"不透明占比 >= 50%"的行 = 石面开始，
          该行之前的透明通道 = 印框刻槽，安全剥离。
          替代了所有版本的 _remove_seal_frame。
        """
        if img.mode not in ("RGBA", "LA"):
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img)
            return 255 - np.array(bg.convert("L"))

        alpha = np.array(img.split()[-1])
        opaque = (alpha >= 128).astype(np.uint8)

        # ── 阶段1: 聚合 Alpha CCA ───────────────────────────
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            opaque, connectivity=8
        )

        if num_labels <= 1:
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[-1])
            return np.array(bg.convert("L"))

        largest_label = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
        max_area = int(stats[largest_label, cv2.CC_STAT_AREA])

        largest_cx = stats[largest_label, cv2.CC_STAT_LEFT] + stats[largest_label, cv2.CC_STAT_WIDTH] / 2.0
        largest_cy = stats[largest_label, cv2.CC_STAT_TOP] + stats[largest_label, cv2.CC_STAT_HEIGHT] / 2.0
        ref_dim = min(
            int(stats[largest_label, cv2.CC_STAT_WIDTH]),
            int(stats[largest_label, cv2.CC_STAT_HEIGHT]),
        )

        min_x, min_y = int(alpha.shape[1]), int(alpha.shape[0])
        max_x, max_y = 0, 0
        valid_chunks = 0

        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area <= max_area * 0.05:
                continue

            cx = stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] / 2.0
            cy = stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] / 2.0
            if abs(cx - largest_cx) >= ref_dim * 2.0 or abs(cy - largest_cy) >= ref_dim * 2.0:
                continue

            min_x = min(min_x, int(stats[i, cv2.CC_STAT_LEFT]))
            min_y = min(min_y, int(stats[i, cv2.CC_STAT_TOP]))
            max_x = max(max_x, int(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] - 1))
            max_y = max(max_y, int(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] - 1))
            valid_chunks += 1

        if valid_chunks == 0:
            min_x = int(stats[largest_label, cv2.CC_STAT_LEFT])
            min_y = int(stats[largest_label, cv2.CC_STAT_TOP])
            max_x = min_x + int(stats[largest_label, cv2.CC_STAT_WIDTH]) - 1
            max_y = min_y + int(stats[largest_label, cv2.CC_STAT_HEIGHT]) - 1
            valid_chunks = 1

        by0, bx0, by1, bx1 = min_y, min_x, max_y, max_x
        bh = by1 - by0 + 1
        bw = bx1 - bx0 + 1

        logger.info(
            "Alpha CCA 阶段1: 聚合 %d 石面碎块, bbox=%dx%d @(%d,%d)",
            valid_chunks, bw, bh, bx0, by0,
        )

        # ── 阶段2: CCA bbox 内边缘扫描，剥离印框 ────────────
        bbox_alpha = alpha[by0 : by1 + 1, bx0 : bx1 + 1]
        max_scan = max(5, int(min(bh, bw) * 0.15))

        def scan_frame_thickness(strips: np.ndarray) -> int:
            for t in range(min(max_scan, len(strips))):
                if (strips[t] >= 128).mean() >= 0.50:
                    return max(t, 1)
            return max_scan

        mt = scan_frame_thickness(bbox_alpha)
        mb = scan_frame_thickness(bbox_alpha[::-1])
        ml = scan_frame_thickness(bbox_alpha.T)
        mr = scan_frame_thickness(bbox_alpha[:, ::-1].T)

        iy0 = by0 + mt + 1
        iy1 = by1 - mb - 1
        ix0 = bx0 + ml + 1
        ix1 = bx1 - mr - 1

        if iy0 >= iy1 or ix0 >= ix1:
            logger.warning(
                "内缩过度 (mt=%d mb=%d ml=%d mr=%d), 回退到 CCA bbox",
                mt, mb, ml, mr,
            )
            iy0, iy1, ix0, ix1 = by0, by1, bx0, bx1

        logger.info(
            "Alpha CCA 阶段2: 印框剥离 top=%d bot=%d left=%d right=%d → 提取区 %dx%d",
            mt, mb, ml, mr, ix1 - ix0 + 1, iy1 - iy0 + 1,
        )

        # ── 提取字形刻槽 → 白底黑字 ─────────────────────────
        gray = np.full(alpha.shape, 255, dtype=np.uint8)
        inner_alpha = alpha[iy0 : iy1 + 1, ix0 : ix1 + 1]
        gray[iy0 : iy1 + 1, ix0 : ix1 + 1] = np.where(
            inner_alpha < 128, 0, 255
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
