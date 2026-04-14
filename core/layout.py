"""
Traditional Chinese seal character layout engine.

Reading order (right-to-left, top-to-bottom):
  1 char  → centered
  2 chars → vertical stack (top / bottom)
  3 chars → right column 2 + left column 1 (vertically centered)
  4 chars → 2×2 grid: 右上→右下→左上→左下
  >4 chars → warning + auto-shrink to fit

Features:
  - Dynamic margin by (style, shape) via MARGIN_TABLE
  - Conditional vertical/horizontal stretch for tall/wide cells (max 1.25x)
  - Zhuwen margin bleeding: 4% bleed with text_scale=0.98 (冲刀破边)
  - Stroke width normalization via distance transform
  - Extreme flat chars (一二三): reverse-constructed from sibling stroke width
  - Visual centroid compensation (0.65 coefficient)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Placement:
    """One character's position in the text area."""

    img: Image.Image  # mode "L" mask
    x: int
    y: int
    w: int
    h: int

    def as_dict(self) -> dict:
        return {"img": self.img, "x": self.x, "y": self.y, "w": self.w, "h": self.h}


class SealLayout:
    """Arrange character masks inside a text-area rectangle."""

    # Dynamic margin by (style, shape). Smaller = more filled seal.
    MARGIN_TABLE: dict[tuple[str, str], float] = {
        ("baiwen", "square"): 0.06,
        ("baiwen", "oval"): 0.04,
        ("zhuwen", "square"): 0.03,
        ("zhuwen", "oval"): 0.01,
    }

    def arrange(
        self,
        char_imgs: list[Image.Image],
        shape: str,
        canvas_size: tuple[int, int],
        style: str = "baiwen",
    ) -> list[dict]:
        """Multi-phase layout: fit → detect extreme → reverse-construct → normalize.

        Phase 1: Normal fit_to_cell for all chars (equal-ratio + cell stretch)
        Phase 2: Detect extreme-aspect chars, compute sibling stroke width
        Phase 3: Reverse-construct extreme chars using sibling stroke width
        Phase 4: Stroke-width normalize remaining normal chars
        Phase 5: Build placements with centroid compensation
        """
        n = len(char_imgs)
        if n == 0:
            return []

        tw, th = canvas_size
        grid = self._grid_map(n)
        margin = self.MARGIN_TABLE.get((style, shape), 0.04)

        if n > 4:
            logger.warning("超过4字 (%d字), 自动缩小适配", n)

        # ── Phase 1: Normal fit ─────────────────────────────
        cells: list[tuple[int, int, int, int]] = []
        fitted_list: list[Image.Image] = []

        for i, (rx, ry, rw, rh) in enumerate(grid):
            cell_x = int(rx * tw)
            cell_y = int(ry * th)
            cell_w = int(rw * tw)
            cell_h = int(rh * th)
            cells.append((cell_x, cell_y, cell_w, cell_h))

            mask = char_imgs[i]
            logger.debug(
                "[R9-P1] char[%d] src=%dx%d cell=(%d,%d,%dx%d)",
                i, mask.size[0], mask.size[1], cell_x, cell_y, cell_w, cell_h,
            )
            if style == "zhuwen":
                bleed = int(min(cell_w, cell_h) * 0.04)
                fitted = self._fit_to_cell(mask, cell_w + bleed, cell_h + bleed, margin=0.0)
            else:
                fitted = self._fit_to_cell(mask, cell_w, cell_h, margin)
            logger.debug(
                "[R9-P1] char[%d] fitted=%dx%d", i, fitted.width, fitted.height,
            )
            fitted_list.append(fitted)

        # ── Phase 1.5: 2-char fill balance ──────────────────
        if n == 2:
            fitted_list = self._balance_two_chars(
                fitted_list, grid, tw, th
            )

        # ── Phase 2: Detect extreme chars + sibling stroke width ──
        extreme_indices: list[tuple[int, str]] = []
        normal_widths: list[float] = []

        for i, mask in enumerate(char_imgs):
            src_w, src_h = mask.size
            aspect = src_w / max(src_h, 1)
            if aspect > 2.5:
                extreme_indices.append((i, "horizontal"))
            elif aspect < 0.4:
                extreme_indices.append((i, "vertical"))
            else:
                sw = self._estimate_stroke_width(np.array(fitted_list[i]))
                if sw > 0:
                    normal_widths.append(sw)

        if normal_widths:
            sibling_sw = float(np.median(normal_widths))
        else:
            # All chars are extreme — use canvas-relative fallback
            sibling_sw = min(tw, th) * 0.025

        # ── Phase 3: Reverse-construct extreme chars ────────
        extreme_set = {idx for idx, _ in extreme_indices}
        for idx, orientation in extreme_indices:
            cell_x, cell_y, cell_w, cell_h = cells[idx]
            fitted_list[idx] = self._fit_extreme_flat(
                char_imgs[idx], cell_w, cell_h,
                target_stroke_width=sibling_sw,
                orientation=orientation,
            )

        # ── Phase 4: Stroke-width normalize normal chars ────
        if n > 1 and sibling_sw > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            for i in range(n):
                if i in extreme_set:
                    continue
                sw = self._estimate_stroke_width(np.array(fitted_list[i]))
                if sw <= 0:
                    continue
                ratio = sw / sibling_sw
                if ratio < 0.75:
                    arr = np.array(fitted_list[i])
                    arr = cv2.dilate(arr, k, iterations=1)
                    fitted_list[i] = Image.fromarray(arr, "L")
                elif ratio > 1.35:
                    arr = np.array(fitted_list[i])
                    arr = cv2.erode(arr, k, iterations=1)
                    fitted_list[i] = Image.fromarray(arr, "L")

        # ── Phase 5: Build placements with centroid offset ──
        placements: list[dict] = []
        for i in range(n):
            cell_x, cell_y, cell_w, cell_h = cells[i]
            fitted = fitted_list[i]

            dx, dy = self._centroid_offset(fitted)
            px = cell_x + (cell_w - fitted.width) // 2 - int(dx * 0.65)
            py = cell_y + (cell_h - fitted.height) // 2 - int(dy * 0.65)

            overflow = (px < cell_x or py < cell_y
                        or px + fitted.width > cell_x + cell_w
                        or py + fitted.height > cell_y + cell_h)
            logger.debug(
                "[R9-P1] char[%d] placement=(%d,%d) final=%dx%d "
                "cell_bounds=(%d,%d,%d,%d) centroid_dx=%d dy=%d %s",
                i, px, py, fitted.width, fitted.height,
                cell_x, cell_y, cell_x + cell_w, cell_y + cell_h,
                dx, dy, "OVERFLOW!" if overflow else "ok",
            )

            placements.append({
                "img": fitted,
                "x": px,
                "y": py,
                "w": fitted.width,
                "h": fitted.height,
            })

        return placements

    # ── grid definitions ─────────────────────────────────────

    @staticmethod
    def _grid_map(n: int) -> list[tuple[float, float, float, float]]:
        """Return grid cells as (rel_x, rel_y, rel_w, rel_h) for n characters."""
        if n == 1:
            return [(0.0, 0.0, 1.0, 1.0)]
        if n == 2:
            return [
                (0.0, 0.0, 1.0, 0.5),
                (0.0, 0.5, 1.0, 0.5),
            ]
        if n == 3:
            return [
                (0.5, 0.0, 0.5, 0.5),
                (0.5, 0.5, 0.5, 0.5),
                (0.0, 0.0, 0.5, 1.0),
            ]
        if n == 4:
            return [
                (0.5, 0.0, 0.5, 0.5),
                (0.5, 0.5, 0.5, 0.5),
                (0.0, 0.0, 0.5, 0.5),
                (0.0, 0.5, 0.5, 0.5),
            ]
        cols = 2
        rows = (n + cols - 1) // cols
        cells: list[tuple[float, float, float, float]] = []
        cw = 1.0 / cols
        ch = 1.0 / rows
        idx = 0
        for col in range(cols - 1, -1, -1):
            for row in range(rows):
                if idx >= n:
                    break
                cells.append((col * cw, row * ch, cw, ch))
                idx += 1
        return cells

    # ── fitting ──────────────────────────────────────────────

    @staticmethod
    def _fit_to_cell(
        mask: Image.Image, cell_w: int, cell_h: int, margin: float = 0.04
    ) -> Image.Image:
        """Resize mask to fit within cell. Conditional stretch for tall/wide cells."""
        max_w = max(1, int(cell_w * (1 - margin)))
        max_h = max(1, int(cell_h * (1 - margin)))

        src_w, src_h = mask.size
        if src_w == 0 or src_h == 0:
            return mask

        scale = min(max_w / src_w, max_h / src_h)
        new_w = max(1, int(src_w * scale))
        new_h = max(1, int(src_h * scale))

        # Tall cells: up to 1.25x vertical stretch
        cell_ratio = max_h / max(max_w, 1)
        if cell_ratio > 1.5 and new_h / max_h < 0.85:
            target_h = int(max_h * 0.90)
            stretch = min(1.25, target_h / max(new_h, 1))
            new_h = int(new_h * stretch)
        elif cell_ratio < 0.67 and new_w / max_w < 0.85:
            target_w = int(max_w * 0.90)
            stretch = min(1.25, target_w / max(new_w, 1))
            new_w = int(new_w * stretch)

        return mask.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # ── extreme flat character handling ─────────────────────

    @staticmethod
    def _balance_two_chars(
        fitted_list: list[Image.Image],
        grid: list[tuple[float, float, float, float]],
        tw: int,
        th: int,
    ) -> list[Image.Image]:
        """Equalize fill ratio for 2-char seals.

        When one char fills significantly less of its cell, scale it up
        to match the other (max 1.25x, won't exceed cell width).
        """
        fill_ratios = []
        for i, fitted in enumerate(fitted_list):
            cell_h_px = int(grid[i][3] * th)
            fill_ratios.append(fitted.height / cell_h_px if cell_h_px > 0 else 0)

        if not all(r > 0 for r in fill_ratios):
            return fitted_list

        target = max(fill_ratios) * 0.95
        result = list(fitted_list)
        for i, fitted in enumerate(fitted_list):
            if fill_ratios[i] < target * 0.90:
                cell_w_px = int(grid[i][2] * tw)
                scale = min(
                    target / fill_ratios[i],
                    (cell_w_px * 0.90) / max(fitted.width, 1),
                    1.25,
                )
                if scale > 1.05:
                    new_w = int(fitted.width * scale)
                    new_h = int(fitted.height * scale)
                    result[i] = fitted.resize(
                        (new_w, new_h), Image.Resampling.LANCZOS
                    )

        return result

    @staticmethod
    def _estimate_stroke_width(mask_arr: np.ndarray) -> float:
        """Estimate stroke width in pixels via distance transform (p70 × 2)."""
        binary = (mask_arr > 128).astype(np.uint8)
        if not np.any(binary):
            return 0.0
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        vals = dist[binary > 0]
        if len(vals) == 0:
            return 0.0
        return float(np.percentile(vals, 70)) * 2.0

    @staticmethod
    def _fit_extreme_flat(
        mask: Image.Image,
        cell_w: int,
        cell_h: int,
        target_stroke_width: float,
        orientation: str = "horizontal",
    ) -> Image.Image:
        """Reverse-construct layout for extreme-aspect chars (一二三).

        The source mask for chars like 一 is a solid rectangle (the
        extractor crops to bbox). Simply resizing preserves the solid block.

        Instead: resize the stroke to target dimensions, then center it
        in a larger canvas to guarantee breathing room above/below.
        This ensures ink_ratio stays well under 55%.
        """
        src_arr = np.array(mask)
        if not np.any(src_arr > 128):
            return mask

        src_w, src_h = mask.size

        if orientation == "horizontal":
            target_length = int(cell_w * 0.70)
            stroke_h = max(3, int(target_stroke_width))
            # Canvas 3.5x stroke height for clear breathing room
            canvas_h = max(int(target_stroke_width * 3.5), int(cell_h * 0.20))
            canvas_h = min(canvas_h, int(cell_h * 0.50))
            canvas_w = min(target_length, int(cell_w * 0.90))

            # Resize source to (length × stroke_height)
            stroke_img = mask.resize(
                (max(1, canvas_w), max(1, stroke_h)),
                Image.Resampling.LANCZOS,
            )
            # Center stroke in canvas with background padding
            canvas = Image.new("L", (max(1, canvas_w), max(1, canvas_h)), 0)
            y_off = (canvas_h - stroke_h) // 2
            canvas.paste(stroke_img, (0, max(0, y_off)))
            return canvas
        else:
            target_length = int(cell_h * 0.70)
            stroke_w = max(3, int(target_stroke_width))
            canvas_w = max(int(target_stroke_width * 3.5), int(cell_w * 0.20))
            canvas_w = min(canvas_w, int(cell_w * 0.50))
            canvas_h = min(target_length, int(cell_h * 0.90))

            stroke_img = mask.resize(
                (max(1, stroke_w), max(1, canvas_h)),
                Image.Resampling.LANCZOS,
            )
            canvas = Image.new("L", (max(1, canvas_w), max(1, canvas_h)), 0)
            x_off = (canvas_w - stroke_w) // 2
            canvas.paste(stroke_img, (max(0, x_off), 0))
            return canvas

    # ── visual centroid ─────────────────────────────────────

    @staticmethod
    def _centroid_offset(mask: Image.Image) -> tuple[int, int]:
        """Pixel-weighted centroid offset from bbox geometric center."""
        arr = np.array(mask)
        ys, xs = np.where(arr > 128)
        if len(xs) == 0:
            return 0, 0
        cx_px = float(np.mean(xs))
        cy_px = float(np.mean(ys))
        cx_geom = (arr.shape[1] - 1) / 2.0
        cy_geom = (arr.shape[0] - 1) / 2.0
        return int(round(cx_px - cx_geom)), int(round(cy_px - cy_geom))

    @staticmethod
    def debug_render(
        placements: list[dict],
        canvas_size: tuple[int, int],
        ta_offset: tuple[int, int] = (0, 0),
    ) -> Image.Image:
        """Render debug overlay showing grid cells, ink bounds, and centroids.

        Returns RGBA image: blue rectangles = cell boundaries, red rectangles =
        ink bounding boxes, green dots = pixel-weighted centroids.
        """
        from PIL import ImageDraw
        tw, th = canvas_size
        debug = Image.new("RGBA", (tw, th), (255, 255, 255, 128))
        draw = ImageDraw.Draw(debug)

        for item in placements:
            x = item["x"] - ta_offset[0]
            y = item["y"] - ta_offset[1]
            w, h = item["w"], item["h"]
            # Cell boundary (blue)
            draw.rectangle([x, y, x + w, y + h], outline=(0, 0, 255, 200), width=2)
            # Ink bounding box (red)
            mask_arr = np.array(item["img"])
            ys, xs = np.where(mask_arr > 128)
            if len(xs) > 0:
                ink_x0 = int(xs.min()) + x
                ink_y0 = int(ys.min()) + y
                ink_x1 = int(xs.max()) + x
                ink_y1 = int(ys.max()) + y
                draw.rectangle(
                    [ink_x0, ink_y0, ink_x1, ink_y1],
                    outline=(255, 0, 0, 200), width=1,
                )
                # Centroid (green dot)
                cx = int(np.mean(xs)) + x
                cy = int(np.mean(ys)) + y
                draw.ellipse(
                    [cx - 3, cy - 3, cx + 3, cy + 3], fill=(0, 255, 0, 255)
                )

        return debug
