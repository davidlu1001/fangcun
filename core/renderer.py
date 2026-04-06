"""
Seal renderer: composites laid-out characters into a complete seal image.

Supports two styles:
  白文 (baiwen/intaglio): red background, white text cutout
  朱文 (zhuwen/relief):   transparent background, red text + red frame

Supports two shapes:
  方章 (square):  size × size
  竖椭圆 (oval): size × (size × 1.35), with double-line inner frame

Oval frame dimensions (short side S):
  outer line width = S × 0.012
  inner line width = S × 0.006
  gap between      = S × 0.025
  text area        = inner frame interior × 0.78
"""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)


OVAL_RATIO = 1.35  # height / width
DEFAULT_COLOR = (178, 34, 34)  # 朱砂红


class SealRenderer:
    """Render a complete seal image from pre-arranged character layouts."""

    def render(
        self,
        layout: list[dict],
        shape: str = "oval",
        style: str = "baiwen",
        color: tuple[int, int, int] = DEFAULT_COLOR,
        size: int = 600,
    ) -> Image.Image:
        """
        Args:
            layout: [{'img': Image(L), 'x': int, 'y': int, 'w': int, 'h': int}, ...]
                    Coordinates are absolute (already offset into canvas).
            shape:  'oval' | 'square'
            style:  'baiwen' | 'zhuwen'
            color:  RGB tuple
            size:   short-side pixels

        Returns:
            RGBA seal image.
        """
        if shape == "oval":
            w = size
            h = int(size * OVAL_RATIO)
        else:
            w = h = size

        if style == "baiwen":
            return self._render_baiwen(layout, shape, color, w, h)
        return self._render_zhuwen(layout, shape, color, w, h)

    # ── baiwen (白文 / 阴文) ─────────────────────────────────

    def _render_baiwen(
        self,
        layout: list[dict],
        shape: str,
        color: tuple[int, int, int],
        w: int,
        h: int,
    ) -> Image.Image:
        """Red background with white text cutout."""
        canvas = Image.new("RGBA", (w, h), (*color, 255))
        draw = ImageDraw.Draw(canvas)

        # Inner white frame (oval only — double line)
        if shape == "oval":
            s = min(w, h)
            outer_lw = max(1, round(s * 0.012))
            inner_lw = max(1, round(s * 0.006))
            gap = round(s * 0.025)

            # Outer thin white line
            m1 = outer_lw // 2 + 2
            draw.ellipse(
                [m1, m1, w - m1 - 1, h - m1 - 1],
                outline=(255, 255, 255, 255),
                width=outer_lw,
            )
            # Inner thin white line
            m2 = m1 + outer_lw + gap
            draw.ellipse(
                [m2, m2, w - m2 - 1, h - m2 - 1],
                outline=(255, 255, 255, 255),
                width=inner_lw,
            )

        # Paste characters as white cutouts
        paper_color = (245, 242, 238, 255)  # warm paper white
        for idx, item in enumerate(layout):
            mask = item["img"]  # mode "L", 255 = stroke
            x, y = item["x"], item["y"]
            target_w, target_h = item["w"], item["h"]

            outside = x < 0 or y < 0 or x + target_w > w or y + target_h > h
            logger.debug(
                "[R9-P1] baiwen paste[%d]: pos=(%d,%d) size=%dx%d canvas=%dx%d %s",
                idx, x, y, target_w, target_h, w, h,
                "OUTSIDE!" if outside else "ok",
            )

            resized_mask = mask.resize(
                (target_w, target_h), Image.Resampling.LANCZOS
            )
            white_layer = Image.new("RGBA", (target_w, target_h), paper_color)
            canvas.paste(white_layer, (x, y), mask=resized_mask)

        # Apply shape mask (clip corners for oval)
        canvas = self._apply_shape_mask(canvas, shape, w, h)

        return canvas

    # ── zhuwen (朱文 / 阳文) ─────────────────────────────────

    def _render_zhuwen(
        self,
        layout: list[dict],
        shape: str,
        color: tuple[int, int, int],
        w: int,
        h: int,
    ) -> Image.Image:
        """Transparent background with red frame and red text.

        Chars and frame are rendered to separate alpha layers then merged
        via max-alpha so bleeding chars and frame lines fuse naturally
        (冲刀破边 effect).
        """
        rgba_color = (*color, 255)

        # ── Frame alpha layer ───────────────────────────────
        frame_layer = Image.new("L", (w, h), 0)
        frame_draw = ImageDraw.Draw(frame_layer)

        if shape == "oval":
            s = min(w, h)
            outer_lw = max(1, round(s * 0.012))
            inner_lw = max(1, round(s * 0.006))
            gap = round(s * 0.025)

            m1 = outer_lw // 2 + 2
            frame_draw.ellipse(
                [m1, m1, w - m1 - 1, h - m1 - 1],
                fill=None, outline=255,
                width=outer_lw,
            )
            m2 = m1 + outer_lw + gap
            frame_draw.ellipse(
                [m2, m2, w - m2 - 1, h - m2 - 1],
                fill=None, outline=255,
                width=inner_lw,
            )
        else:
            border_w = max(2, round(w * 0.018))
            p = border_w // 2 + 1
            frame_draw.rectangle(
                [p, p, w - p - 1, h - p - 1],
                fill=None, outline=255,
                width=border_w,
            )

        frame_alpha = np.array(frame_layer)

        # ── Character alpha layer ───────────────────────────
        char_alpha = np.zeros((h, w), dtype=np.uint8)
        for idx, item in enumerate(layout):
            mask = item["img"]
            x, y = item["x"], item["y"]
            target_w, target_h = item["w"], item["h"]

            outside = x < 0 or y < 0 or x + target_w > w or y + target_h > h
            logger.debug(
                "[R9-P1] zhuwen paste[%d]: pos=(%d,%d) size=%dx%d canvas=%dx%d %s",
                idx, x, y, target_w, target_h, w, h,
                "OUTSIDE!" if outside else "ok",
            )

            resized = np.array(
                mask.resize((target_w, target_h), Image.Resampling.LANCZOS)
            )

            # Clip to canvas bounds
            src_y0 = max(0, -y)
            src_x0 = max(0, -x)
            dst_y0 = max(0, y)
            dst_x0 = max(0, x)
            src_y1 = min(target_h, h - y)
            src_x1 = min(target_w, w - x)
            if src_y1 <= src_y0 or src_x1 <= src_x0:
                continue

            region = resized[src_y0:src_y1, src_x0:src_x1]
            char_alpha[dst_y0 : dst_y0 + region.shape[0],
                       dst_x0 : dst_x0 + region.shape[1]] = np.maximum(
                char_alpha[dst_y0 : dst_y0 + region.shape[0],
                           dst_x0 : dst_x0 + region.shape[1]],
                region,
            )

        # ── Merge: max alpha (chars + frame fuse at bleed points) ──
        merged_alpha = np.maximum(frame_alpha, char_alpha)

        canvas_arr = np.zeros((h, w, 4), dtype=np.uint8)
        canvas_arr[:, :, 0] = color[0]
        canvas_arr[:, :, 1] = color[1]
        canvas_arr[:, :, 2] = color[2]
        canvas_arr[:, :, 3] = merged_alpha

        canvas = Image.fromarray(canvas_arr, "RGBA")

        # Apply shape mask
        canvas = self._apply_shape_mask(canvas, shape, w, h)

        return canvas

    # ── shape mask ───────────────────────────────────────────

    @staticmethod
    def _apply_shape_mask(
        img: Image.Image, shape: str, w: int, h: int
    ) -> Image.Image:
        """Clip image to shape boundary (oval or square)."""
        if shape != "oval":
            return img

        mask = Image.new("L", (w, h), 0)
        ImageDraw.Draw(mask).ellipse([0, 0, w - 1, h - 1], fill=255)

        # Combine with existing alpha
        alpha = np.array(img.split()[3])
        shape_arr = np.array(mask)
        combined = np.minimum(alpha, shape_arr)

        result = img.copy()
        result.putalpha(Image.fromarray(combined))
        return result

    # ── utility ──────────────────────────────────────────────

    @staticmethod
    def canvas_dimensions(shape: str, size: int) -> tuple[int, int]:
        """Return (width, height) for the given shape and short-side size."""
        if shape == "oval":
            return size, int(size * OVAL_RATIO)
        return size, size

    @staticmethod
    def text_area(
        shape: str, size: int, style: str = "baiwen", char_count: int = 2
    ) -> tuple[int, int, int, int]:
        """
        Return (x, y, w, h) of the text area inside the frame.
        Coordinates are relative to the canvas origin.

        text_scale varies by style and char_count:
          - zhuwen:          0.98 (near frame for bleed)
          - baiwen, 1 char:  0.93 (顶天立地 fill)
          - baiwen, 2+ chars: 0.86 (leave red border)
        """
        w, h = SealRenderer.canvas_dimensions(shape, size)
        s = min(w, h)

        if shape == "oval":
            outer_lw = max(1, round(s * 0.012))
            inner_lw = max(1, round(s * 0.006))
            gap = round(s * 0.025)
            frame_total = outer_lw + gap + inner_lw + 4
        else:
            frame_total = max(2, round(w * 0.018)) + 2

        if style == "zhuwen" and shape == "oval":
            # R11: Oval's 4 corners fall outside the ellipse curve.
            # text_scale=0.98 pushes cells into those corners, causing
            # shape-mask clipping (e.g. 大观園 観字 right half cut off).
            # Use 0.88 for oval to keep text area inside the ellipse.
            # Square zhuwen keeps 0.98 (no corner clipping issue).
            text_scale = 0.88
        elif style == "zhuwen":
            text_scale = 0.98
        elif char_count == 1:
            text_scale = 0.93
        else:
            text_scale = 0.86
        area_w = int((w - 2 * frame_total) * text_scale)
        area_h = int((h - 2 * frame_total) * text_scale)
        area_x = (w - area_w) // 2
        area_y = (h - area_h) // 2

        return area_x, area_y, area_w, area_h
