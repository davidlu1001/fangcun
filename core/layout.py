"""
Traditional Chinese seal character layout engine.

Reading order (right-to-left, top-to-bottom):
  1 char  → centered, 15% margin
  2 chars → vertical stack (top / bottom)
  3 chars → right column 2 + left column 1 (vertically centered)
  4 chars → 2×2 grid: 右上→右下→左上→左下
  >4 chars → warning + auto-shrink to fit
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

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

    MARGIN_RATIO = 0.15  # padding inside each grid cell

    def arrange(
        self,
        char_imgs: list[Image.Image],
        shape: str,
        canvas_size: tuple[int, int],
    ) -> list[dict]:
        """
        Args:
            char_imgs: list of mode-"L" character masks (already extracted)
            shape:     'oval' | 'square'
            canvas_size: (width, height) of the text area

        Returns:
            [{'img': Image, 'x': int, 'y': int, 'w': int, 'h': int}, ...]
            Coordinates are relative to the text-area origin.
        """
        n = len(char_imgs)
        if n == 0:
            return []

        tw, th = canvas_size
        grid = self._grid_map(n)

        if n > 4:
            logger.warning("超过4字 (%d字), 自动缩小适配", n)

        placements: list[dict] = []
        for i, (rx, ry, rw, rh) in enumerate(grid):
            cell_x = int(rx * tw)
            cell_y = int(ry * th)
            cell_w = int(rw * tw)
            cell_h = int(rh * th)

            mask = char_imgs[i]
            fitted = self._fit_to_cell(mask, cell_w, cell_h)

            # Center the fitted character in its grid cell
            px = cell_x + (cell_w - fitted.width) // 2
            py = cell_y + (cell_h - fitted.height) // 2

            placements.append(
                {
                    "img": fitted,
                    "x": px,
                    "y": py,
                    "w": fitted.width,
                    "h": fitted.height,
                }
            )

        return placements

    # ── grid definitions ─────────────────────────────────────

    @staticmethod
    def _grid_map(n: int) -> list[tuple[float, float, float, float]]:
        """
        Return grid cells as (rel_x, rel_y, rel_w, rel_h) for *n* characters.
        Traditional right-to-left reading order.
        """
        if n == 1:
            return [(0.0, 0.0, 1.0, 1.0)]

        if n == 2:
            # Vertical stack
            return [
                (0.0, 0.0, 1.0, 0.5),   # top
                (0.0, 0.5, 1.0, 0.5),   # bottom
            ]

        if n == 3:
            # Right column: char 0 (top), char 1 (bottom)
            # Left column:  char 2 (vertically centered → full height)
            return [
                (0.5, 0.0, 0.5, 0.5),   # right-top
                (0.5, 0.5, 0.5, 0.5),   # right-bottom
                (0.0, 0.0, 0.5, 1.0),   # left-center
            ]

        if n == 4:
            # 2×2 grid, reading: 右上→右下→左上→左下
            return [
                (0.5, 0.0, 0.5, 0.5),   # right-top
                (0.5, 0.5, 0.5, 0.5),   # right-bottom
                (0.0, 0.0, 0.5, 0.5),   # left-top
                (0.0, 0.5, 0.5, 0.5),   # left-bottom
            ]

        # >4 chars: calculate grid dynamically
        cols = 2
        rows = (n + cols - 1) // cols
        cells: list[tuple[float, float, float, float]] = []
        cw = 1.0 / cols
        ch = 1.0 / rows
        idx = 0
        # Right column first, then left (right-to-left)
        for col in range(cols - 1, -1, -1):
            for row in range(rows):
                if idx >= n:
                    break
                cells.append((col * cw, row * ch, cw, ch))
                idx += 1
        return cells

    # ── fitting ──────────────────────────────────────────────

    def _fit_to_cell(
        self, mask: Image.Image, cell_w: int, cell_h: int
    ) -> Image.Image:
        """Resize mask to fit within cell respecting margin and aspect ratio."""
        margin = self.MARGIN_RATIO
        max_w = max(1, int(cell_w * (1 - margin)))
        max_h = max(1, int(cell_h * (1 - margin)))

        src_w, src_h = mask.size
        if src_w == 0 or src_h == 0:
            return mask

        scale = min(max_w / src_w, max_h / src_h)
        new_w = max(1, int(src_w * scale))
        new_h = max(1, int(src_h * scale))

        return mask.resize((new_w, new_h), Image.Resampling.LANCZOS)
