"""Unit tests for the `_relative_stroke_width` helper used by R12.

These are pure unit tests (no network, no SealGenerator). They live at the
top of tests/ so the regression `gen` fixture is not pulled in.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from core.scraper import CalligraphyScraper


@pytest.mark.unit
class TestRelativeStrokeWidth:
    """Unit tests for the _relative_stroke_width helper."""

    def test_stroke_width_measurable(self) -> None:
        """_relative_stroke_width should return non-zero for real character masks."""
        # Create a synthetic "character" image (dark strokes on light bg)
        img = Image.new("L", (200, 200), 255)
        arr = np.array(img)
        arr[80:120, 50:150] = 0  # horizontal stroke (40 px thick x 100 wide)
        arr[50:150, 90:110] = 0  # vertical stroke (20 px wide x 100 tall)
        img = Image.fromarray(arr, "L")

        sw = CalligraphyScraper._relative_stroke_width(img)
        assert sw > 0, f"Stroke width should be positive, got {sw}"
        assert sw < 1.0, f"Relative stroke width should be < 1.0, got {sw}"

    def test_stroke_width_blank_image_zero(self) -> None:
        """All-white image has no strokes — expect 0.0."""
        img = Image.new("L", (100, 100), 255)
        assert CalligraphyScraper._relative_stroke_width(img) == 0.0

    def test_stroke_width_thick_greater_than_thin(self) -> None:
        """Thicker stroke should yield a higher relative stroke width than a thin one."""
        size = 200

        thin_arr = np.full((size, size), 255, dtype=np.uint8)
        thin_arr[95:105, 20:180] = 0  # 10 px thick
        thin_img = Image.fromarray(thin_arr, "L")

        thick_arr = np.full((size, size), 255, dtype=np.uint8)
        thick_arr[70:130, 20:180] = 0  # 60 px thick
        thick_img = Image.fromarray(thick_arr, "L")

        thin_sw = CalligraphyScraper._relative_stroke_width(thin_img)
        thick_sw = CalligraphyScraper._relative_stroke_width(thick_img)

        assert thick_sw > thin_sw, (
            f"Thick stroke ({thick_sw}) should exceed thin stroke ({thin_sw})"
        )
