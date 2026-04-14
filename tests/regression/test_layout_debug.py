"""Regression test for SealGenerator.render_layout_debug public API."""

from __future__ import annotations

import pytest
from PIL import Image


@pytest.mark.regression
class TestLayoutDebugRender:
    def test_render_layout_debug_produces_rgba_image(self, gen) -> None:
        img = gen.render_layout_debug(
            "禅", shape="oval", style="baiwen", seal_type="leisure"
        )
        assert isinstance(img, Image.Image)
        assert img.mode == "RGBA"
        # Should have non-zero size
        assert img.size[0] > 0 and img.size[1] > 0
