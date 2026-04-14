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

    def test_generate_return_debug_includes_overlay(self, gen) -> None:
        """generate(return_debug=True) returns both seal and overlay in one pass."""
        result = gen.generate(
            "禅", shape="oval", style="baiwen", seal_type="leisure",
            return_debug=True,
        )
        assert "image_transparent" in result
        assert "image_preview" in result
        assert "image_layout_debug" in result
        overlay = result["image_layout_debug"]
        assert isinstance(overlay, Image.Image)
        assert overlay.mode == "RGBA"

    def test_generate_without_return_debug_omits_overlay(self, gen) -> None:
        """generate() default: no layout overlay in result."""
        result = gen.generate(
            "禅", shape="oval", style="baiwen", seal_type="leisure"
        )
        assert "image_layout_debug" not in result
