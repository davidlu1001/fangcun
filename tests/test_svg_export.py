"""Unit tests for SVG vector export.

Verifies that SealRenderer.render_svg produces valid, self-contained SVG
XML suitable for print / vector editing. No texture is applied (by design).
"""

from __future__ import annotations

import re
from xml.etree import ElementTree as ET

import numpy as np
import pytest
from PIL import Image

from core.renderer import SealRenderer


def _make_dummy_layout() -> list[dict]:
    """Single centered mask for testing."""
    arr = np.zeros((100, 100), dtype=np.uint8)
    arr[20:80, 30:70] = 255
    mask = Image.fromarray(arr, "L")
    return [{"img": mask, "x": 200, "y": 200, "w": 100, "h": 100}]


@pytest.mark.unit
class TestSvgExport:
    """Verify SVG output structure and correctness."""

    def test_svg_is_valid_xml(self) -> None:
        renderer = SealRenderer()
        svg = renderer.render_svg(
            _make_dummy_layout(), shape="square", style="baiwen",
            color=(178, 34, 34), size=600,
        )
        # Should parse as valid XML
        root = ET.fromstring(svg)
        assert root.tag.endswith("svg"), f"Root is not <svg>, got {root.tag}"

    def test_svg_has_viewbox(self) -> None:
        renderer = SealRenderer()
        svg = renderer.render_svg(
            _make_dummy_layout(), shape="square", style="baiwen",
            color=(178, 34, 34), size=600,
        )
        assert 'viewBox="0 0 600 600"' in svg, (
            "Square seal should have viewBox='0 0 600 600'"
        )

    def test_svg_oval_uses_ellipse_frame(self) -> None:
        renderer = SealRenderer()
        svg = renderer.render_svg(
            _make_dummy_layout(), shape="oval", style="baiwen",
            color=(178, 34, 34), size=600,
        )
        assert "<ellipse" in svg, "Oval seal should contain <ellipse>"

    def test_svg_baiwen_has_red_background(self) -> None:
        renderer = SealRenderer()
        svg = renderer.render_svg(
            _make_dummy_layout(), shape="square", style="baiwen",
            color=(178, 34, 34), size=600,
        )
        # Baiwen: red-filled shape
        assert (
            "#B22222" in svg.upper()
            or "rgb(178,34,34)" in svg.replace(" ", "")
        )

    def test_svg_zhuwen_no_background_fill(self) -> None:
        """Zhuwen shouldn't fill the background with red — frame + chars only."""
        renderer = SealRenderer()
        zhuwen_svg = renderer.render_svg(
            _make_dummy_layout(), shape="square", style="zhuwen",
            color=(178, 34, 34), size=600,
        )
        baiwen_svg = renderer.render_svg(
            _make_dummy_layout(), shape="square", style="baiwen",
            color=(178, 34, 34), size=600,
        )
        # Smoke test: zhuwen and baiwen must produce visibly different SVG
        assert zhuwen_svg != baiwen_svg
        # More specific: zhuwen shouldn't have a full-canvas rect fill
        # Count any full-canvas rect with non-"none" fill
        full_canvas_fill = re.search(
            r'<rect[^>]*width="600"[^>]*fill="#[0-9A-Fa-f]{6}"', zhuwen_svg
        )
        assert full_canvas_fill is None, (
            "Zhuwen should not have a filled full-canvas background rect"
        )

    def test_svg_character_paths_present(self) -> None:
        renderer = SealRenderer()
        svg = renderer.render_svg(
            _make_dummy_layout(), shape="square", style="baiwen",
            color=(178, 34, 34), size=600,
        )
        # Character contour should produce at least one <path>
        assert "<path" in svg, "Character should produce at least one <path>"

    def test_svg_self_contained_no_external_refs(self) -> None:
        """No external file references, suitable for standalone use."""
        renderer = SealRenderer()
        svg = renderer.render_svg(
            _make_dummy_layout(), shape="oval", style="zhuwen",
            color=(178, 34, 34), size=600,
        )
        # No external href (data: URIs are fine but we don't emit any)
        assert "href" not in svg or 'xlink:href="data:' in svg

    def test_svg_oval_has_clip_path(self) -> None:
        """Oval output clips the entire canvas to an ellipse shape."""
        renderer = SealRenderer()
        svg = renderer.render_svg(
            _make_dummy_layout(), shape="oval", style="baiwen",
            color=(178, 34, 34), size=600,
        )
        assert "<clipPath" in svg, "Oval should define a clipPath"
        assert 'clip-path="url(#shapeClip)"' in svg, (
            "Oval should reference the clipPath"
        )

    def test_svg_character_path_uses_evenodd(self) -> None:
        """Character paths should use fill-rule=evenodd so holes render right."""
        renderer = SealRenderer()
        svg = renderer.render_svg(
            _make_dummy_layout(), shape="square", style="baiwen",
            color=(178, 34, 34), size=600,
        )
        assert 'fill-rule="evenodd"' in svg, (
            "Character paths must use fill-rule=evenodd for hole handling"
        )

    def test_svg_path_coordinates_respect_offset(self) -> None:
        """Path data should be offset to canvas-absolute coords."""
        renderer = SealRenderer()
        layout = _make_dummy_layout()  # offset x=200, y=200
        svg = renderer.render_svg(
            layout, shape="square", style="baiwen",
            color=(178, 34, 34), size=600,
        )
        # Extract the d= attribute from the first <path>
        match = re.search(r'<path d="([^"]+)"', svg)
        assert match is not None
        d_attr = match.group(1)
        # Parse numeric tokens from the path data
        nums = [int(n) for n in re.findall(r"\d+", d_attr)]
        assert nums, "Path should contain numeric coordinates"
        # All coords must be >= offset (200) — since the mask stroke starts
        # at local (30, 20) and the offset is (200, 200), we expect >= ~220
        assert min(nums) >= 200, (
            f"Path coords should be offset into canvas (>=200), "
            f"got min={min(nums)}"
        )
