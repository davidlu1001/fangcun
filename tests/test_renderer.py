"""Unit tests verifying baiwen and zhuwen render paths are mutually exclusive.

Guards against "yin/yang mixing" regressions where the style switch at
core.renderer.SealRenderer.render() could accidentally produce a hybrid
(e.g. opaque background in zhuwen, or transparent background in baiwen).
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from core.renderer import SealRenderer


def _make_dummy_layout(w: int = 600, h: int = 600) -> list[dict]:
    """Create a minimal layout with one character mask."""
    arr = np.zeros((100, 100), dtype=np.uint8)
    arr[20:80, 20:80] = 255
    mask = Image.fromarray(arr, "L")
    return [{"img": mask, "x": 200, "y": 200, "w": 100, "h": 100}]


# Shape-aware thresholds. Oval clips ~21.5% of canvas to fully-transparent
# corners (π/4 inscribed ellipse), so opaque-baiwen tops out around 0.78 and
# the same threshold can't apply to both shapes.
_BAIWEN_OPAQUE_FLOOR = {"square": 0.8, "oval": 0.65}
_ZHUWEN_TRANSPARENT_FLOOR = {"square": 0.5, "oval": 0.5}


@pytest.mark.unit
class TestStyleExclusivity:
    """Verify baiwen and zhuwen rendering produce fundamentally different outputs."""

    @pytest.mark.parametrize("shape", ["square", "oval"])
    def test_baiwen_has_opaque_background(self, shape: str) -> None:
        """Baiwen: solid red background, white text cutouts."""
        renderer = SealRenderer()
        layout = _make_dummy_layout()
        result = renderer.render(
            layout, shape=shape, style="baiwen", color=(178, 34, 34), size=600
        )
        arr = np.array(result)
        opaque_ratio = (arr[:, :, 3] > 200).sum() / arr[:, :, 3].size
        floor = _BAIWEN_OPAQUE_FLOOR[shape]
        assert opaque_ratio > floor, (
            f"Baiwen {shape} should have opaque background "
            f"(>{floor}), got {opaque_ratio:.2f}"
        )

    @pytest.mark.parametrize("shape", ["square", "oval"])
    def test_zhuwen_has_transparent_background(self, shape: str) -> None:
        """Zhuwen: transparent background, red text + frame."""
        renderer = SealRenderer()
        layout = _make_dummy_layout()
        result = renderer.render(
            layout, shape=shape, style="zhuwen", color=(178, 34, 34), size=600
        )
        arr = np.array(result)
        transparent_ratio = (arr[:, :, 3] < 10).sum() / arr[:, :, 3].size
        floor = _ZHUWEN_TRANSPARENT_FLOOR[shape]
        assert transparent_ratio > floor, (
            f"Zhuwen {shape} should have transparent bg "
            f"(>{floor}), got {transparent_ratio:.2f}"
        )

    @pytest.mark.parametrize("shape", ["square", "oval"])
    def test_baiwen_zhuwen_structurally_different(self, shape: str) -> None:
        """The two styles must produce structurally different alpha channels."""
        renderer = SealRenderer()
        layout = _make_dummy_layout()
        bw = renderer.render(
            layout, shape=shape, style="baiwen", color=(178, 34, 34), size=600
        )
        zw = renderer.render(
            layout, shape=shape, style="zhuwen", color=(178, 34, 34), size=600
        )
        bw_alpha = np.array(bw)[:, :, 3]
        zw_alpha = np.array(zw)[:, :, 3]
        diff = np.abs(bw_alpha.astype(float) - zw_alpha.astype(float)).mean()
        assert diff > 100, (
            f"Baiwen vs zhuwen ({shape}) should differ substantially, "
            f"mean diff={diff:.1f}"
        )
