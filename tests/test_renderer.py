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


@pytest.mark.unit
class TestStyleExclusivity:
    """Verify baiwen and zhuwen rendering produce fundamentally different outputs."""

    def test_baiwen_has_opaque_background(self) -> None:
        """Baiwen: solid red background, white text cutouts."""
        renderer = SealRenderer()
        layout = _make_dummy_layout()
        result = renderer.render(
            layout, shape="square", style="baiwen", color=(178, 34, 34), size=600
        )
        arr = np.array(result)
        # Baiwen has mostly opaque pixels (red background)
        opaque_ratio = (arr[:, :, 3] > 200).sum() / arr[:, :, 3].size
        assert opaque_ratio > 0.8, (
            f"Baiwen should have opaque background, got {opaque_ratio:.2f}"
        )

    def test_zhuwen_has_transparent_background(self) -> None:
        """Zhuwen: transparent background, red text + frame."""
        renderer = SealRenderer()
        layout = _make_dummy_layout()
        result = renderer.render(
            layout, shape="square", style="zhuwen", color=(178, 34, 34), size=600
        )
        arr = np.array(result)
        # Zhuwen has mostly transparent pixels
        transparent_ratio = (arr[:, :, 3] < 10).sum() / arr[:, :, 3].size
        assert transparent_ratio > 0.5, (
            f"Zhuwen should have transparent bg, got {transparent_ratio:.2f}"
        )

    def test_baiwen_zhuwen_structurally_different(self) -> None:
        """The two styles must produce structurally different alpha channels."""
        renderer = SealRenderer()
        layout = _make_dummy_layout()
        bw = renderer.render(
            layout, shape="square", style="baiwen", color=(178, 34, 34), size=600
        )
        zw = renderer.render(
            layout, shape="square", style="zhuwen", color=(178, 34, 34), size=600
        )
        bw_alpha = np.array(bw)[:, :, 3]
        zw_alpha = np.array(zw)[:, :, 3]
        # They should be very different (one is mostly opaque, other mostly transparent)
        diff = np.abs(bw_alpha.astype(float) - zw_alpha.astype(float)).mean()
        assert diff > 100, (
            f"Baiwen and zhuwen should differ substantially, mean diff={diff:.1f}"
        )
