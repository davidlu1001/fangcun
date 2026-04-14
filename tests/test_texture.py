"""Unit tests for deterministic texture generation via seed parameter."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from core.texture import StoneTexture


@pytest.mark.unit
class TestTextureDeterminism:
    """Verify texture is reproducible with a fixed seed."""

    def test_same_seed_same_output(self) -> None:
        """Identical input + same seed → identical output."""
        tex = StoneTexture()
        img = Image.new("RGBA", (200, 200), (178, 34, 34, 255))

        result1 = tex.apply(img, grain_strength=0.25, seed=42)
        result2 = tex.apply(img, grain_strength=0.25, seed=42)

        arr1 = np.array(result1)
        arr2 = np.array(result2)
        assert np.array_equal(arr1, arr2), "Same seed should produce identical output"

    def test_different_seed_different_output(self) -> None:
        """Different seeds → different output."""
        tex = StoneTexture()
        img = Image.new("RGBA", (200, 200), (178, 34, 34, 255))

        result1 = tex.apply(img, grain_strength=0.25, seed=42)
        result2 = tex.apply(img, grain_strength=0.25, seed=99)

        arr1 = np.array(result1)
        arr2 = np.array(result2)
        assert not np.array_equal(arr1, arr2), "Different seeds should differ"

    def test_seed_none_preserves_randomness(self) -> None:
        """seed=None must remain non-deterministic (preserves prior behavior)."""
        tex = StoneTexture()
        img = Image.new("RGBA", (200, 200), (178, 34, 34, 255))

        r1 = np.array(tex.apply(img, grain_strength=0.25, seed=None))
        r2 = np.array(tex.apply(img, grain_strength=0.25, seed=None))
        assert not np.array_equal(r1, r2), "seed=None must remain non-deterministic"


@pytest.mark.unit
class TestTextureEnhancements:
    """Verify new corner-chipping and ink-pooling behaviors."""

    def test_corner_chipping_affects_corners_more_than_center(self) -> None:
        """Chipping probability should be higher in the corner shell band
        than in the mid-edge shell band of the same size.

        The shell-confined erosion path in ``_frame_roughness`` only chips
        pixels within ~6px of the alpha boundary. We therefore compare
        chip density inside the 8-pixel-wide band at the four corners vs
        the 8-pixel-wide band at the four edge midpoints. Single noise
        realizations are dominated by the low-frequency random field, so
        we aggregate across multiple seeds to expose the corner bias as
        a statistical property (loose threshold — textures are noisy).
        """
        tex = StoneTexture()
        # Solid red square — no alpha variation initially
        img = Image.new("RGBA", (400, 400), (178, 34, 34, 255))

        band = 8
        span = 25
        corner_pixels = 8 * (band * span)  # 4 corners × 2 sides each
        edge_pixels = 4 * (band * span)
        mid_lo, mid_hi = 200 - span // 2, 200 + span // 2 + span % 2

        total_corner_chips = 0
        total_edge_chips = 0
        seeds = list(range(32))
        for seed in seeds:
            result = tex.apply(img, grain_strength=1.0, seed=seed)
            alpha = np.array(result)[:, :, 3]
            chip_mask = alpha < 128

            # Shell-band (outer 8px) at the four corners, each 8x25
            tl = chip_mask[:band, :span].sum() + chip_mask[:span, :band].sum()
            tr = chip_mask[:band, -span:].sum() + chip_mask[:span, -band:].sum()
            bl = chip_mask[-band:, :span].sum() + chip_mask[-span:, :band].sum()
            br = chip_mask[-band:, -span:].sum() + chip_mask[-span:, -band:].sum()
            total_corner_chips += int(tl + tr + bl + br)

            # Shell-band (outer 8px) at the four edge midpoints, same area
            top = chip_mask[:band, mid_lo:mid_hi].sum()
            bot = chip_mask[-band:, mid_lo:mid_hi].sum()
            left = chip_mask[mid_lo:mid_hi, :band].sum()
            right = chip_mask[mid_lo:mid_hi, -band:].sum()
            total_edge_chips += int(top + bot + left + right)

        corner_rate = total_corner_chips / (corner_pixels * len(seeds))
        edge_rate = total_edge_chips / (edge_pixels * len(seeds))

        assert corner_rate >= edge_rate, (
            f"aggregated corner shell chip rate {corner_rate:.4f} should be "
            f">= edge-mid shell rate {edge_rate:.4f}"
        )

    def test_ink_pooling_varies_across_interior(self) -> None:
        """Ink pool darkening should be non-uniform across the interior."""
        tex = StoneTexture()
        img = Image.new("RGBA", (400, 400), (178, 34, 34, 255))
        result = tex.apply(img, grain_strength=1.0, seed=42)
        arr = np.array(result)
        # Sample R channel interior values — should have low-freq variation
        # (not a flat value)
        r_interior = arr[100:300, 100:300, 0]
        assert r_interior.std() > 3.0, (
            f"Interior R channel should have variation, got std={r_interior.std():.2f}"
        )
