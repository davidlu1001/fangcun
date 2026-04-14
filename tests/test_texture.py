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
