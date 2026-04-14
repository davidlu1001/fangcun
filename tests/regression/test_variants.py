"""Regression tests for SealGenerator.generate_variants().

Variants share scrape+extract+layout and differ only in texture seed.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.regression
class TestVariants:
    def test_generate_variants_returns_n_dicts(self, gen) -> None:
        variants = gen.generate_variants("禅", n=3, seeds=[1, 2, 3])
        assert len(variants) == 3
        for v in variants:
            assert "image_transparent" in v
            assert "image_preview" in v
            assert "seed" in v
            assert "font_used" in v

    def test_variants_differ_only_in_texture(self, gen) -> None:
        """Same text, different seeds → images differ but font/source/level identical."""
        variants = gen.generate_variants("禅", n=3, seeds=[1, 2, 3])
        # All should share identical source selection
        fonts = {v["font_used"] for v in variants}
        assert len(fonts) == 1, f"Fonts should be identical, got {fonts}"

        # Images must differ (different seed → different texture noise)
        imgs = [np.array(v["image_transparent"]) for v in variants]
        assert not np.array_equal(imgs[0], imgs[1]), "Seed 1 and 2 should differ"
        assert not np.array_equal(imgs[0], imgs[2]), "Seed 1 and 3 should differ"

    def test_explicit_seed_reproducible(self, gen) -> None:
        """Same seed → identical image."""
        v1 = gen.generate_variants("禅", n=1, seeds=[42])[0]
        v2 = gen.generate_variants("禅", n=1, seeds=[42])[0]
        arr1 = np.array(v1["image_transparent"])
        arr2 = np.array(v2["image_transparent"])
        assert np.array_equal(arr1, arr2)
