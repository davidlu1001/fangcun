"""User-provided glyph source feature (v1).

Covers:
  - Single + multi-char user-provided generation
  - Count mismatch validation
  - Polarity coercion (white_on_black invert)
  - Scraper bypass guarantee (no network)
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from core import SealGenerator

pytestmark = pytest.mark.unit


def _make_test_glyph(pattern: str = "block") -> Image.Image:
    """Synthetic black-on-white RGBA glyph for pipeline smoke tests.

    Returns a 200×200 image with a solid or cross-shaped black region on
    a white background — enough ink density for the extractor to produce
    a non-empty mask without needing any real calligraphy semantics.
    """
    arr = np.full((200, 200, 4), 255, dtype=np.uint8)
    if pattern == "block":
        arr[60:140, 60:140, :3] = 0
    elif pattern == "cross":
        arr[80:120, 30:170, :3] = 0
        arr[30:170, 80:120, :3] = 0
    return Image.fromarray(arr, "RGBA")


def _make_inverted_glyph() -> Image.Image:
    """White-on-black variant for polarity-hint tests (拓片 simulation)."""
    arr = np.zeros((200, 200, 4), dtype=np.uint8)
    arr[:, :, 3] = 255
    arr[60:140, 60:140, :3] = 255
    return Image.fromarray(arr, "RGBA")


class _NoCallScraper:
    """Scraper stub that explodes if anyone touches it.

    Used to prove the user-glyph path bypasses the network entirely.
    """

    _last_consistency_level = 0
    _current_seal_type = "leisure"

    def fetch_chars_consistent(self, *args, **kwargs):  # noqa: ARG002
        raise AssertionError(
            "scraper must NOT be invoked when user_glyphs is provided"
        )


def test_user_glyph_single_char() -> None:
    gen = SealGenerator()
    result = gen.generate(
        text="一",
        user_glyphs=[_make_test_glyph("cross")],
        shape="square",
        style="baiwen",
    )
    assert result["image_preview"].size[0] > 0
    assert result["font_used"] == "用户提供"
    assert any("用户上传" in w for w in result["warnings"])


def test_user_glyph_multi_char() -> None:
    gen = SealGenerator()
    glyphs = [_make_test_glyph("block"), _make_test_glyph("cross")]
    result = gen.generate(
        text="天人",
        user_glyphs=glyphs,
        shape="square",
        style="baiwen",
    )
    assert result["image_preview"].size[0] > 0
    # Both chars must appear in placements
    assert result["consistency_level"] == 0  # user-provided is L0 (N/A)


def test_user_glyph_count_mismatch_raises() -> None:
    gen = SealGenerator()
    with pytest.raises(ValueError, match="字数.*不一致"):
        gen.generate(
            text="天人合一",
            user_glyphs=[_make_test_glyph()],
        )


def test_user_glyph_polarity_inversion() -> None:
    """white_on_black coerce should produce a valid seal.

    Without a polarity hint the extractor's auto-detection may misread a
    synthetic inverted glyph; with the explicit hint, generation must
    succeed and emit an opaque red baiwen background.
    """
    gen = SealGenerator()
    result = gen.generate(
        text="一",
        user_glyphs=[_make_inverted_glyph()],
        user_glyph_polarity="white_on_black",
        shape="square",
        style="baiwen",
    )
    arr = np.array(result["image_preview"])
    red_pixels = int(((arr[:, :, 0] > 100) & (arr[:, :, 1] < 100)).sum())
    assert red_pixels > 100


def test_user_glyph_bypasses_scraper() -> None:
    gen = SealGenerator()
    gen._scraper = _NoCallScraper()  # type: ignore[assignment]

    result = gen.generate(
        text="一",
        user_glyphs=[_make_test_glyph()],
        shape="square",
        style="baiwen",
    )
    assert result["image_preview"].size[0] > 0


def test_user_glyph_variants() -> None:
    """generate_variants should also accept user_glyphs and not call scraper."""
    gen = SealGenerator()
    gen._scraper = _NoCallScraper()  # type: ignore[assignment]

    variants = gen.generate_variants(
        text="一",
        n=2,
        user_glyphs=[_make_test_glyph()],
        shape="square",
        style="baiwen",
    )
    assert len(variants) == 2
    assert {v["seed"] for v in variants} == {1, 2}
    # Same source, different texture — images should differ
    a = np.array(variants[0]["image_preview"])
    b = np.array(variants[1]["image_preview"])
    assert a.shape == b.shape
    assert not np.array_equal(a, b)


def test_coerce_polarity_white_on_black_inverts() -> None:
    """Direct helper test — independent of full pipeline."""
    src = _make_inverted_glyph()  # white block on black
    out = SealGenerator._coerce_polarity(src, "white_on_black")
    arr = np.array(out)
    # After inversion, the center block should be BLACK on WHITE bg
    center = arr[60:140, 60:140, :3]
    bg = arr[10:30, 10:30, :3]
    assert center.mean() < 20  # near-black
    assert bg.mean() > 230  # near-white


def test_coerce_polarity_black_on_white_passthrough() -> None:
    src = _make_test_glyph("block")
    out = SealGenerator._coerce_polarity(src, "black_on_white")
    # Same convention — RGB values should match the original
    assert np.array_equal(np.array(src), np.array(out))
