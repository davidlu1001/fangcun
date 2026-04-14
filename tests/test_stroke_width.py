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


@pytest.mark.unit
class TestAnchorEligibility:
    """Verify R12 anchor-eligibility filter for extreme-aspect chars.

    Extreme-aspect chars (一 with aspect > 2.5, or narrow tall chars with
    aspect < 0.4) have rel_sw ≈ 1.0 because the whole image IS the stroke.
    Including them in the anchor median poisons the target and makes
    normal sibling chars look like big deviations. The eligibility helper
    filters them out.
    """

    def test_extreme_horizontal_aspect_excluded(self) -> None:
        """A char with aspect > 2.5 (e.g. 一) should not be anchor-eligible."""
        img = Image.new("L", (300, 100), 255)  # aspect = 3.0
        assert CalligraphyScraper._is_anchor_eligible(img) is False

    def test_extreme_vertical_aspect_excluded(self) -> None:
        """A char with aspect < 0.4 should not be anchor-eligible."""
        img = Image.new("L", (100, 300), 255)  # aspect ≈ 0.33
        assert CalligraphyScraper._is_anchor_eligible(img) is False

    def test_normal_aspect_included(self) -> None:
        """A normal aspect ratio (0.4 <= aspect <= 2.5) is anchor-eligible."""
        for w, h in [(100, 100), (150, 100), (100, 150), (200, 100), (100, 200)]:
            img = Image.new("L", (w, h), 255)
            assert CalligraphyScraper._is_anchor_eligible(img), (
                f"aspect {w / h:.2f} ({w}x{h}) should be eligible"
            )

    def test_boundary_aspect_included(self) -> None:
        """Exact boundary values (2.5 and 0.4) are inclusive."""
        wide = Image.new("L", (250, 100), 255)  # aspect = 2.5
        tall = Image.new("L", (100, 250), 255)  # aspect = 0.4
        assert CalligraphyScraper._is_anchor_eligible(wide) is True
        assert CalligraphyScraper._is_anchor_eligible(tall) is True

    def test_zero_height_rejected(self) -> None:
        """Guard against division-by-zero if height is 0."""
        img = Image.new("L", (100, 1), 255)
        # width 100, height 1 → aspect 100 → extreme, excluded
        assert CalligraphyScraper._is_anchor_eligible(img) is False


def _stroke_img(thickness: int) -> Image.Image:
    """Build a 200x200 image with a centered horizontal stroke of given thickness."""
    arr = np.full((200, 200), 255, dtype=np.uint8)
    half = thickness // 2
    arr[100 - half : 100 + half, 20:180] = 0
    return Image.fromarray(arr, "L")


@pytest.mark.unit
class TestAdaptivePick:
    """Verify the adaptive R12 window-expansion picker.

    Reproduces the 宇宙洪荒 / 中国篆刻大字典 case where the only ±5-eligible
    variant has rel_sw far from target_sw, but a Δ=10-15 lower-scoring
    variant matches almost perfectly. Adaptive expansion finds it.
    """

    def setup_method(self) -> None:
        self.scraper = CalligraphyScraper()

    def test_picks_within_tight_window_when_match_exists(self) -> None:
        """If a ±5 variant already matches target, no expansion needed."""
        # Three variants: thin/medium/thick, all in ±5 score window
        variants = [
            (_stroke_img(40), 100.0, "字典"),  # thick — top
            (_stroke_img(20), 98.0, "字典"),   # medium
            (_stroke_img(8),  97.0, "字典"),   # thin
        ]
        target_sw = self.scraper._relative_stroke_width(_stroke_img(20))
        chosen, window = self.scraper._adaptive_pick(variants, top_score=100.0, target_sw=target_sw)
        # Should pick the medium one (matches target exactly), not top (thick)
        assert chosen[1] == 98.0, f"Should pick medium (98.0), got score {chosen[1]}"
        assert window == 5.0, f"Should not expand window, got {window}"

    def test_expands_window_when_tight_misses(self) -> None:
        """When ±5 has only a poor match, expand to ±15 to find a better one."""
        # Top variant is thick. Only ±5 candidate. A medium variant exists at Δ=12.
        # 宇宙洪荒/宙 case: top=100 rel_sw=0.129, target=0.062, real best at score=88.1 (Δ=11.9, rel_sw=0.060).
        variants = [
            (_stroke_img(60), 100.0, "字典"),  # thick (only ±5 candidate)
            (_stroke_img(20), 88.0, "字典"),   # medium (Δ=12, matches target)
            (_stroke_img(10), 80.0, "字典"),   # thin (Δ=20)
        ]
        target_sw = self.scraper._relative_stroke_width(_stroke_img(20))
        chosen, window = self.scraper._adaptive_pick(variants, top_score=100.0, target_sw=target_sw)
        assert chosen[1] == 88.0, (
            f"Should expand to ±15 and pick the medium variant (88.0), got {chosen[1]}"
        )
        assert window == 15.0, f"Should have expanded to ±15, got {window}"

    def test_falls_through_to_widest_window_when_none_satisfy(self) -> None:
        """When no variant gets within deviation threshold, return widest-window best."""
        # All variants are far from target (target=very-thin, all variants thick)
        variants = [
            (_stroke_img(60), 100.0, "字典"),
            (_stroke_img(50), 90.0, "字典"),
            (_stroke_img(40), 80.0, "字典"),
        ]
        # target is way off — synthetic stroke not in any variant
        target_sw = 0.001
        chosen, window = self.scraper._adaptive_pick(variants, top_score=100.0, target_sw=target_sw)
        # Should fall through to the widest tier
        assert window == 25.0, f"Should reach widest tier, got {window}"
        # Should pick the closest available (smallest stroke) — 40
        assert chosen[1] == 80.0, f"Should pick smallest stroke (80.0), got {chosen[1]}"

    def test_single_variant_returns_unchanged(self) -> None:
        """One-element list: no choice, return it."""
        variants = [(_stroke_img(20), 100.0, "字典")]
        target_sw = 0.01
        chosen, _ = self.scraper._adaptive_pick(variants, top_score=100.0, target_sw=target_sw)
        assert chosen[1] == 100.0

    def test_quality_floor_reverts_to_top_when_both_bad(self) -> None:
        """When adaptive picker would pick a low-score variant that still
        deviates badly, revert to the top variant (trust score over harmony).
        """
        # Top is thick, far from target. Only a Δ=30 (below quality floor)
        # variant has a closer stroke, but it's still far off.
        top_sw = self.scraper._relative_stroke_width(_stroke_img(60))
        variants = [
            (_stroke_img(60), 100.0, "字典"),  # top: thick, score 100
            (_stroke_img(40), 65.0, "字典"),   # Δ=35 (below floor), still not matching
        ]
        # target is very thin — neither variant is close
        target_sw = 0.001
        chosen, window = self.scraper._adaptive_pick(
            variants, top_score=100.0, target_sw=target_sw,
        )
        # Quality floor should revert to top (score=100)
        assert chosen[1] == 100.0, (
            f"Quality floor should revert to top when Δ too large and dev still high, "
            f"got score {chosen[1]}"
        )
        # Stroke width of chosen matches top
        assert abs(self.scraper._relative_stroke_width(chosen[0]) - top_sw) < 0.001

    def test_quality_floor_accepts_large_delta_when_harmony_is_good(self) -> None:
        """Large Δ is OK if the chosen variant matches target well enough."""
        variants = [
            (_stroke_img(60), 100.0, "字典"),  # top: thick, far from target
            (_stroke_img(20), 75.0, "字典"),   # Δ=25 (within ±25 window), matches target
        ]
        target_sw = self.scraper._relative_stroke_width(_stroke_img(20))
        chosen, _ = self.scraper._adaptive_pick(
            variants, top_score=100.0, target_sw=target_sw,
        )
        # Should pick the Δ=25 variant — it matches target, harmony is good,
        # so quality floor shouldn't kick in.
        assert chosen[1] == 75.0, (
            f"Should accept Δ=25 variant when it matches target, got {chosen[1]}"
        )
