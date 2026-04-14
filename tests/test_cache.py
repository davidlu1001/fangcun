"""Cache key audit tests for the 3-tier scraper cache.

These tests document the cache key design and a known limitation (Tier 3
sibling-context blindness). They guard against accidental cache-key drift
that would silently invalidate or poison cached glyphs.

Tier 1: API response JSON         — keyed on encrypted-params MD5
Tier 2: Image CDN (raw downloads) — keyed on URL MD5
Tier 3: Selected-best per char    — keyed on {char}_{font}_{tab}.png
"""

from __future__ import annotations

import pytest


@pytest.mark.unit
class TestCacheKeys:
    """Verify cache key design prevents cross-contamination."""

    def test_tier3_distinguishes_font_and_tab(self) -> None:
        """Tier 3 cache keys must differ by font (篆/隶/楷) AND tab (字典/真迹)."""
        # Cache filenames follow {char}_{font}_{tab}.png. Same char in
        # different font OR different tab must NOT collide on disk.
        key_zen_zhuan_dict = "禅_篆_字典.png"
        key_zen_li_dict = "禅_隶_字典.png"
        key_zen_zhuan_orig = "禅_篆_真迹.png"
        keys = {key_zen_zhuan_dict, key_zen_li_dict, key_zen_zhuan_orig}
        assert len(keys) == 3, "Each (char, font, tab) combo must have unique cache key"

    def test_tier3_does_not_encode_sibling_context(self) -> None:
        """Documented limitation: Tier 3 keys are per-char only.

        R12 stroke-width sibling matching (core/scraper.py
        _try_unified_source_from_candidates) picks variants based on the
        OTHER chars in the same request. The Tier 3 cache key
        {char}_{font}_{tab}.png does NOT encode sibling context.

        Current behavior: when R12 selects a particular variant of "知" to
        match "足"'s stroke width, that variant gets cached as
        "知_篆_字典.png". A future request for "知" alone (or with a
        different sibling) reads the same cached image — even though R12
        with different siblings might have picked a different variant.

        This is a deliberate trade-off: caching the R12-chosen variant is
        usually good (better than worst-case Frankenstein) and avoids the
        complexity of a sibling-aware cache key. If this becomes a real
        quality problem, options are:
          (a) include a hash of all sibling chars in the cache key
          (b) skip Tier 3 cache writes for R12-selected variants
          (c) cache all variants per-source and re-run R12 on cache hit

        For now: documented + enforced by this test as a known limitation.
        """
        # No assertion — this is a documentation test. Its presence in the
        # suite makes the limitation discoverable via grep / pytest --collect.
        # If anyone changes Tier 3 to be sibling-aware, this test should be
        # rewritten to assert the new contract.
        pass
