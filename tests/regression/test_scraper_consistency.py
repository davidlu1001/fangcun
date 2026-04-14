"""Regression tests: consistency level surfaced in generate() result dict.

These tests use the shared `gen` fixture from regression/conftest.py, so they
inherit the warm-cache SealGenerator instance (and thus may issue network
calls on a cold cache). They are marked `regression` for filtering.
"""

from __future__ import annotations

import pytest


@pytest.mark.regression
class TestConsistencyLevel:
    """Verify result dict surfaces consistency level (1-5)."""

    def test_result_has_consistency_level(self, gen) -> None:
        """Every successful generation must report consistency_level in result."""
        result = gen.generate(
            text="禅", style="baiwen", shape="oval", seal_type="leisure"
        )
        assert "consistency_level" in result
        assert 1 <= result["consistency_level"] <= 5
        assert "source_names" in result
        assert isinstance(result["source_names"], list)

    def test_single_char_is_level_1(self, gen) -> None:
        """Single-char short-circuit always achieves level 1 (one source)."""
        result = gen.generate(
            text="禅", style="baiwen", shape="oval", seal_type="leisure"
        )
        assert result["consistency_level"] == 1, (
            f"Single-char should be level 1, got {result['consistency_level']}"
        )

    def test_source_names_length_matches_text(self, gen) -> None:
        """source_names must have one entry per character in text."""
        result = gen.generate(
            text="禅", style="baiwen", shape="oval", seal_type="leisure"
        )
        assert len(result["source_names"]) == 1
