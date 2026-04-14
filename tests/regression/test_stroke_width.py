"""Regression tests: R12 stroke-width source-selection logging.

Uses the shared `gen` fixture from regression/conftest.py (warm cache).
"""

from __future__ import annotations

import logging

import pytest


@pytest.mark.regression
class TestR12StrokeWidth:
    """Verify R12 stroke-width matching is invoked and logged for multi-char seals."""

    def test_r12_invoked_for_multichar(self, gen, caplog) -> None:
        """Multi-char seal should trigger R12 or at least log source selection."""
        with caplog.at_level(logging.DEBUG, logger="core.scraper"):
            gen.generate(
                text="知足", style="baiwen", shape="square", seal_type="leisure"
            )

        # Should see either R12 stroke matching or unified source selection
        logs = caplog.text
        has_source_log = (
            "统一来源" in logs or "[R12]" in logs or "[Final]" in logs
        )
        assert has_source_log, "Multi-char seal should log source selection path"
