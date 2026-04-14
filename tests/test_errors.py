"""Tests for typed pipeline error classes."""

import pytest

from core.errors import (
    CharNotFoundError,
    ExtractionFailedError,
    RateLimitedError,
    SealError,
    SourceInconsistencyError,
    UpstreamApiError,
)


@pytest.mark.unit
class TestErrorHierarchy:
    """All errors must inherit from SealError so callers can catch generically."""

    def test_all_inherit_from_seal_error(self) -> None:
        errors: list[Exception] = [
            CharNotFoundError("鬱", ["篆"]),
            SourceInconsistencyError("修齐", 4),
            ExtractionFailedError("禅", "noise"),
            UpstreamApiError(500, "server error"),
            RateLimitedError(),
        ]
        for e in errors:
            assert isinstance(e, SealError)
            assert isinstance(e, Exception)

    def test_error_messages_include_suggestion(self) -> None:
        e = CharNotFoundError("鬱", ["篆", "隶"])
        msg = str(e)
        assert "建议" in msg, f"Expected 建议 (suggestion) in: {msg}"
        assert "鬱" in msg

    def test_source_inconsistency_carries_level(self) -> None:
        e = SourceInconsistencyError("修齐", 4)
        assert e.text == "修齐"
        assert e.level == 4
        assert "4" in str(e)

    def test_upstream_api_with_status_code(self) -> None:
        e = UpstreamApiError(500, "internal error")
        assert e.status_code == 500
        assert "500" in str(e)
        assert "internal error" in str(e)

    def test_upstream_api_without_status_code(self) -> None:
        e = UpstreamApiError(detail="connection refused")
        assert e.status_code is None
        assert "connection refused" in str(e)
        assert "HTTP" not in str(e), "Should not show HTTP code when None"

    def test_char_not_found_carries_scripts(self) -> None:
        e = CharNotFoundError("鬱", ["篆", "隶", "楷"])
        assert e.char == "鬱"
        assert e.scripts_tried == ["篆", "隶", "楷"]
        assert "篆/隶/楷" in str(e)

    def test_rate_limited_message(self) -> None:
        e = RateLimitedError()
        assert "频率" in str(e) or "rate" in str(e).lower()
