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


@pytest.mark.unit
class TestErrorRaiseSites:
    """Verify the right error type is raised at each pipeline boundary.

    These tests use mocks to force specific failure modes — they do not
    hit the real network. `_query_glyph_list` retries 3 times with
    exponential backoff + jitter; we monkeypatch `time.sleep` to zero so
    tests run fast, and monkeypatch `requests.Session.post` to force the
    desired failure on every retry.
    """

    def _make_scraper(self, monkeypatch):
        """Build a scraper with cache disabled and sleeps stubbed out."""
        import core.scraper as scraper_mod

        # Stub sleeps so 3 retries don't take ~10 seconds of real time.
        monkeypatch.setattr(scraper_mod.time, "sleep", lambda _s: None)
        return scraper_mod.CalligraphyScraper(no_api_cache=True)

    def test_upstream_api_error_on_connection_failure(self, monkeypatch) -> None:
        """Pure network failure (no HTTP response) -> UpstreamApiError with no status_code."""
        import requests

        def fake_post(*args, **kwargs):
            raise requests.ConnectionError("simulated network failure")

        monkeypatch.setattr(requests.Session, "post", fake_post)
        scraper = self._make_scraper(monkeypatch)

        with pytest.raises(UpstreamApiError) as excinfo:
            scraper._query_glyph_list("禅", font_style="篆", tab_type=3)
        assert excinfo.value.status_code is None
        assert "simulated network failure" in str(excinfo.value)

    def test_rate_limited_on_http_429(self, monkeypatch) -> None:
        """HTTP 429 on every attempt -> RateLimitedError."""
        import requests

        class FakeResp:
            status_code = 429
            text = ""

            def raise_for_status(self):
                raise requests.HTTPError("429", response=self)

        def fake_post(*args, **kwargs):
            return FakeResp()

        monkeypatch.setattr(requests.Session, "post", fake_post)
        scraper = self._make_scraper(monkeypatch)

        with pytest.raises(RateLimitedError):
            scraper._query_glyph_list("禅", font_style="篆", tab_type=3)

    def test_upstream_api_error_on_http_500(self, monkeypatch) -> None:
        """Non-429 HTTP error -> UpstreamApiError carrying the status_code."""
        import requests

        class FakeResp:
            status_code = 500
            text = ""

            def raise_for_status(self):
                raise requests.HTTPError("500 server error", response=self)

        def fake_post(*args, **kwargs):
            return FakeResp()

        monkeypatch.setattr(requests.Session, "post", fake_post)
        scraper = self._make_scraper(monkeypatch)

        with pytest.raises(UpstreamApiError) as excinfo:
            scraper._query_glyph_list("禅", font_style="篆", tab_type=3)
        assert excinfo.value.status_code == 500

    def test_typed_errors_catchable_as_seal_error(self, monkeypatch) -> None:
        """Callers can catch any pipeline failure via the SealError base class."""
        import requests

        def fake_post(*args, **kwargs):
            raise requests.Timeout("simulated timeout")

        monkeypatch.setattr(requests.Session, "post", fake_post)
        scraper = self._make_scraper(monkeypatch)

        with pytest.raises(SealError):
            scraper._query_glyph_list("禅", font_style="篆", tab_type=3)
