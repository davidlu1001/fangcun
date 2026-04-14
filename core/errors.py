"""Classified failure modes for the seal generation pipeline.

Each error type carries the context needed to produce a user-actionable
message. Catch SealError to handle any pipeline failure generically;
catch a specific subtype when the calling code needs to differentiate.
"""

from __future__ import annotations


class SealError(Exception):
    """Base class for all seal generation errors."""
    pass


class CharNotFoundError(SealError):
    """Requested character not found in any available script."""

    def __init__(self, char: str, scripts_tried: list[str]) -> None:
        self.char = char
        self.scripts_tried = scripts_tried
        super().__init__(
            f"字符 '{char}' 在 {'/'.join(scripts_tried)} 中均无字源。"
            f"建议：尝试繁体输入或降低 seal_type 限制。"
        )


class SourceInconsistencyError(SealError):
    """Strict consistency mode rejected available results."""

    def __init__(self, text: str, level: int) -> None:
        self.text = text
        self.level = level
        super().__init__(
            f"'{text}' 统一来源等级为 {level}（要求 ≤2）。"
            f"建议：关闭 --strict-consistency 或减少字数。"
        )


class ExtractionFailedError(SealError):
    """Glyph found but couldn't be cleaned to acceptable quality."""

    def __init__(self, char: str, reason: str) -> None:
        self.char = char
        self.reason = reason
        super().__init__(f"字符 '{char}' 提取失败: {reason}")


class UpstreamApiError(SealError):
    """ygsf.com API is unreachable."""

    def __init__(self, status_code: int | None = None, detail: str = "") -> None:
        self.status_code = status_code
        super().__init__(
            f"ygsf.com API 不可用 (HTTP {status_code}): {detail}"
            if status_code
            else f"ygsf.com API 不可用: {detail}"
        )


class RateLimitedError(SealError):
    """ygsf.com rate-limited us."""

    def __init__(self, retry_after: float | None = None) -> None:
        self.retry_after = retry_after
        if retry_after is not None:
            super().__init__(
                f"ygsf.com 请求频率超限，建议 {retry_after:.0f}s 后重试。"
            )
        else:
            super().__init__("ygsf.com 请求频率超限，请稍后重试。")
