"""
Unified seal generation pipeline.

Both app.py (Gradio) and cli.py share this single entry point.
The core package never depends on any UI framework.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .extractor import CharExtractor
from .layout import SealLayout
from .renderer import SealRenderer
from .scraper import CalligraphyScraper, cache_info, clear_cache
from .texture import StoneTexture

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SealResult:
    """Immutable result from seal generation."""

    image_transparent: Image.Image
    image_preview: Image.Image
    font_used: str
    font_fallback: bool
    warnings: tuple[str, ...]


class SealGenerator:
    """
    Orchestrate the full pipeline:
      scraper → extractor → layout → renderer → texture → rotation
    """

    def __init__(self, no_api_cache: bool = False) -> None:
        self._scraper = CalligraphyScraper(no_api_cache=no_api_cache)
        self._extractor = CharExtractor()
        self._layout = SealLayout()
        self._renderer = SealRenderer()
        self._texture = StoneTexture()

    def generate(
        self,
        text: str,
        shape: str = "oval",
        style: str = "baiwen",
        seal_type: str = "leisure",
        color: str = "#B22222",
        grain: float = 0.25,
        rotation: float = 2.0,
        size: int = 600,
        seed: int | None = None,
    ) -> dict:
        """
        Generate a complete seal image.

        Args:
            text:      seal characters (1–4 recommended)
            shape:     'oval' | 'square'
            style:     'baiwen' | 'zhuwen'
            seal_type: 'leisure' | 'name'
            color:     hex color string, e.g. '#B22222'
            grain:     texture strength 0.0–1.0
            rotation:  rotation angle in degrees
            size:      short-side pixels
            seed:      optional RNG seed for reproducible texture output

        Returns:
            dict with keys: image_transparent, image_preview,
                           font_used, font_fallback, warnings
        """
        warnings: list[str] = []
        color_rgb = _hex_to_rgb(color)

        if not text:
            raise ValueError("印章文字不能为空")

        if len(text) > 4:
            warnings.append(f"超过4字 ({len(text)}字), 将自动缩小适配")

        # ── 1. Fetch character images (consistency-first) ────
        #
        # 金石学原则：同一方印所有字必须同一书体。
        # fetch_chars_consistent tries each priority style for ALL
        # characters before falling back, so "禅宗" won't mix 篆+隶.
        #
        # Tab priority per font style: 字典 → 真迹 (不用字库).
        # Tab source affects extractor preprocessing intensity.
        #
        raw_images, font_used, any_fallback, tab_sources, source_names, fetch_warnings = (
            self._scraper.fetch_chars_consistent(text, seal_type)
        )
        warnings.extend(fetch_warnings)
        font_display = font_used
        consistency_level = getattr(self._scraper, "_last_consistency_level", 0)

        # ── 2. Extract character masks (source-aware) ────────
        # source_name passed for Tier 1 印谱 whitelist detection
        masks = [
            self._extractor.extract(img, source=tab, source_name=src_name)
            for img, tab, src_name in zip(raw_images, tab_sources, source_names)
        ]

        # ── 3. Layout ────────────────────────────────────────
        ta_x, ta_y, ta_w, ta_h = SealRenderer.text_area(shape, size, style, len(text))
        placements = self._layout.arrange(masks, shape, (ta_w, ta_h), style)

        # Offset placements from text-area-local to canvas-absolute
        for p in placements:
            p["x"] += ta_x
            p["y"] += ta_y

        # ── 4. Render ────────────────────────────────────────
        seal = self._renderer.render(placements, shape, style, color_rgb, size)

        # Style-exclusivity sanity check: baiwen must yield an opaque seal,
        # zhuwen must yield a mostly-transparent one. Catches yin/yang
        # mixing bugs before the texture/rotation stages obscure them.
        #
        # Thresholds are shape-agnostic with ≥2× safety margin:
        #   - Oval baiwen opaque floor ≈ π/4 ≈ 0.785 (inscribed ellipse area)
        #     minus char cutout, so 0.5 is conservative.
        #   - Oval zhuwen transparent ≈ 0.85+ (mostly empty inside ellipse +
        #     fully-clipped corners), so 0.2 is conservative.
        if __debug__:
            _arr = np.array(seal)
            if style == "baiwen":
                _opaque = (_arr[:, :, 3] > 200).sum() / _arr[:, :, 3].size
                assert _opaque > 0.5, (
                    f"Baiwen style sanity check failed: opaque_ratio={_opaque:.2f}"
                )
            else:
                _transparent = (_arr[:, :, 3] < 10).sum() / _arr[:, :, 3].size
                assert _transparent > 0.2, (
                    f"Zhuwen style sanity check failed: transparent_ratio={_transparent:.2f}"
                )

        # ── 5. Texture ───────────────────────────────────────
        seal = self._texture.apply(seal, grain, seed=seed)

        # ── 6. Rotation ──────────────────────────────────────
        if rotation != 0.0:
            seal = seal.rotate(
                rotation,
                resample=Image.Resampling.BICUBIC,
                expand=True,
            )

        # ── 7. White-background preview ──────────────────────
        preview = Image.new("RGBA", seal.size, (255, 255, 255, 255))
        preview = Image.alpha_composite(preview, seal)

        return {
            "image_transparent": seal,
            "image_preview": preview,
            "font_used": font_display,
            "font_fallback": any_fallback,
            "warnings": warnings,
            "consistency_level": consistency_level,
            "source_names": list(source_names),
        }


def _hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    """Convert '#B22222' → (178, 34, 34)."""
    h = hex_str.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"Invalid hex color: {hex_str}")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
