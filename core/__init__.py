"""
Unified seal generation pipeline.

Both app.py (Gradio) and cli.py share this single entry point.
The core package never depends on any UI framework.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path

import numpy as np
from PIL import Image

from .extractor import CharExtractor
from .layout import SealLayout
from .renderer import SealRenderer
from .scraper import CalligraphyScraper, cache_info, clear_cache
from .texture import StoneTexture

logger = logging.getLogger(__name__)

def _read_version_from_pyproject() -> str:
    """Fallback when the package isn't installed (checkout without `pip install -e`)."""
    try:
        import tomllib
        pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
        with pyproject.open("rb") as f:
            return tomllib.load(f)["project"]["version"]
    except (OSError, KeyError, ImportError):
        return "0.0.0+unknown"


try:
    __version__ = _pkg_version("fangcun")
except PackageNotFoundError:
    __version__ = _read_version_from_pyproject()


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
        return_debug: bool = False,
        user_glyphs: list[Image.Image] | None = None,
        user_glyph_polarity: str = "auto",
    ) -> dict:
        """
        Generate a complete seal image.

        Args:
            text:         seal characters (1–4 recommended)
            shape:        'oval' | 'square'
            style:        'baiwen' | 'zhuwen'
            seal_type:    'leisure' | 'name'
            color:        hex color string, e.g. '#B22222'
            grain:        texture strength 0.0–1.0
            rotation:     rotation angle in degrees
            size:         short-side pixels
            seed:         optional RNG seed for reproducible texture output
            return_debug: if True, include `image_layout_debug` (RGBA overlay
                          showing cell boundaries, ink bboxes, centroids) in
                          the result. Cheap — reuses the same placements,
                          no pipeline re-run.
            user_glyphs:  optional list of PIL images, one per character, to
                          use instead of scraping. Bypasses `_scraper` entirely
                          — no network call. `len(user_glyphs) == len(text)`
                          required.
            user_glyph_polarity: "auto" | "black_on_white" | "white_on_black".
                          Only used with `user_glyphs`. When "white_on_black"
                          (ink rubbings), the image is RGB-inverted before
                          extraction. "auto" leaves the existing extractor
                          three-tier polarity detection to decide.

        Returns:
            dict with keys: image_transparent, image_preview,
                           font_used, font_fallback, warnings,
                           consistency_level, source_names
                           (+ image_layout_debug if return_debug=True)
        """
        prep = self._prepare_placements(
            text, shape, style, seal_type, size,
            user_glyphs=user_glyphs,
            user_glyph_polarity=user_glyph_polarity,
        )
        rendered = self._render_and_texture(
            placements=prep["placements"],
            shape=shape,
            style=style,
            color=color,
            grain=grain,
            rotation=rotation,
            size=size,
            seed=seed,
        )
        result = {
            **rendered,
            "font_used": prep["font_used"],
            "font_fallback": prep["font_fallback"],
            "warnings": prep["warnings"],
            "consistency_level": prep["consistency_level"],
            "source_names": prep["source_names"],
        }
        if return_debug:
            canvas_w, canvas_h = SealRenderer.canvas_dimensions(shape, size)
            result["image_layout_debug"] = self._layout.debug_render(
                prep["placements"], (canvas_w, canvas_h),
            )
        return result

    def generate_variants(
        self,
        text: str,
        n: int = 3,
        shape: str = "oval",
        style: str = "baiwen",
        seal_type: str = "leisure",
        color: str = "#B22222",
        grain: float = 0.25,
        rotation: float = 2.0,
        size: int = 600,
        seeds: list[int] | None = None,
        user_glyphs: list[Image.Image] | None = None,
        user_glyph_polarity: str = "auto",
    ) -> list[dict]:
        """Generate N variations of a seal differing only in texture seed.

        Same source selection, same layout, same render — only the stone texture
        differs. Use when the user wants to pick the "best-looking" of several
        slight stylistic variations.

        The expensive steps (scrape + extract + layout) run ONCE; only the
        cheap per-output steps (render + texture + rotate) loop per seed.

        Args:
            n: Number of variations to produce (default 3).
            seeds: Optional explicit seed list of length n. If None, uses
                [1, 2, ..., n] for reproducibility.
            Other args: same as generate().

        Returns:
            List of dicts with the same keys as generate(), plus "seed" key.
        """
        if n < 1:
            raise ValueError(f"n must be ≥ 1, got {n}")

        if seeds is None:
            seeds = list(range(1, n + 1))
        elif len(seeds) != n:
            raise ValueError(
                f"seeds length ({len(seeds)}) must match n ({n})"
            )

        prep = self._prepare_placements(
            text, shape, style, seal_type, size,
            user_glyphs=user_glyphs,
            user_glyph_polarity=user_glyph_polarity,
        )

        variants: list[dict] = []
        for seed in seeds:
            rendered = self._render_and_texture(
                placements=prep["placements"],
                shape=shape,
                style=style,
                color=color,
                grain=grain,
                rotation=rotation,
                size=size,
                seed=seed,
            )
            variants.append({
                **rendered,
                "font_used": prep["font_used"],
                "font_fallback": prep["font_fallback"],
                "warnings": prep["warnings"],
                "consistency_level": prep["consistency_level"],
                "source_names": prep["source_names"],
                "seed": seed,
            })
        return variants

    def generate_svg(
        self,
        text: str,
        shape: str = "oval",
        style: str = "baiwen",
        seal_type: str = "leisure",
        color: str = "#B22222",
        size: int = 600,
    ) -> dict:
        """Generate a seal as SVG (clean vector, no texture).

        Texture, rotation, and preview are NOT applied. Use generate() for
        those raster-only effects. SVG output is suitable for print,
        business cards, T-shirts, posters and further editing in Illustrator
        or Inkscape.

        Returns:
            dict with keys: svg (str), font_used, font_fallback, warnings,
                           consistency_level, source_names
        """
        prep = self._prepare_placements(text, shape, style, seal_type, size)
        color_rgb = _hex_to_rgb(color)
        svg_str = self._renderer.render_svg(
            prep["placements"], shape, style, color_rgb, size,
        )
        return {
            "svg": svg_str,
            "font_used": prep["font_used"],
            "font_fallback": prep["font_fallback"],
            "warnings": prep["warnings"],
            "consistency_level": prep["consistency_level"],
            "source_names": prep["source_names"],
        }

    # ── Private helpers shared by generate() and generate_variants() ──

    def _prepare_placements(
        self,
        text: str,
        shape: str,
        style: str,
        seal_type: str,
        size: int,
        user_glyphs: list[Image.Image] | None = None,
        user_glyph_polarity: str = "auto",
    ) -> dict:
        """Run the expensive prefix: scrape → extract → layout.

        If `user_glyphs` is provided, the scraper is bypassed entirely and
        the provided images are fed straight into the extractor. Useful for
        calligraphers / researchers with their own glyph references.

        Returns a dict with absolute-canvas placements plus provenance
        metadata. Callers then feed this into `_render_and_texture` one
        or more times with different seeds.
        """
        warnings: list[str] = []

        if not text:
            raise ValueError("印章文字不能为空")

        if len(text) > 4:
            warnings.append(f"超过4字 ({len(text)}字), 将自动缩小适配")

        # ── 1. Source: scraper OR user-provided ──────────────
        #
        # 金石学原则：同一方印所有字必须同一书体。scraper-based
        # fetch_chars_consistent enforces this; user-provided mode trusts
        # the caller to pick style-consistent glyphs.
        #
        if user_glyphs is not None:
            if len(user_glyphs) != len(text):
                raise ValueError(
                    f"字数 ({len(text)}) 与上传图片数量 ({len(user_glyphs)}) 不一致"
                )

            raw_images = list(user_glyphs)
            font_used = "用户提供"
            any_fallback = False
            # Pick tab "字典" so the extractor routes to the Otsu path
            # (cleaner threshold for user uploads whose provenance we
            # don't know). Empty source_name avoids matching the
            # KNOWN_YINPU_SOURCES whitelist — polarity is settled by the
            # extractor's Tier 2/3 detection unless the caller forces it
            # via `user_glyph_polarity` below.
            tab_sources = ["字典"] * len(text)
            source_names = [""] * len(text)
            consistency_level = 0  # N/A for user-provided mode
            warnings.append("使用用户上传图片作为字源（已跳过自动选源）")

            if user_glyph_polarity != "auto":
                raw_images = [
                    self._coerce_polarity(img, user_glyph_polarity)
                    for img in raw_images
                ]
        else:
            # Original scraper path — 字典 → 真迹 tab priority, 5-level
            # source unification, etc.
            raw_images, font_used, any_fallback, tab_sources, source_names, fetch_warnings = (
                self._scraper.fetch_chars_consistent(text, seal_type)
            )
            warnings.extend(fetch_warnings)
            consistency_level = getattr(self._scraper, "_last_consistency_level", 0)

        # ── 2. Extract character masks (source-aware) ────────
        # source_name passed for Tier 1 印谱 whitelist detection.
        # When debug_dir is set, nest per-char subdirs so intermediates
        # from each char don't overwrite each other (P3#11).
        base_debug = self._extractor.debug_dir
        try:
            masks = []
            for idx, (img, tab, src_name, ch) in enumerate(
                zip(raw_images, tab_sources, source_names, text)
            ):
                if base_debug is not None:
                    self._extractor.debug_dir = base_debug / f"{idx:02d}_{ch}"
                masks.append(
                    self._extractor.extract(img, source=tab, source_name=src_name)
                )
        finally:
            self._extractor.debug_dir = base_debug

        # ── 3. Layout ────────────────────────────────────────
        ta_x, ta_y, ta_w, ta_h = SealRenderer.text_area(shape, size, style, len(text))
        placements = self._layout.arrange(masks, shape, (ta_w, ta_h), style)

        # Offset placements from text-area-local to canvas-absolute
        for p in placements:
            p["x"] += ta_x
            p["y"] += ta_y

        return {
            "placements": placements,
            "font_used": font_used,
            "font_fallback": any_fallback,
            "warnings": warnings,
            "consistency_level": consistency_level,
            "source_names": list(source_names),
        }

    def _render_and_texture(
        self,
        placements: list,
        shape: str,
        style: str,
        color: str,
        grain: float,
        rotation: float,
        size: int,
        seed: int | None,
    ) -> dict:
        """Run the cheap suffix: render → texture → rotate → preview.

        Returns a dict with `image_transparent` and `image_preview`.
        """
        color_rgb = _hex_to_rgb(color)

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
        seal = self._texture.apply(seal, grain, seed=seed, style=style)

        # ── 6. Rotation ──────────────────────────────────────
        # Square seals are never rotated: tilted corners look unbalanced and
        # the rotation bbox expansion leaves diagonal paper-white wedges
        # around the edge. Oval seals can take a small tilt — the ellipse
        # mask already hides the corner expansion.
        if shape != "square" and rotation != 0.0:
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
        }

    # ── Public debug API ────────────────────────────────────
    #
    # These methods expose debug hooks without forcing callers (CLI, tests)
    # to reach into private attributes. Backwards-compatible: the underlying
    # `_extractor.debug_dir` attribute still works for legacy callers.

    def set_extract_debug_dir(self, path: Path | None) -> None:
        """Enable/disable saving extractor intermediate stages.

        When `path` is set, every extract() call saves intermediate PNGs:
          01_normalized.png, 02_binary.png, 03_denoised.png, 04_cropped.png

        For multi-char seals, `generate()` nests per-char subdirs:
          {path}/00_{char0}/01_normalized.png, {path}/01_{char1}/..., etc.
        Set to None to disable.
        """
        self._extractor.debug_dir = path

    def render_layout_debug(
        self,
        text: str,
        shape: str = "oval",
        style: str = "baiwen",
        seal_type: str = "leisure",
        size: int = 600,
    ) -> Image.Image:
        """Render a standalone layout debug overlay.

        Runs scraper → extractor → layout (but not renderer/texture/rotation).
        Cache-backed, so no network cost after first run. If you also want the
        actual seal, prefer `generate(..., return_debug=True)` which gives you
        both in one prepare pass.

        Returns an RGBA image at canvas dimensions with:
          - Blue rectangles: cell boundaries
          - Red rectangles: ink bounding boxes
          - Green dots: pixel-weighted centroids
        """
        prep = self._prepare_placements(text, shape, style, seal_type, size)
        canvas_w, canvas_h = SealRenderer.canvas_dimensions(shape, size)
        return self._layout.debug_render(prep["placements"], (canvas_w, canvas_h))

    # ── User-glyph helpers ──────────────────────────────────

    @staticmethod
    def _coerce_polarity(img: Image.Image, polarity: str) -> Image.Image:
        """Force polarity on a user-uploaded image before extraction.

        Args:
            polarity: "black_on_white" (pass-through) or "white_on_black"
                      (RGB-invert, useful for ink-rubbing / 印谱 拓片).

        Returns:
            RGBA image in black-on-white convention (what the extractor
            expects). Alpha channel preserved so synthetic transparent
            backgrounds on user uploads still let Tier 2 alpha detection
            find the opaque region containing strokes.
        """
        if polarity == "black_on_white":
            return img.convert("RGBA")

        if polarity == "white_on_black":
            rgba = img.convert("RGBA")
            arr = np.array(rgba)
            arr[:, :, :3] = 255 - arr[:, :, :3]
            return Image.fromarray(arr, "RGBA")

        return img


def _hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    """Convert '#B22222' → (178, 34, 34)."""
    h = hex_str.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"Invalid hex color: {hex_str}")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
