"""
Stone-carved texture effects for seal images.

Style-aware: baiwen (red background + white text cutouts) needs to treat
white-cutout pixels as "paper" (skip grain/drift/pressure-on-alpha there),
whereas zhuwen treats every visible pixel as ink.

Layers applied sequentially when grain_strength > 0:
  1. Pressure variation  — baiwen: RGB brightness on ink only;
                           zhuwen: low-freq alpha modulation
  2. Frame edge roughness — shell-confined erosion + downsampled noise
  3. Stroke chipping     — edge-local erosion + salt noise (~0.3% density)
  4. Ink grain            — low-freq (σ=8) × 0.6 + high-freq (σ=1) × 0.4
                           on ink_mask only
  4.5 Stroke-intersection darkening — capped distance transform on ink_mask
  4.6 Color temp drift   — low-freq RGB shift on ink_mask only
  5. Final blur           — GaussianBlur radius 0.5
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageFilter

from .renderer import PAPER_COLOR


class StoneTexture:
    """Apply gold-and-stone (金石) texture effects to a rendered seal image."""

    def apply(
        self,
        img: Image.Image,
        grain_strength: float = 0.25,
        seed: int | None = None,
        style: str = "baiwen",
    ) -> Image.Image:
        """
        Args:
            img:            RGBA seal image
            grain_strength: 0.0 = clean digital, 1.0 = maximum roughness
            seed:           optional RNG seed for reproducible texture output.
                            Uses a *local* np.random.Generator — no global
                            numpy state is touched, so concurrent callers
                            with different seeds don't interfere.
            style:          'baiwen' | 'zhuwen'. Controls ink_mask
                            construction and pressure-variation path.

        Returns:
            New RGBA image with texture applied.
        """
        if grain_strength <= 0.0:
            return img.copy()

        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

        arr = np.array(img, dtype=np.uint8).copy()
        alpha = arr[:, :, 3].copy()

        # Build ink_mask: True for real-ink pixels, False for paper cutouts
        # and transparent areas. For baiwen, exclude pixels near PAPER_COLOR
        # so white cutouts (text) don't get red noise / color drift / dark
        # pressure tint.
        if style == "baiwen":
            paper = np.array(PAPER_COLOR, dtype=np.float32)
            rgb_dist = np.sqrt(
                np.sum((arr[:, :, :3].astype(np.float32) - paper) ** 2, axis=2)
            )
            ink_mask = (alpha > 128) & (rgb_dist > 30)
        else:
            ink_mask = alpha > 128

        # 1. Pressure variation (style-aware)
        if style == "baiwen":
            # Baiwen: brightness modulation on ink only, keep alpha opaque
            arr = self._pressure_variation_rgb(arr, ink_mask, grain_strength, rng)
        else:
            alpha = self._pressure_variation(alpha, grain_strength, rng)

        # 2. Frame edge roughness (alpha-based — OK for both styles)
        alpha = self._frame_roughness(alpha, grain_strength, rng)

        # 3. Stroke chipping (alpha-based — OK for both styles)
        alpha = self._stroke_chipping(alpha, grain_strength, rng)

        # 4. Ink grain on ink pixels
        arr = self._ink_grain(arr, ink_mask, grain_strength, rng)

        # 4.5 Stroke intersection darkening (capped distance)
        arr = self._stroke_intersection_darkening(
            arr, ink_mask, grain_strength, rng
        )

        # 4.6 Color temperature drift on ink pixels
        arr = self._color_temperature_drift(arr, ink_mask, grain_strength, rng)

        arr[:, :, 3] = alpha
        result = Image.fromarray(arr, "RGBA")

        # 5. Final subtle blur
        result = result.filter(ImageFilter.GaussianBlur(radius=0.5))

        return result

    # ── layer implementations ────────────────────────────────

    @staticmethod
    def _frame_roughness(
        alpha: np.ndarray, strength: float, rng: np.random.Generator
    ) -> np.ndarray:
        """
        Block-shaped stone chipping with dual-path: shell erosion for thick
        areas, random line breakage for thin lines.

        Thin-line detection: erode(7x7) once; if < 20% pixels survive,
        the scene is dominated by thin lines (zhuwen frame). Use denser
        noise to punch random gaps along the line instead of shell erosion.

        Thick areas use shell-confined erosion with radial weighting and
        contagion expansion (R2-P2-1) so adjacent frame lines share damage.
        """
        if strength <= 0:
            return alpha

        h, w = alpha.shape
        visible_count = int(np.sum(alpha > 128))
        if visible_count == 0:
            return alpha

        # ── Thin-line detection ─────────────────────────────
        thin_probe = cv2.erode(
            alpha,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            iterations=1,
        )
        thin_ratio = float(np.sum(thin_probe > 128)) / max(1, visible_count)

        if thin_ratio < 0.20:
            # Thin-line path: random gap breakage along lines
            sh, sw = max(1, h // 6), max(1, w // 6)
            raw = rng.standard_normal((sh, sw)).astype(np.float32)
            blurred = cv2.GaussianBlur(raw, (0, 0), sigmaX=3, sigmaY=3)
            blurred = cv2.resize(blurred, (w, h), interpolation=cv2.INTER_CUBIC)
            mn, mx = blurred.min(), blurred.max()
            if mx > mn:
                blurred = (blurred - mn) / (mx - mn)
            break_mask = (blurred > (1.0 - strength * 0.18)) & (alpha > 128)
            result = alpha.copy()
            result[break_mask] = 0
            return result

        # ── Thick-area path: shell-confined erosion ─────────

        # 1. Downsampled low-frequency noise (fast path)
        sh, sw = max(1, h // 8), max(1, w // 8)
        raw = rng.standard_normal((sh, sw)).astype(np.float32)
        blurred = cv2.GaussianBlur(raw, (0, 0), sigmaX=4, sigmaY=4)
        blurred = cv2.resize(blurred, (w, h), interpolation=cv2.INTER_CUBIC)
        n_min, n_max = blurred.min(), blurred.max()
        if n_max > n_min:
            blurred = (blurred - n_min) / (n_max - n_min)

        # 2. Morphological shell: pixels within ~6px of alpha boundary
        kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        inner = cv2.erode(alpha, kernel5, iterations=3)
        shell_mask = (alpha > 128) & (inner < 128)

        # 3. Radial weight: outer 25% gets +15% chip probability
        h_grid, w_grid = np.ogrid[:h, :w]
        cy, cx = h / 2.0, w / 2.0
        radial = np.sqrt(
            ((h_grid - cy) / max(h / 2.0, 1)) ** 2
            + ((w_grid - cx) / max(w / 2.0, 1)) ** 2
        )
        edge_boost = np.clip((radial - 0.75) / 0.25, 0, 1).astype(np.float32)

        # 3b. Corner proximity: real seals wear most at the four corners
        # (storage bumps + initial stamp contact). Both normalized axes
        # must be near 1.0 simultaneously to qualify — edges have high
        # ny OR nx but not both. Use min(ny,nx) which is only near 1.0
        # at the four corners (edges have min≈0).
        ny = np.abs(h_grid - cy) / max(h / 2.0, 1)
        nx = np.abs(w_grid - cx) / max(w / 2.0, 1)
        corner_closeness = np.minimum(ny, nx).astype(np.float32)
        # Kick in within ~15% of the corner (corner_closeness > 0.85)
        corner_boost = np.clip(
            (corner_closeness - 0.85) / 0.15, 0, 1
        ).astype(np.float32)

        # Cap total additive boost at 0.30. Corner weight lifted from 0.18
        # → 0.22 so the four corners actually read as the most worn area
        # (real seals wear first at corners).
        total_boost = np.minimum(
            edge_boost * 0.15 + corner_boost * 0.22, 0.30
        )
        combined_prob = blurred + total_boost

        # 4. Conservative threshold (legibility over art)
        threshold = 1.0 - strength * 0.25
        final_chip_mask = (combined_prob > threshold) & shell_mask

        # 5. Contagion: expand chips ~2.5% to hit adjacent frame lines
        contagion_r = max(8, int(min(h, w) * 0.025))
        contagion_k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (contagion_r, contagion_r)
        )
        expanded = cv2.dilate(
            final_chip_mask.astype(np.uint8), contagion_k, iterations=1
        ).astype(bool)
        combined_chip = (final_chip_mask | (expanded & shell_mask))

        kernel_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        eroded = cv2.erode(alpha, kernel_e, iterations=2)

        result = alpha.copy()
        result[combined_chip] = eroded[combined_chip]
        return result

    @staticmethod
    def _stroke_chipping(
        alpha: np.ndarray, strength: float, rng: np.random.Generator
    ) -> np.ndarray:
        """Add salt noise (random transparent holes) to visible pixels."""
        density = 0.003 * strength
        visible = alpha > 128
        salt = rng.random(alpha.shape) < density

        result = alpha.copy()
        result[visible & salt] = 0
        return result

    @staticmethod
    def _ink_grain(
        arr: np.ndarray,
        ink_mask: np.ndarray,
        strength: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Layer low-freq and high-freq noise on ink pixels for ink texture."""
        h, w = ink_mask.shape

        if not np.any(ink_mask):
            return arr

        # Low-frequency noise (smooth undulations)
        low_raw = rng.standard_normal((h, w)).astype(np.float32)
        low_freq = cv2.GaussianBlur(low_raw, (0, 0), sigmaX=8, sigmaY=8)
        # Normalize to [-1, 1]
        lf_min, lf_max = low_freq.min(), low_freq.max()
        if lf_max - lf_min > 0:
            low_freq = (low_freq - lf_min) / (lf_max - lf_min) * 2 - 1

        # High-frequency noise (fine grain)
        high_freq = rng.standard_normal((h, w)).astype(np.float32)
        hf_min, hf_max = high_freq.min(), high_freq.max()
        if hf_max - hf_min > 0:
            high_freq = (high_freq - hf_min) / (hf_max - hf_min) * 2 - 1

        # Blend: 60% low + 40% high
        grain = low_freq * 0.6 + high_freq * 0.4  # range ~ [-1, 1]
        grain_amount = strength * 0.3  # max ±30% color shift at strength=1

        result = arr.copy()
        for c in range(3):
            channel = result[:, :, c].astype(np.float32)
            delta = grain * grain_amount * 255
            channel[ink_mask] += delta[ink_mask]
            result[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)

        return result

    @staticmethod
    def _color_temperature_drift(
        arr: np.ndarray,
        ink_mask: np.ndarray,
        strength: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Add low-frequency R/G/B drift to simulate ink paste unevenness.

        Uses downsampled noise for speed. R channel drifts most (ink red
        variation is most visible). Max ±12 color levels at strength=1.
        """
        if strength <= 0:
            return arr

        h, w = ink_mask.shape
        if not np.any(ink_mask):
            return arr

        result = arr.copy()
        drift_amount = strength * 12.0
        # Downsampled low-freq noise per channel
        sh, sw = max(1, h // 8), max(1, w // 8)
        sigma = max(3.0, min(sw, sh) * 0.06)

        # R drifts most; G/B more subdued so the red reads warm-then-cool
        # rather than desaturating toward grey.
        weights = [1.0, 0.45, 0.25]
        for c in range(3):
            raw = rng.standard_normal((sh, sw)).astype(np.float32)
            blurred = cv2.GaussianBlur(raw, (0, 0), sigmaX=sigma, sigmaY=sigma)
            drift = cv2.resize(blurred, (w, h), interpolation=cv2.INTER_CUBIC)
            mn, mx = drift.min(), drift.max()
            if mx > mn:
                drift = (drift - mn) / (mx - mn) * 2 - 1

            channel = result[:, :, c].astype(np.float32)
            channel[ink_mask] += drift[ink_mask] * drift_amount * weights[c]
            result[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)

        return result

    @staticmethod
    def _pressure_variation(
        alpha: np.ndarray, strength: float, rng: np.random.Generator
    ) -> np.ndarray:
        """Low-frequency multiplicative alpha modulation for stamp pressure unevenness.

        Used only for zhuwen (alpha-based fading). Baiwen uses
        ``_pressure_variation_rgb`` instead, because modulating alpha on a
        fully opaque baiwen seal would make the red background semi-transparent.

        Multiplier range [0.82, 1.0] at strength=1.
        """
        if strength <= 0:
            return alpha

        h, w = alpha.shape
        sh, sw = max(1, h // 8), max(1, w // 8)
        raw = rng.standard_normal((sh, sw)).astype(np.float32)
        sigma = max(3.0, min(sw, sh) * 0.08)
        field = cv2.GaussianBlur(raw, (0, 0), sigmaX=sigma, sigmaY=sigma)
        field = cv2.resize(field, (w, h), interpolation=cv2.INTER_CUBIC)
        mn, mx = field.min(), field.max()
        if mx > mn:
            field = (field - mn) / (mx - mn)

        # [0.82, 1.0] multiplier range
        multiplier = 1.0 - field * strength * 0.18
        result = (alpha.astype(np.float32) * multiplier).clip(0, 255).astype(np.uint8)
        return result

    @staticmethod
    def _pressure_variation_rgb(
        arr: np.ndarray,
        ink_mask: np.ndarray,
        strength: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Brightness-based pressure variation for baiwen (keeps alpha intact).

        Same low-frequency field as the alpha-based version, but modulates RGB
        brightness of ink pixels instead. Multiplier range [0.90, 1.0] at
        strength=1 — subtler than the alpha version's [0.82, 1.0] because
        color shifts are more perceptible than transparency changes, and
        baiwen's red background never goes see-through on real paper.
        """
        if strength <= 0 or not np.any(ink_mask):
            return arr

        h, w = arr.shape[:2]
        sh, sw = max(1, h // 8), max(1, w // 8)
        raw = rng.standard_normal((sh, sw)).astype(np.float32)
        sigma = max(3.0, min(sw, sh) * 0.08)
        field = cv2.GaussianBlur(raw, (0, 0), sigmaX=sigma, sigmaY=sigma)
        field = cv2.resize(field, (w, h), interpolation=cv2.INTER_CUBIC)
        mn, mx = field.min(), field.max()
        if mx > mn:
            field = (field - mn) / (mx - mn)

        multiplier = 1.0 - field * strength * 0.10

        result = arr.copy()
        for c in range(3):
            channel = result[:, :, c].astype(np.float32)
            channel[ink_mask] *= multiplier[ink_mask]
            result[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)
        return result

    @staticmethod
    def _stroke_intersection_darkening(
        arr: np.ndarray,
        ink_mask: np.ndarray,
        strength: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Darken stroke interiors/crossings to simulate ink pooling.

        Uses distance transform on ``ink_mask``: pixels far from the ink
        boundary are deep inside strokes (or at intersections). The maximum
        influence distance is capped to ``max(w, h) * 0.03`` (≈18px @ 600px)
        — real ink pooling only occurs within ~15px of stroke edges. Without
        the cap, a near-fully-opaque baiwen mask yields d_max of 200+ and
        the "deep interior" mask blackens the entire image center.
        """
        if strength <= 0:
            return arr

        if not np.any(ink_mask):
            return arr

        dist = cv2.distanceTransform(ink_mask.astype(np.uint8), cv2.DIST_L2, 5)
        d_max = dist.max()
        if d_max <= 0:
            return arr

        h, w = ink_mask.shape
        cap = min(float(d_max), max(w, h) * 0.03)
        if cap <= 0:
            return arr

        dist_norm = np.clip(dist / cap, 0, 1)
        deep_mask = dist_norm > 0.3
        if not np.any(deep_mask):
            return arr

        darken_amount = strength * 30.0

        # Low-frequency multiplicative noise: real ink pools unevenly,
        # with splotches of deeper ink rather than uniform interior darkening.
        raw = rng.standard_normal((h, w)).astype(np.float32)
        low_freq = cv2.GaussianBlur(raw, (0, 0), sigmaX=15, sigmaY=15)
        lf_min, lf_max = low_freq.min(), low_freq.max()
        if lf_max > lf_min:
            low_freq = (low_freq - lf_min) / (lf_max - lf_min)
        # Normalize to [0.5, 1.5] splotch multiplier
        splotch = 0.5 + low_freq

        result = arr.copy()
        for c in range(3):
            channel = result[:, :, c].astype(np.float32)
            channel[deep_mask] -= (
                dist_norm[deep_mask] * darken_amount * splotch[deep_mask]
            )
            result[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)
        return result
