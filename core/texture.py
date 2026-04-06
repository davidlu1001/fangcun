"""
Stone-carved texture effects for seal images.

Layers applied sequentially when grain_strength > 0:
  1. Pressure variation  — low-freq alpha modulation (stamp unevenness)
  2. Frame edge roughness — shell-confined erosion + downsampled noise
  3. Stroke chipping     — edge-local erosion + salt noise (~0.3% density)
  4. Ink grain            — low-freq (σ=8) × 0.6 + high-freq (σ=1) × 0.4
                           alpha = grain_strength × 0.3, colored pixels only
  4.5 Color temp drift   — low-freq RGB shift on colored pixels
  5. Final blur           — GaussianBlur radius 0.5
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageFilter


class StoneTexture:
    """Apply gold-and-stone (金石) texture effects to a rendered seal image."""

    def apply(self, img: Image.Image, grain_strength: float = 0.25) -> Image.Image:
        """
        Args:
            img:            RGBA seal image
            grain_strength: 0.0 = clean digital, 1.0 = maximum roughness

        Returns:
            New RGBA image with texture applied.
        """
        if grain_strength <= 0.0:
            return img.copy()

        arr = np.array(img, dtype=np.uint8).copy()
        alpha = arr[:, :, 3].copy()

        # 1. Pressure variation (before roughness — it's a global modulation)
        alpha = self._pressure_variation(alpha, grain_strength)

        # 2. Frame edge roughness (shell-confined)
        alpha = self._frame_roughness(alpha, grain_strength)

        # 3. Stroke chipping (salt noise on visible pixels)
        alpha = self._stroke_chipping(alpha, grain_strength)

        # 4. Ink grain on colored pixels
        arr = self._ink_grain(arr, alpha, grain_strength)

        # 4.5 Stroke intersection darkening (ink pooling)
        arr = self._stroke_intersection_darkening(arr, alpha, grain_strength)

        # 4.6 Color temperature drift
        arr = self._color_temperature_drift(arr, alpha, grain_strength)

        arr[:, :, 3] = alpha
        result = Image.fromarray(arr, "RGBA")

        # 5. Final subtle blur
        result = result.filter(ImageFilter.GaussianBlur(radius=0.5))

        return result

    # ── layer implementations ────────────────────────────────

    @staticmethod
    def _frame_roughness(alpha: np.ndarray, strength: float) -> np.ndarray:
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
            raw = np.random.randn(sh, sw).astype(np.float32)
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
        raw = np.random.randn(sh, sw).astype(np.float32)
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
        combined_prob = blurred + edge_boost * 0.15

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
    def _stroke_chipping(alpha: np.ndarray, strength: float) -> np.ndarray:
        """Add salt noise (random transparent holes) to visible pixels."""
        density = 0.003 * strength
        visible = alpha > 128
        salt = np.random.random(alpha.shape) < density

        result = alpha.copy()
        result[visible & salt] = 0
        return result

    @staticmethod
    def _ink_grain(
        arr: np.ndarray, alpha: np.ndarray, strength: float
    ) -> np.ndarray:
        """Layer low-freq and high-freq noise on colored pixels for ink texture."""
        h, w = alpha.shape
        colored = alpha > 128

        if not np.any(colored):
            return arr

        # Low-frequency noise (smooth undulations)
        low_raw = np.random.randn(h, w).astype(np.float32)
        low_freq = cv2.GaussianBlur(low_raw, (0, 0), sigmaX=8, sigmaY=8)
        # Normalize to [-1, 1]
        lf_min, lf_max = low_freq.min(), low_freq.max()
        if lf_max - lf_min > 0:
            low_freq = (low_freq - lf_min) / (lf_max - lf_min) * 2 - 1

        # High-frequency noise (fine grain)
        high_freq = np.random.randn(h, w).astype(np.float32)
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
            channel[colored] += delta[colored]
            result[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)

        return result

    @staticmethod
    def _color_temperature_drift(
        arr: np.ndarray, alpha: np.ndarray, strength: float
    ) -> np.ndarray:
        """Add low-frequency R/G/B drift to simulate ink paste unevenness.

        Uses downsampled noise for speed. R channel drifts most (ink red
        variation is most visible). Max +-8 color levels at strength=1.
        """
        if strength <= 0:
            return arr

        h, w = alpha.shape
        colored = alpha > 128
        if not np.any(colored):
            return arr

        result = arr.copy()
        drift_amount = strength * 8.0
        # Downsampled low-freq noise per channel
        sh, sw = max(1, h // 8), max(1, w // 8)
        sigma = max(3.0, min(sw, sh) * 0.06)

        weights = [1.0, 0.6, 0.4]  # R drifts most
        for c in range(3):
            raw = np.random.randn(sh, sw).astype(np.float32)
            blurred = cv2.GaussianBlur(raw, (0, 0), sigmaX=sigma, sigmaY=sigma)
            drift = cv2.resize(blurred, (w, h), interpolation=cv2.INTER_CUBIC)
            mn, mx = drift.min(), drift.max()
            if mx > mn:
                drift = (drift - mn) / (mx - mn) * 2 - 1

            channel = result[:, :, c].astype(np.float32)
            channel[colored] += drift[colored] * drift_amount * weights[c]
            result[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)

        return result

    @staticmethod
    def _pressure_variation(alpha: np.ndarray, strength: float) -> np.ndarray:
        """Low-frequency multiplicative alpha modulation for stamp pressure unevenness.

        Produces 2-3 slightly faded areas where the stamp wasn't pressed evenly.
        Multiplier range [0.82, 1.0] at strength=1. Uses downsampled noise.
        """
        if strength <= 0:
            return alpha

        h, w = alpha.shape
        sh, sw = max(1, h // 8), max(1, w // 8)
        raw = np.random.randn(sh, sw).astype(np.float32)
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
    def _stroke_intersection_darkening(
        arr: np.ndarray, alpha: np.ndarray, strength: float
    ) -> np.ndarray:
        """Darken stroke interiors/crossings to simulate ink pooling.

        Uses distance transform: pixels far from alpha boundary are deep
        inside strokes (or at intersections). Darkens these by up to 30
        color levels at strength=1.
        """
        if strength <= 0:
            return arr

        visible = (alpha > 128).astype(np.uint8)
        if not np.any(visible):
            return arr

        dist = cv2.distanceTransform(visible, cv2.DIST_L2, 5)
        d_max = dist.max()
        if d_max <= 0:
            return arr

        dist_norm = dist / d_max
        deep_mask = dist_norm > 0.3
        if not np.any(deep_mask):
            return arr

        darken_amount = strength * 30.0
        result = arr.copy()
        for c in range(3):
            channel = result[:, :, c].astype(np.float32)
            channel[deep_mask] -= dist_norm[deep_mask] * darken_amount
            result[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)
        return result
