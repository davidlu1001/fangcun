"""
Stone-carved texture effects for seal images.

Layers applied sequentially when grain_strength > 0:
  1. Frame edge roughness — morphological erosion + noise displacement on border area
  2. Stroke chipping     — edge-local erosion + salt noise (~0.3% density)
  3. Ink grain            — low-freq (σ=8) × 0.6 + high-freq (σ=1) × 0.4
                           alpha = grain_strength × 0.3, colored pixels only
  4. Final blur           — GaussianBlur radius 0.5
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
        h, w = alpha.shape

        # 1. Frame edge roughness
        alpha = self._frame_roughness(alpha, grain_strength)

        # 2. Stroke chipping (salt noise on visible pixels)
        alpha = self._stroke_chipping(alpha, grain_strength)

        # 3. Ink grain on colored pixels
        arr = self._ink_grain(arr, alpha, grain_strength)

        arr[:, :, 3] = alpha
        result = Image.fromarray(arr, "RGBA")

        # 4. Final subtle blur
        result = result.filter(ImageFilter.GaussianBlur(radius=0.5))

        return result

    # ── layer implementations ────────────────────────────────

    @staticmethod
    def _frame_roughness(alpha: np.ndarray, strength: float) -> np.ndarray:
        """Erode border pixels with noise-gated mask for natural edge breakup."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        eroded = cv2.erode(alpha, kernel, iterations=1)

        # Only erode where random noise says so (probability ∝ strength)
        noise_gate = np.random.random(alpha.shape).astype(np.float32)
        threshold = 1.0 - strength * 0.15
        mask = noise_gate > threshold

        result = alpha.copy()
        result[mask] = eroded[mask]
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
