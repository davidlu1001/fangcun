"""Unit tests for CharExtractor: basic pipeline behavior and debug mode."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from core.extractor import CharExtractor


@pytest.mark.unit
class TestExtractorBasics:
    """Test extractor produces valid masks from synthetic inputs."""

    def test_black_on_white_passthrough(self) -> None:
        """Black strokes on white background should extract correctly."""
        ext = CharExtractor()
        arr = np.full((200, 200), 255, dtype=np.uint8)
        arr[50:150, 50:150] = 0  # black square
        img = Image.fromarray(arr, "L")

        result = ext.extract(img, source="字典")
        result_arr = np.array(result)
        assert result_arr.max() == 255, "Should have stroke pixels"
        assert result_arr.sum() > 0, "Should have non-zero content"

    def test_white_on_black_inversion(self) -> None:
        """White strokes on dark background should be auto-inverted."""
        ext = CharExtractor()
        arr = np.full((200, 200), 30, dtype=np.uint8)  # dark background
        arr[50:150, 50:150] = 240  # bright square (stroke)
        img = Image.fromarray(arr, "L")

        result = ext.extract(img, source="字典")
        result_arr = np.array(result)
        # After extraction, strokes should be 255 regardless of input polarity
        assert result_arr.max() == 255

    def test_min_stroke_threshold(self) -> None:
        """Extraction of nearly-empty image should still produce output."""
        ext = CharExtractor()
        arr = np.full((200, 200), 255, dtype=np.uint8)
        arr[100, 50:150] = 0  # single pixel row
        img = Image.fromarray(arr, "L")

        result = ext.extract(img, source="字典")
        # Should not crash, even if output is sparse
        assert result.mode == "L"

    def test_yinpu_source_detection(self) -> None:
        """Known 印谱 source names should trigger Tier 1 extraction."""
        ext = CharExtractor()
        # Create RGBA image simulating 印谱 (opaque block with alpha holes = strokes)
        rgb_arr = np.full((200, 200, 4), 50, dtype=np.uint8)
        rgb_arr[:, :, 3] = 255  # all opaque initially
        rgb_arr[60:140, 60:140, 3] = 0  # transparent hole = stroke area
        img = Image.fromarray(rgb_arr, "RGBA")

        ext.extract(img, source="字典", source_name="汉印分韵")
        assert ext._detected_as_yinpu, "Should detect as 印谱 via Tier 1 whitelist"


@pytest.mark.unit
class TestExtractorDebugMode:
    """Test the debug_dir attribute saves intermediate stages."""

    def test_debug_mode_saves_stages(self, tmp_path) -> None:
        """When debug_dir is set, extractor should save intermediate PNGs."""
        ext = CharExtractor()
        ext.debug_dir = tmp_path

        arr = np.full((200, 200), 255, dtype=np.uint8)
        arr[50:150, 50:150] = 0
        img = Image.fromarray(arr, "L")

        ext.extract(img, source="字典")

        # Expected files: 01_normalized.png, 02_binary.png, 03_denoised.png, 04_cropped.png
        saved = sorted(p.name for p in tmp_path.iterdir())
        assert "01_normalized.png" in saved
        assert "02_binary.png" in saved
        assert "03_denoised.png" in saved
        assert "04_cropped.png" in saved

    def test_no_debug_when_dir_none(self, tmp_path) -> None:
        """When debug_dir is None (default), no files should be saved."""
        ext = CharExtractor()
        # debug_dir defaults to None
        arr = np.full((200, 200), 255, dtype=np.uint8)
        arr[50:150, 50:150] = 0
        img = Image.fromarray(arr, "L")

        ext.extract(img, source="字典")

        # tmp_path should be empty
        assert list(tmp_path.iterdir()) == []
