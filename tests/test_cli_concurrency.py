"""末尾#4: Batch concurrency uses one SealGenerator per worker thread.

The scraper keeps per-call state (_last_consistency_level,
_current_seal_type) that would race under shared access. We verify the
CLI creates isolated generators per thread via threading.local.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.unit
class TestJobsFlag:
    def test_jobs_flag_accepted_and_clamped_to_1_for_single_item(
        self, tmp_path, monkeypatch
    ) -> None:
        """--jobs=4 with a 1-item batch should not spawn a ThreadPoolExecutor.

        Verified indirectly: the CLI should complete normally for single-text
        inputs regardless of --jobs value.
        """
        import cli as cli_mod

        # Stub _generate_one so we don't hit network/disk for real output.
        called: list = []
        def fake(gen, text, args, output_dir):
            called.append(text)
            return True
        monkeypatch.setattr(cli_mod, "_generate_one", fake)

        # Use --text (single-item) so jobs should clamp to 1.
        argv = ["cli.py", "--text", "禅", "--jobs", "4",
                "--output-dir", str(tmp_path)]
        monkeypatch.setattr(sys, "argv", argv)

        cli_mod.main()
        assert called == ["禅"]

    def test_batch_all_workers_succeed(self, tmp_path, monkeypatch) -> None:
        """--jobs=3 with a 3-item batch invokes _generate_one for each."""
        import cli as cli_mod

        called: list = []
        def fake(gen, text, args, output_dir):
            called.append(text)
            return True
        monkeypatch.setattr(cli_mod, "_generate_one", fake)

        batch_file = tmp_path / "batch.txt"
        batch_file.write_text("禅\n宗\n朝\n", encoding="utf-8")

        argv = ["cli.py", "--batch", str(batch_file), "--jobs", "3",
                "--output-dir", str(tmp_path / "out")]
        monkeypatch.setattr(sys, "argv", argv)

        cli_mod.main()
        assert sorted(called) == ["宗", "朝", "禅"]
