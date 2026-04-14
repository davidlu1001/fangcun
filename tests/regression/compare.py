#!/usr/bin/env python3
"""
Side-by-side regression comparison between two runs.

Usage:
    python tests/regression/compare.py baseline HEAD
    python tests/regression/compare.py baseline abc1234
"""
from __future__ import annotations

import argparse
import html
import re
import sys
from base64 import b64encode
from pathlib import Path

import numpy as np
from PIL import Image

OUTPUT_BASE = Path(__file__).parent / "output"

# Run IDs go into filenames and directory paths — restrict to a safe charset
# to prevent path traversal and shell-unfriendly names.
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def _validate_run_id(name: str) -> str:
    if not _SAFE_RUN_ID.match(name):
        print(
            f"Invalid run ID '{name}' — only [A-Za-z0-9_.-] allowed."
        )
        sys.exit(2)
    return name


def _img_to_b64(img_path: Path) -> str:
    if not img_path.exists():
        return ""
    with open(img_path, "rb") as f:
        return b64encode(f.read()).decode("ascii")


def _rmse(path_a: Path, path_b: Path) -> float:
    """Compute RMSE between two images. Returns 0.0 if identical."""
    if not path_a.exists() or not path_b.exists():
        return -1.0  # missing
    a = np.array(Image.open(path_a).convert("RGBA"), dtype=np.float64)
    b = np.array(Image.open(path_b).convert("RGBA"), dtype=np.float64)
    if a.shape != b.shape:
        # Resize b to match a for comparison
        b_img = Image.open(path_b).convert("RGBA").resize(
            (a.shape[1], a.shape[0]), Image.Resampling.LANCZOS
        )
        b = np.array(b_img, dtype=np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def compare(run_a: str, run_b: str) -> Path:
    """Generate side-by-side comparison HTML."""
    dir_a = OUTPUT_BASE / run_a
    dir_b = OUTPUT_BASE / run_b

    if not dir_a.exists():
        print(f"Run '{run_a}' not found at {dir_a}")
        sys.exit(1)
    if not dir_b.exists():
        print(f"Run '{run_b}' not found at {dir_b}")
        sys.exit(1)

    # Collect all test IDs from both runs
    ids_a = {p.stem for p in dir_a.glob("*.png")}
    ids_b = {p.stem for p in dir_b.glob("*.png")}
    all_ids = sorted(ids_a | ids_b)

    rows = []
    changed_count = 0

    for test_id in all_ids:
        path_a = dir_a / f"{test_id}.png"
        path_b = dir_b / f"{test_id}.png"

        b64_a = _img_to_b64(path_a)
        b64_b = _img_to_b64(path_b)

        img_a_html = f'<img src="data:image/png;base64,{b64_a}" width="200">' if b64_a else '<span class="missing">MISSING</span>'
        img_b_html = f'<img src="data:image/png;base64,{b64_b}" width="200">' if b64_b else '<span class="missing">MISSING</span>'

        rmse = _rmse(path_a, path_b)
        if rmse < 0:
            status = "MISSING"
            row_class = "missing-row"
            changed_count += 1
        elif rmse < 0.5:
            status = "IDENTICAL"
            row_class = "identical"
        elif rmse < 5.0:
            status = f"MINOR (rmse={rmse:.1f})"
            row_class = "minor"
            changed_count += 1
        else:
            status = f"CHANGED (rmse={rmse:.1f})"
            row_class = "changed"
            changed_count += 1

        rows.append(f"""
        <tr class="{row_class}">
            <td>{html.escape(test_id)}</td>
            <td>{img_a_html}</td>
            <td>{img_b_html}</td>
            <td class="status">{html.escape(status)}</td>
        </tr>""")

    report_path = OUTPUT_BASE / f"compare_{run_a}_vs_{run_b}.html"
    html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Compare: {html.escape(run_a)} vs {html.escape(run_b)}</title>
<style>
body {{ font-family: monospace; margin: 20px; background: #fafafa; }}
h1 {{ font-size: 18px; }}
.summary {{ margin: 10px 0; padding: 8px; background: #eee; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ccc; padding: 6px 8px; vertical-align: middle; text-align: center; }}
th {{ background: #333; color: #fff; }}
.identical {{ background: #f0fff0; }}
.minor {{ background: #fffff0; }}
.changed {{ background: #fff0f0; }}
.missing-row {{ background: #f0f0ff; }}
.status {{ font-weight: bold; white-space: nowrap; }}
.missing {{ color: #999; font-style: italic; }}
img {{ border: 1px solid #ddd; }}
</style>
</head>
<body>
<h1>Regression Comparison: {html.escape(run_a)} vs {html.escape(run_b)}</h1>
<div class="summary">
    Total: {len(all_ids)} test cases | Changed: {changed_count} | Identical: {len(all_ids) - changed_count}
</div>
<table>
<tr>
    <th>Test ID</th>
    <th>{html.escape(run_a)} (before)</th>
    <th>{html.escape(run_b)} (after)</th>
    <th>Delta</th>
</tr>
{"".join(rows)}
</table>
</body>
</html>"""

    report_path.write_text(html_content, encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two regression runs")
    parser.add_argument("run_a", help="First run ID (e.g. 'baseline')")
    parser.add_argument("run_b", help="Second run ID (e.g. git SHA or 'HEAD')")
    args = parser.parse_args()

    run_a = _validate_run_id(args.run_a)
    run_b = _validate_run_id(args.run_b)

    report = compare(run_a, run_b)
    print(f"Comparison report: {report}")


if __name__ == "__main__":
    main()
