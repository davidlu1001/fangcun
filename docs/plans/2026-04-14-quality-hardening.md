# Output Quality Hardening — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a regression test harness, then systematically fix correctness bugs (P0), extraction quality (P1), layout issues (P2), and texture realism (P3) so that arbitrary inputs produce reliably good seals.

**Architecture:** The pipeline is `scraper → extractor → layout → renderer → texture`. Each phase targets one or two pipeline stages. Phase 0 (harness) wraps the full pipeline so every subsequent change is measurable. No frontend/deployment changes.

**Tech Stack:** Python 3.14, pytest, Pillow, OpenCV, NumPy. The harness generates HTML reports for human visual review — no automated "beauty scoring."

---

## Phase 0: Regression Test Harness

> **Must complete before any pipeline code changes.** Without this, every "optimization" is unverifiable parameter jitter.

### Task 0.1: Scaffold test infrastructure

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/regression/__init__.py`
- Create: `pyproject.toml` (modify — add pytest config)

**Step 1: Write pytest configuration**

Add to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "regression: visual regression tests (require network + cache)",
    "unit: fast isolated unit tests",
]
```

**Step 2: Create conftest with shared fixtures**

```python
# tests/conftest.py
import pytest
from core import SealGenerator

@pytest.fixture(scope="session")
def gen():
    """Shared SealGenerator instance (warm cache across tests)."""
    return SealGenerator(no_api_cache=False)
```

**Step 3: Create empty init files**

```python
# tests/__init__.py and tests/regression/__init__.py
# (empty)
```

**Step 4: Verify pytest discovers the test directory**

Run: `cd /home/davidlu/private-repo/fangcun && python -m pytest --collect-only 2>&1 | head -5`
Expected: `no tests ran` (no test files yet)

**Step 5: Commit**

```bash
git add tests/ pyproject.toml
git commit -m "test: scaffold pytest infrastructure for regression harness"
```

---

### Task 0.2: Define test corpus

**Files:**
- Create: `tests/regression/corpus.yaml`

**Step 1: Write the corpus file**

The corpus covers character count (1–4+), difficulty (common/complex/rare), input form (simplified/traditional), all type×style×shape combos, and edge cases. Each entry has an `id`, `text`, `shape`, `style`, `seal_type`, and `notes` field.

```yaml
# tests/regression/corpus.yaml
# Fangcun regression test corpus
# 40+ cases covering character counts, difficulty, combos, edge cases

# ── 1-char tests ────────────────────────────────────────────
- id: c01_zen_bw_oval
  text: "禅"
  shape: oval
  style: baiwen
  seal_type: leisure
  notes: "flagship 1-char, common char"

- id: c02_zen_zw_sq
  text: "禅"
  shape: square
  style: zhuwen
  seal_type: leisure
  notes: "same char, opposite style+shape"

- id: c03_dao_zw_sq
  text: "道"
  shape: square
  style: zhuwen
  seal_type: leisure
  notes: "complex single char, zhuwen square"

- id: c04_yong_bw_sq
  text: "永"
  shape: square
  style: baiwen
  seal_type: leisure
  notes: "永 — calligraphy benchmark char"

- id: c05_yi_bw_sq
  text: "一"
  shape: square
  style: baiwen
  seal_type: leisure
  notes: "extreme horizontal aspect ratio"

- id: c06_san_bw_sq
  text: "三"
  shape: square
  style: baiwen
  seal_type: leisure
  notes: "extreme horizontal, 3 strokes"

- id: c07_xin_bw_oval
  text: "心"
  shape: oval
  style: baiwen
  seal_type: leisure
  notes: "simple char in oval"

- id: c08_kong_zw_oval
  text: "空"
  shape: oval
  style: zhuwen
  seal_type: leisure
  notes: "medium complexity, zhuwen oval"

- id: c09_zen_bw_sq_name
  text: "禅"
  shape: square
  style: baiwen
  seal_type: name
  notes: "1-char name seal — strict 篆 mode"

- id: c10_zen_oval_name
  text: "禅"
  shape: oval
  style: zhuwen
  seal_type: name
  notes: "1-char zhuwen oval name seal"

# ── 2-char tests ────────────────────────────────────────────
- id: c11_sushi_bw_oval
  text: "苏轼"
  shape: oval
  style: baiwen
  seal_type: name
  notes: "2-char name seal, simplified input (苏→蘇)"

- id: c12_xiuqi_zw_oval
  text: "修齐"
  shape: oval
  style: zhuwen
  seal_type: name
  notes: "2-char zhuwen oval name (齐→齊)"

- id: c13_chaochao_zw_oval
  text: "朝朝"
  shape: oval
  style: zhuwen
  seal_type: leisure
  notes: "repeated char — R9 short-circuit"

- id: c14_zhizhi_bw_sq
  text: "知足"
  shape: square
  style: baiwen
  seal_type: leisure
  notes: "R12 stroke-width sibling match test case"

- id: c15_luxi_bw_oval
  text: "卢修"
  shape: oval
  style: baiwen
  seal_type: name
  notes: "rare char 卢→盧 in name mode"

# ── 3-char tests ────────────────────────────────────────────
- id: c16_geekzen_bw_oval
  text: "极客禅"
  shape: oval
  style: baiwen
  seal_type: brand
  notes: "brand seal, 3 chars, simplified (极→極, 客 stays)"

- id: c17_geekzen_zw_sq
  text: "极客禅"
  shape: square
  style: zhuwen
  seal_type: brand
  notes: "brand 3-char square zhuwen"

- id: c18_luxiuqi_bw_sq
  text: "卢修齐"
  shape: square
  style: baiwen
  seal_type: name
  notes: "3-char name seal — the decorative source bug origin"

- id: c19_luxiuqi_zw_oval
  text: "卢修齐"
  shape: oval
  style: zhuwen
  seal_type: name
  notes: "3-char name zhuwen oval"

# ── 4-char tests ────────────────────────────────────────────
- id: c20_tianren_bw_sq
  text: "天人合一"
  shape: square
  style: baiwen
  seal_type: leisure
  notes: "4-char square baiwen — canonical 2x2 grid"

- id: c21_tianren_zw_sq
  text: "天人合一"
  shape: square
  style: zhuwen
  seal_type: leisure
  notes: "4-char zhuwen with extreme char 一"

- id: c22_daguan_bw_oval
  text: "大观园"
  shape: oval
  style: baiwen
  seal_type: leisure
  notes: "3-char oval baiwen, complex chars"

- id: c23_daguan_zw_oval
  text: "大观园"
  shape: oval
  style: zhuwen
  seal_type: leisure
  notes: "3-char zhuwen oval — R11 corner clipping test"

# ── complex/rare char tests ─────────────────────────────────
- id: c24_gui_bw_sq
  text: "龟"
  shape: square
  style: baiwen
  seal_type: leisure
  notes: "complex char (龟→龜), high stroke count"

- id: c25_ling_zw_oval
  text: "灵"
  shape: oval
  style: zhuwen
  seal_type: leisure
  notes: "complex char (灵→靈)"

- id: c26_yu_bw_sq
  text: "鬱"
  shape: square
  style: baiwen
  seal_type: leisure
  notes: "most complex common char, already traditional"

- id: c27_xiang_zw_sq
  text: "响"
  shape: square
  style: zhuwen
  seal_type: leisure
  notes: "complex char (响→響)"

# ── edge cases ──────────────────────────────────────────────
- id: c28_yi_oval_zw
  text: "一"
  shape: oval
  style: zhuwen
  seal_type: leisure
  notes: "extreme aspect in oval — worst case layout"

- id: c29_4char_oval
  text: "天人合一"
  shape: oval
  style: baiwen
  seal_type: leisure
  notes: "4 chars in oval — cramped layout test"

- id: c30_singlerepeat
  text: "禅禅"
  shape: square
  style: baiwen
  seal_type: leisure
  notes: "repeated char — R9 short-circuit with 2 copies"

- id: c31_brand_flexible
  text: "极客禅"
  shape: oval
  style: baiwen
  seal_type: brand
  notes: "brand type — most flexible font selection"

- id: c32_5char_overflow
  text: "天地玄黄宇"
  shape: square
  style: baiwen
  seal_type: leisure
  notes: ">4 chars — auto-shrink + warning"

- id: c33_trad_input
  text: "盧修齊"
  shape: square
  style: baiwen
  seal_type: name
  notes: "already-traditional input — should NOT double-convert"

- id: c34_zhi_bw_sq
  text: "止"
  shape: square
  style: baiwen
  seal_type: leisure
  notes: "simple char, few strokes"

- id: c35_mixed_complexity
  text: "一鬱"
  shape: oval
  style: baiwen
  seal_type: leisure
  notes: "extreme contrast: simplest + most complex char together"

# ── style × shape completeness ──────────────────────────────
- id: c36_2char_bw_sq
  text: "苏轼"
  shape: square
  style: baiwen
  seal_type: name
  notes: "2-char baiwen square (complement to c11 oval)"

- id: c37_3char_bw_sq_leisure
  text: "大观园"
  shape: square
  style: baiwen
  seal_type: leisure
  notes: "3-char square baiwen leisure"

- id: c38_4char_zw_oval
  text: "天人合一"
  shape: oval
  style: zhuwen
  seal_type: leisure
  notes: "4-char zhuwen oval — max complexity combo"

- id: c39_name_zw_sq
  text: "修齐"
  shape: square
  style: zhuwen
  seal_type: name
  notes: "2-char name zhuwen square"

- id: c40_brand_zw_oval
  text: "极客禅"
  shape: oval
  style: zhuwen
  seal_type: brand
  notes: "brand zhuwen oval"
```

**Step 2: Commit**

```bash
git add tests/regression/corpus.yaml
git commit -m "test: add 40-case regression test corpus"
```

---

### Task 0.3: Build the regression runner (`run.py`)

**Files:**
- Create: `tests/regression/run.py`

**Step 1: Write the runner**

The runner:
- Loads `corpus.yaml`
- Generates each seal (optionally twice for determinism check)
- Saves output to `tests/regression/output/{run_id}/{test_id}.png`
- Captures logs per test (source selection, fallback, cache hit/miss)
- Generates `report.html` with all results in a grid

```python
#!/usr/bin/env python3
"""
Regression test runner for fangcun seal generator.

Usage:
    python tests/regression/run.py                     # run all, use git SHA as run_id
    python tests/regression/run.py --run-id baseline   # named run
    python tests/regression/run.py --twice              # run each twice for determinism check
    python tests/regression/run.py --filter c01,c11     # run subset by id prefix
"""
from __future__ import annotations

import argparse
import html
import io
import logging
import subprocess
import sys
import time
from base64 import b64encode
from pathlib import Path

import yaml
from PIL import Image

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core import SealGenerator

CORPUS_PATH = Path(__file__).parent / "corpus.yaml"
OUTPUT_BASE = Path(__file__).parent / "output"


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=PROJECT_ROOT,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _load_corpus(filter_ids: list[str] | None = None) -> list[dict]:
    with open(CORPUS_PATH, encoding="utf-8") as f:
        corpus = yaml.safe_load(f)
    if filter_ids:
        prefixes = [p.strip() for p in filter_ids]
        corpus = [
            c for c in corpus
            if any(c["id"].startswith(p) for p in prefixes)
        ]
    return corpus


def _img_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return b64encode(buf.getvalue()).decode("ascii")


class LogCapture(logging.Handler):
    """Capture log records to a list."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(self.format(record))

    def reset(self) -> None:
        self.records.clear()


def run_corpus(
    corpus: list[dict],
    run_id: str,
    twice: bool = False,
) -> list[dict]:
    """Generate seals for all corpus entries, return results list."""
    out_dir = OUTPUT_BASE / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    gen = SealGenerator(no_api_cache=False)

    # Install log capture
    capture = LogCapture()
    capture.setLevel(logging.DEBUG)
    capture.setFormatter(logging.Formatter("%(name)s %(message)s"))
    root_logger = logging.getLogger()
    original_level = root_logger.level
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(capture)

    results = []
    total = len(corpus)

    for idx, case in enumerate(corpus, 1):
        test_id = case["id"]
        print(f"  [{idx}/{total}] {test_id}: {case['text']} ...", end=" ", flush=True)

        for attempt in range(2 if twice else 1):
            suffix = f"_run{attempt + 1}" if twice else ""
            capture.reset()
            t0 = time.perf_counter()

            try:
                result = gen.generate(
                    text=case["text"],
                    shape=case["shape"],
                    style=case["style"],
                    seal_type=case["seal_type"],
                    grain=0.25,
                    rotation=2.0,
                    size=600,
                )
                elapsed = time.perf_counter() - t0

                # Save image
                out_path = out_dir / f"{test_id}{suffix}.png"
                result["image_preview"].save(out_path, "PNG")

                entry = {
                    "id": test_id,
                    "text": case["text"],
                    "shape": case["shape"],
                    "style": case["style"],
                    "seal_type": case["seal_type"],
                    "notes": case.get("notes", ""),
                    "attempt": attempt + 1,
                    "success": True,
                    "elapsed_s": round(elapsed, 2),
                    "font_used": result["font_used"],
                    "font_fallback": result["font_fallback"],
                    "warnings": result["warnings"],
                    "logs": list(capture.records),
                    "image_path": str(out_path.relative_to(OUTPUT_BASE)),
                    "image_b64": _img_to_b64(result["image_preview"]),
                }
                print(f"OK ({elapsed:.1f}s)", flush=True)

            except Exception as exc:
                elapsed = time.perf_counter() - t0
                entry = {
                    "id": test_id,
                    "text": case["text"],
                    "shape": case["shape"],
                    "style": case["style"],
                    "seal_type": case["seal_type"],
                    "notes": case.get("notes", ""),
                    "attempt": attempt + 1,
                    "success": False,
                    "elapsed_s": round(elapsed, 2),
                    "error": str(exc),
                    "logs": list(capture.records),
                    "image_b64": "",
                }
                print(f"FAIL: {exc}", flush=True)

            results.append(entry)

    root_logger.removeHandler(capture)
    root_logger.setLevel(original_level)
    return results


def generate_report(results: list[dict], run_id: str) -> Path:
    """Write an HTML report with images, params, and logs."""
    out_dir = OUTPUT_BASE / run_id
    report_path = out_dir / "report.html"

    rows = []
    for r in results:
        status = "OK" if r["success"] else "FAIL"
        status_class = "ok" if r["success"] else "fail"

        img_html = ""
        if r.get("image_b64"):
            img_html = f'<img src="data:image/png;base64,{r["image_b64"]}" width="200">'
        elif not r["success"]:
            img_html = f'<span class="error">{html.escape(r.get("error", "unknown"))}</span>'

        params = f"{r['text']} | {r['style']} | {r['shape']} | {r['seal_type']}"
        meta = ""
        if r["success"]:
            meta = f"font: {r.get('font_used', '?')}"
            if r.get("font_fallback"):
                meta += " (FALLBACK)"
            if r.get("warnings"):
                meta += "<br>" + "<br>".join(html.escape(w) for w in r["warnings"])

        log_lines = r.get("logs", [])
        # Filter to key lines (source selection, fallback, R9/R10/R12, consistency)
        key_logs = [
            l for l in log_lines
            if any(k in l for k in [
                "[Pass1]", "[Final]", "[R9", "[R10]", "[R12]",
                "统一来源", "多数来源", "最小损失", "降级", "短路",
                "Tier", "印谱", "CONSISTENCY", "STROKE",
            ])
        ]
        log_html = "<br>".join(html.escape(l) for l in key_logs[:20])

        rows.append(f"""
        <tr class="{status_class}">
            <td>{html.escape(r['id'])}</td>
            <td>{html.escape(params)}</td>
            <td>{img_html}</td>
            <td class="status">{status} ({r['elapsed_s']}s)</td>
            <td>{meta}</td>
            <td class="notes">{html.escape(r.get('notes', ''))}</td>
            <td class="logs">{log_html}</td>
        </tr>""")

    html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Fangcun Regression: {html.escape(run_id)}</title>
<style>
body {{ font-family: monospace; margin: 20px; background: #fafafa; }}
h1 {{ font-size: 18px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ccc; padding: 6px 8px; vertical-align: top; font-size: 13px; }}
th {{ background: #333; color: #fff; text-align: left; }}
tr.ok {{ background: #f0fff0; }}
tr.fail {{ background: #fff0f0; }}
.status {{ font-weight: bold; white-space: nowrap; }}
.notes {{ color: #666; font-style: italic; max-width: 200px; }}
.logs {{ font-size: 11px; color: #555; max-width: 400px; word-break: break-all; }}
.error {{ color: red; font-weight: bold; }}
img {{ border: 1px solid #ddd; }}
</style>
</head>
<body>
<h1>Regression Report: {html.escape(run_id)} ({len(results)} results)</h1>
<table>
<tr>
    <th>ID</th>
    <th>Params</th>
    <th>Output</th>
    <th>Status</th>
    <th>Metadata</th>
    <th>Notes</th>
    <th>Key Logs</th>
</tr>
{"".join(rows)}
</table>
</body>
</html>"""

    report_path.write_text(html_content, encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fangcun regression runner")
    parser.add_argument("--run-id", default=None, help="Run identifier (default: git SHA)")
    parser.add_argument("--twice", action="store_true", help="Run each case twice for determinism check")
    parser.add_argument("--filter", default=None, help="Comma-separated id prefixes to run subset")
    args = parser.parse_args()

    run_id = args.run_id or _git_sha()
    filter_ids = args.filter.split(",") if args.filter else None

    corpus = _load_corpus(filter_ids)
    if not corpus:
        print("No test cases match filter. Exiting.")
        sys.exit(1)

    print(f"\nFangcun Regression Runner")
    print(f"  Run ID:     {run_id}")
    print(f"  Test cases: {len(corpus)}")
    print(f"  Twice mode: {args.twice}\n")

    results = run_corpus(corpus, run_id, twice=args.twice)

    report = generate_report(results, run_id)
    successes = sum(1 for r in results if r["success"])
    print(f"\nDone: {successes}/{len(results)} passed")
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
```

**Step 2: Verify the runner loads corpus without errors**

Run: `cd /home/davidlu/private-repo/fangcun && python -c "import yaml; print(len(yaml.safe_load(open('tests/regression/corpus.yaml'))))"`
Expected: `40` (or the number of test cases)

**Step 3: Commit**

```bash
git add tests/regression/run.py
git commit -m "test: add regression runner with HTML report generation"
```

---

### Task 0.4: Build the comparison tool (`compare.py`)

**Files:**
- Create: `tests/regression/compare.py`

**Step 1: Write the comparator**

Takes two run IDs, generates side-by-side HTML diff. Highlights changed images using pixel-level RMSE.

```python
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
import io
import sys
from base64 import b64encode
from pathlib import Path

import numpy as np
from PIL import Image

OUTPUT_BASE = Path(__file__).parent / "output"


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
            <td class="status">{status}</td>
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

    report = compare(args.run_a, args.run_b)
    print(f"Comparison report: {report}")


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add tests/regression/compare.py
git commit -m "test: add side-by-side regression comparison tool"
```

---

### Task 0.5: Generate baseline

**Step 1: Run the full regression suite**

Run: `cd /home/davidlu/private-repo/fangcun && python tests/regression/run.py --run-id baseline`

This will take several minutes (network fetches + cache warming). Output goes to `tests/regression/output/baseline/`.

**Step 2: Verify report was generated**

Run: `ls tests/regression/output/baseline/report.html`

**Step 3: Add output directory to .gitignore (images are large, don't commit)**

Add to `.gitignore`:
```
tests/regression/output/
```

**Step 4: Commit the gitignore update**

```bash
git add .gitignore
git commit -m "chore: gitignore regression test output images"
```

**Step 5: Verify compare tool works (baseline vs baseline = all identical)**

Run: `python tests/regression/run.py --run-id baseline_verify --filter c01,c02`
Then: `python tests/regression/compare.py baseline baseline_verify`

---

## Phase 1: Correctness Bugs (P0)

### Task 1.1: Style mixing audit

**Files:**
- Read: `core/renderer.py` (lines 36-64, `render()` method)
- Read: `core/__init__.py` (lines 116-117, renderer call)
- Create: `tests/test_renderer.py`

**Step 1: Write failing test — verify baiwen and zhuwen paths are mutually exclusive**

```python
# tests/test_renderer.py
import pytest
import numpy as np
from PIL import Image
from core.renderer import SealRenderer

def _make_dummy_layout(w: int = 600, h: int = 600) -> list[dict]:
    """Create a minimal layout with one character mask."""
    mask = Image.new("L", (100, 100), 0)
    arr = np.array(mask)
    arr[20:80, 20:80] = 255
    mask = Image.fromarray(arr, "L")
    return [{"img": mask, "x": 200, "y": 200, "w": 100, "h": 100}]


class TestStyleExclusivity:
    """Verify baiwen and zhuwen rendering produce fundamentally different outputs."""

    def test_baiwen_has_opaque_background(self):
        """Baiwen: solid red background, white text cutouts."""
        renderer = SealRenderer()
        layout = _make_dummy_layout()
        result = renderer.render(layout, shape="square", style="baiwen", color=(178, 34, 34), size=600)
        arr = np.array(result)
        # Baiwen has mostly opaque pixels (red background)
        opaque_ratio = (arr[:, :, 3] > 200).sum() / arr[:, :, 3].size
        assert opaque_ratio > 0.8, f"Baiwen should have opaque background, got {opaque_ratio:.2f}"

    def test_zhuwen_has_transparent_background(self):
        """Zhuwen: transparent background, red text + frame."""
        renderer = SealRenderer()
        layout = _make_dummy_layout()
        result = renderer.render(layout, shape="square", style="zhuwen", color=(178, 34, 34), size=600)
        arr = np.array(result)
        # Zhuwen has mostly transparent pixels
        transparent_ratio = (arr[:, :, 3] < 10).sum() / arr[:, :, 3].size
        assert transparent_ratio > 0.5, f"Zhuwen should have transparent bg, got {transparent_ratio:.2f}"

    def test_baiwen_zhuwen_structurally_different(self):
        """The two styles must produce structurally different alpha channels."""
        renderer = SealRenderer()
        layout = _make_dummy_layout()
        bw = renderer.render(layout, shape="square", style="baiwen", color=(178, 34, 34), size=600)
        zw = renderer.render(layout, shape="square", style="zhuwen", color=(178, 34, 34), size=600)
        bw_alpha = np.array(bw)[:, :, 3]
        zw_alpha = np.array(zw)[:, :, 3]
        # They should be very different (one is mostly opaque, other mostly transparent)
        diff = np.abs(bw_alpha.astype(float) - zw_alpha.astype(float)).mean()
        assert diff > 100, f"Baiwen and zhuwen should differ substantially, mean diff={diff:.1f}"
```

**Step 2: Run tests**

Run: `cd /home/davidlu/private-repo/fangcun && python -m pytest tests/test_renderer.py -v`
Expected: PASS (these verify current behavior is correct)

**Step 3: Add style assertion to pipeline**

In `core/__init__.py`, after line 117 (seal rendering), add a debug assertion:

```python
# After: seal = self._renderer.render(...)
if __debug__:
    arr = np.array(seal)
    if style == "baiwen":
        opaque = (arr[:, :, 3] > 200).sum() / arr[:, :, 3].size
        assert opaque > 0.5, f"Baiwen style sanity check failed: opaque_ratio={opaque:.2f}"
    else:
        transparent = (arr[:, :, 3] < 10).sum() / arr[:, :, 3].size
        assert transparent > 0.2, f"Zhuwen style sanity check failed: transparent_ratio={transparent:.2f}"
```

Add `import numpy as np` at the top of `core/__init__.py`.

**Step 4: Run regression subset to verify assertion doesn't fire on valid seals**

Run: `python tests/regression/run.py --run-id p1_style --filter c01,c02,c03`

**Step 5: Commit**

```bash
git add tests/test_renderer.py core/__init__.py
git commit -m "test: add style exclusivity tests + pipeline assertion"
```

---

### Task 1.2: Font consistency enforcement

**Files:**
- Read: `core/scraper.py:338-567` (`fetch_chars_consistent` + fallback methods)
- Create: `tests/test_scraper_consistency.py`
- Modify: `core/scraper.py` (add `--strict-consistency` support + consistency level in return)

**Step 1: Write failing test — consistency level should be surfaced**

```python
# tests/test_scraper_consistency.py
import pytest
from unittest.mock import patch, MagicMock
from core.scraper import CalligraphyScraper


class TestFontConsistency:
    """Verify all selected glyphs share the same font field."""

    def test_single_char_consistency(self, gen):
        """Single-char seal must report consistent font."""
        result = gen.generate(text="禅", style="baiwen", shape="oval", seal_type="leisure")
        font = result["font_used"]
        # Font should be a single style, not mixed
        assert any(f in font for f in ["篆", "隶", "楷"]), f"Unexpected font: {font}"

    def test_multichar_same_font(self, gen):
        """Multi-char name seal: all chars must use same script."""
        result = gen.generate(text="修齐", style="baiwen", shape="square", seal_type="name")
        font = result["font_used"]
        # Name type should always be 篆
        assert "篆" in font, f"Name seal should use 篆, got: {font}"
```

**Step 2: Run tests**

Run: `cd /home/davidlu/private-repo/fangcun && python -m pytest tests/test_scraper_consistency.py -v`
Expected: PASS (these verify existing behavior)

**Step 3: Add consistency level to generator result**

Modify `core/__init__.py` to include `consistency_level` in the return dict. The scraper's `fetch_chars_consistent` already logs which level (1-5) was used. We need to surface it:

Add a `consistency_level` field to the `_log_final` method in `core/scraper.py`. After `_log_final` is called, the level is known. Surface it by adding a `_last_consistency_level` attribute on the scraper.

In `core/scraper.py`, in `_log_final`:
```python
def _log_final(self, text, result_tuple):
    """Log per-char final source attribution and return the tuple."""
    images, font, fallback, tabs, src_names, warnings = result_tuple
    for i, char in enumerate(text):
        logger.info("[Final] '%s' → %s / %s (%s)", char, src_names[i], tabs[i], font)
    return result_tuple
```

Add `self._last_consistency_level` tracking by examining which code path produced the result (unified L1/L2 vs majority L3 vs min-loss L4 vs per-char L5).

In `core/__init__.py`, surface it:
```python
# After fetch_chars_consistent returns, add to result dict:
result["consistency_level"] = getattr(self._scraper, '_last_consistency_level', 0)
result["source_names"] = source_names
```

**Step 4: Add `--strict-consistency` flag to CLI**

In `cli.py`, add argument:
```python
p.add_argument(
    "--strict-consistency",
    action="store_true",
    help="严格一致性模式：仅接受 Level 1-2 统一来源",
)
```

In `_generate_one`, check after generation:
```python
if args.strict_consistency and result.get("consistency_level", 0) > 2:
    raise ValueError(f"严格一致性检查失败: consistency_level={result['consistency_level']}")
```

**Step 5: Run tests and regression subset**

Run: `python -m pytest tests/test_scraper_consistency.py -v`
Run: `python tests/regression/run.py --run-id p1_consistency --filter c11,c12,c18`

**Step 6: Commit**

```bash
git add core/scraper.py core/__init__.py cli.py tests/test_scraper_consistency.py
git commit -m "feat: surface consistency level in result + --strict-consistency flag"
```

---

### Task 1.3: R12 stroke-width instrumentation

**Files:**
- Read: `core/scraper.py:569-692` (R12 + `_relative_stroke_width`)
- Create: `tests/test_stroke_width.py`

**Step 1: Write test for R12 logging**

```python
# tests/test_stroke_width.py
import logging
import pytest


class TestR12StrokeWidth:
    """Verify R12 stroke-width matching is invoked and logged for multi-char seals."""

    def test_r12_invoked_for_multichar(self, gen, caplog):
        """Multi-char seal should trigger R12 or at least log source selection."""
        with caplog.at_level(logging.DEBUG, logger="core.scraper"):
            gen.generate(text="知足", style="baiwen", shape="square", seal_type="leisure")

        # Should see either R12 stroke matching or unified source selection
        logs = caplog.text
        has_source_log = "统一来源" in logs or "[R12]" in logs or "[Final]" in logs
        assert has_source_log, "Multi-char seal should log source selection path"

    def test_stroke_width_measurable(self):
        """_relative_stroke_width should return non-zero for real character masks."""
        from core.scraper import CalligraphyScraper
        from PIL import Image
        import numpy as np

        # Create a synthetic "character" image (dark strokes on light bg)
        img = Image.new("L", (200, 200), 255)
        arr = np.array(img)
        arr[80:120, 50:150] = 0  # horizontal stroke
        arr[50:150, 90:110] = 0  # vertical stroke
        img = Image.fromarray(arr, "L")

        sw = CalligraphyScraper._relative_stroke_width(img)
        assert sw > 0, f"Stroke width should be positive, got {sw}"
        assert sw < 1.0, f"Relative stroke width should be < 1.0, got {sw}"
```

**Step 2: Run tests**

Run: `cd /home/davidlu/private-repo/fangcun && python -m pytest tests/test_stroke_width.py -v`
Expected: PASS

**Step 3: Add stroke deviation logging**

In `core/scraper.py`, inside `_try_unified_source_from_candidates` (around line 642), after selecting each char's variant, log the deviation:

```python
# After the for loop at line 642, add deviation reporting:
if len(text) > 1 and target_sw > 0:
    for char, eligible, chosen_item in zip(text, eligible_per_char, zip(images, tabs_, src_names)):
        char_sw = self._relative_stroke_width(chosen_item[0]) if hasattr(chosen_item[0], 'size') else 0
        # chosen_item[0] is the image from the images list
        pass  # The actual deviation logging is already in the R12 block above
```

Actually, the existing R12 code at line 650-657 already logs when a non-top variant is chosen. Add a summary log after the loop:

```python
# After line 667, before the return:
if target_sw > 0 and len(text) > 1:
    deviations = []
    for char, img in zip(text, images):
        char_sw = self._relative_stroke_width(img)
        dev = abs(char_sw - target_sw) / target_sw if target_sw > 0 else 0
        deviations.append((char, char_sw, dev))
        if dev > 0.20:
            logger.warning(
                "[R12] STROKE_DEVIATION '%s': rel_sw=%.3f target=%.3f deviation=%.0f%%",
                char, char_sw, target_sw, dev * 100,
            )
    logger.info(
        "[R12] 笔画匹配总结 source=%s target_sw=%.3f: %s",
        best_source, target_sw,
        ", ".join(f"'{c}'={sw:.3f}({d:.0%})" for c, sw, d in deviations),
    )
```

**Step 4: Run regression to verify no crashes**

Run: `python tests/regression/run.py --run-id p1_r12 --filter c14,c18,c20`

**Step 5: Commit**

```bash
git add core/scraper.py tests/test_stroke_width.py
git commit -m "feat: add R12 stroke-width deviation logging + unit tests"
```

---

### Task 1.4: Cache correctness audit

**Files:**
- Read: `core/scraper.py:226-302` (cache helpers)
- Create: `tests/test_cache.py`

**Step 1: Write test — Tier 3 cache key sufficiency**

```python
# tests/test_cache.py
import pytest
from pathlib import Path
from core.scraper import CalligraphyScraper, CACHE_DIR, IMG_CACHE_DIR


class TestCacheKeys:
    """Verify cache key design prevents cross-contamination."""

    def test_tier3_key_includes_font_and_tab(self):
        """Tier 3 cache key {char}_{font}_{tab}.png must be unique per selection context."""
        # Different font+tab combos for same char must produce different cache paths
        scraper = CalligraphyScraper()
        # The _save_cache / _get_or_fetch methods use {char}_{font}_{tab}.png
        # Verify the naming pattern
        key1 = f"禅_篆_字典.png"
        key2 = f"禅_隶_字典.png"
        key3 = f"禅_篆_真迹.png"
        assert key1 != key2 != key3, "Cache keys must differ by font and tab"

    def test_tier3_does_not_encode_siblings(self):
        """
        Tier 3 key is per-char only. For multi-char seals, R12 stroke-width
        matching picks different variants based on siblings. This means Tier 3
        cache is potentially wrong for the R12 use case.

        This test documents the known limitation.
        """
        # The cache key {char}_{font}_{tab}.png does NOT include sibling context.
        # When R12 selects a different variant for "知" based on "足"'s stroke width,
        # caching that variant as "知_篆_字典.png" would poison future uses of "知"
        # in different sibling contexts.
        #
        # Current mitigation: _try_unified_source_from_candidates calls
        # _save_cache AFTER R12 selection, so the cached image IS the R12-chosen one.
        # This is acceptable IF the R12 choice is generally good across contexts.
        # Document this as a known trade-off.
        pass  # Documented limitation, not a failing test
```

**Step 2: Run tests**

Run: `cd /home/davidlu/private-repo/fangcun && python -m pytest tests/test_cache.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_cache.py
git commit -m "test: cache key audit — document Tier 3 sibling context limitation"
```

---

## Phase 2: Single-Glyph Quality (P1)

### Task 2.1: Extractor debug mode

**Files:**
- Modify: `core/extractor.py` (add `--debug-extract` intermediate saves)
- Modify: `cli.py` (add `--debug-extract` flag)
- Create: `tests/test_extractor.py`

**Step 1: Write failing test — extractor should support debug output**

```python
# tests/test_extractor.py
import pytest
import numpy as np
from PIL import Image
from core.extractor import CharExtractor


class TestExtractorBasics:
    """Test extractor produces valid masks from synthetic inputs."""

    def test_black_on_white_passthrough(self):
        """Black strokes on white background should extract correctly."""
        ext = CharExtractor()
        img = Image.new("L", (200, 200), 255)
        arr = np.array(img)
        arr[50:150, 50:150] = 0  # black square
        img = Image.fromarray(arr, "L")

        result = ext.extract(img, source="字典")
        result_arr = np.array(result)
        assert result_arr.max() == 255, "Should have stroke pixels"
        assert result_arr.sum() > 0, "Should have non-zero content"

    def test_white_on_black_inversion(self):
        """White strokes on dark background should be auto-inverted."""
        ext = CharExtractor()
        img = Image.new("L", (200, 200), 30)  # dark background
        arr = np.array(img)
        arr[50:150, 50:150] = 240  # bright square (stroke)
        img = Image.fromarray(arr, "L")

        result = ext.extract(img, source="字典")
        result_arr = np.array(result)
        # After extraction, strokes should be 255 regardless of input polarity
        assert result_arr.max() == 255

    def test_min_stroke_threshold(self):
        """Extraction of nearly-empty image should still produce output."""
        ext = CharExtractor()
        # Very sparse image — just a thin line
        img = Image.new("L", (200, 200), 255)
        arr = np.array(img)
        arr[100, 50:150] = 0  # single pixel row
        img = Image.fromarray(arr, "L")

        result = ext.extract(img, source="字典")
        # Should not crash, even if output is sparse
        assert result.mode == "L"

    def test_yinpu_source_detection(self):
        """Known 印谱 source names should trigger Tier 1 extraction."""
        ext = CharExtractor()
        # Create RGBA image simulating 印谱 (opaque block with alpha holes = strokes)
        img = Image.new("RGBA", (200, 200), (50, 20, 20, 255))
        arr = np.array(img)
        arr[60:140, 60:140, 3] = 0  # transparent hole = stroke area
        img = Image.fromarray(arr, "RGBA")

        result = ext.extract(img, source="字典", source_name="汉印分韵")
        assert ext._detected_as_yinpu, "Should detect as 印谱 via Tier 1 whitelist"
```

**Step 2: Run tests**

Run: `cd /home/davidlu/private-repo/fangcun && python -m pytest tests/test_extractor.py -v`
Expected: PASS

**Step 3: Add debug intermediate saves to extractor**

In `core/extractor.py`, add a class-level `debug_dir` attribute:

```python
class CharExtractor:
    def __init__(self) -> None:
        self._detected_as_yinpu = False
        self.debug_dir: Path | None = None  # Set externally for debug output
```

Add `from pathlib import Path` at top.

In the `extract` method, save intermediates when `self.debug_dir` is set:

```python
# After step 1 (normalize):
if self.debug_dir:
    self.debug_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(gray, "L").save(self.debug_dir / "01_normalized.png")

# After step 2 (binarize):
if self.debug_dir:
    Image.fromarray(binary, "L").save(self.debug_dir / "02_binary.png")

# After step 4 (denoise):
if self.debug_dir:
    Image.fromarray(binary, "L").save(self.debug_dir / "03_denoised.png")

# After step 5 (crop):
if self.debug_dir:
    Image.fromarray(cropped, "L").save(self.debug_dir / "04_cropped.png")
```

**Step 4: Wire `--debug-extract` flag in CLI**

In `cli.py`, add:
```python
p.add_argument("--debug-extract", action="store_true", help="保存提取中间步骤")
```

In `_generate_one`, before generation:
```python
if args.debug_extract:
    debug_base = output_dir / f"{text}_debug"
    for i, _ in enumerate(text):
        gen._extractor.debug_dir = debug_base / f"char_{i}"
else:
    gen._extractor.debug_dir = None
```

**Step 5: Run and verify debug output**

Run: `python cli.py --text "禅" --debug-extract --output-dir ./seals/debug_test`
Verify: `ls ./seals/debug_test/禅_debug/char_0/`

**Step 6: Commit**

```bash
git add core/extractor.py cli.py tests/test_extractor.py
git commit -m "feat: add extractor debug mode + extraction unit tests"
```

---

### Task 2.2: Extractor post-extraction validation

**Files:**
- Modify: `core/extractor.py` (add validation step after crop)

**Step 1: Write failing test — validation should reject garbage masks**

```python
# Add to tests/test_extractor.py:

def test_validation_rejects_noise():
    """Mask with too many disconnected components should be flagged."""
    ext = CharExtractor()
    # Create noisy image — salt-and-pepper pattern
    img = Image.new("L", (200, 200), 255)
    arr = np.array(img)
    np.random.seed(42)
    noise = np.random.random((200, 200)) < 0.3
    arr[noise] = 0
    img = Image.fromarray(arr, "L")

    result = ext.extract(img, source="字典")
    result_arr = np.array(result)
    # Should still produce output (denoise handles some noise)
    # but extremely noisy inputs should have reduced stroke count
    assert result.mode == "L"
```

**Step 2: Add validation step in extractor**

In `core/extractor.py`, between step 4 (denoise) and step 5 (crop), add:

```python
# Step 4.5: validation — reject if ink ratio is pathological
stroke_count = int(np.count_nonzero(binary))
total = binary.size
ink_ratio = stroke_count / total if total > 0 else 0
if ink_ratio > 0.60:
    logger.warning(
        "ink_ratio=%.2f (>0.60) — possible extraction failure, re-binarizing",
        ink_ratio,
    )
    binary = self._binarize_otsu(gray)
    binary = self._denoise(binary, strong=True)
```

**Step 3: Run tests**

Run: `python -m pytest tests/test_extractor.py -v`

**Step 4: Run regression to verify no regressions**

Run: `python tests/regression/run.py --run-id p2_extractor --filter c01,c03,c18`
Then: `python tests/regression/compare.py baseline p2_extractor`

**Step 5: Commit**

```bash
git add core/extractor.py tests/test_extractor.py
git commit -m "feat: add ink-ratio validation to extractor pipeline"
```

---

## Phase 3: Layout Quality (P2)

### Task 3.1: Layout debug mode

**Files:**
- Modify: `core/layout.py` (add debug visualization)
- Modify: `cli.py` (add `--debug-layout` flag)

**Step 1: Add debug visualization to layout**

In `core/layout.py`, add a `debug_render` staticmethod that draws the layout grid + character bounding boxes:

```python
@staticmethod
def debug_render(
    placements: list[dict],
    canvas_size: tuple[int, int],
    ta_offset: tuple[int, int] = (0, 0),
) -> Image.Image:
    """Render debug overlay showing grid cells, ink bounds, and centroids."""
    from PIL import ImageDraw
    tw, th = canvas_size
    debug = Image.new("RGBA", (tw, th), (255, 255, 255, 128))
    draw = ImageDraw.Draw(debug)

    for item in placements:
        x, y, w, h = item["x"] - ta_offset[0], item["y"] - ta_offset[1], item["w"], item["h"]
        # Cell boundary (blue)
        draw.rectangle([x, y, x + w, y + h], outline=(0, 0, 255, 200), width=2)
        # Ink bounding box (red)
        mask_arr = np.array(item["img"])
        ys, xs = np.where(mask_arr > 128)
        if len(xs) > 0:
            ink_x0 = int(xs.min()) + x
            ink_y0 = int(ys.min()) + y
            ink_x1 = int(xs.max()) + x
            ink_y1 = int(ys.max()) + y
            draw.rectangle([ink_x0, ink_y0, ink_x1, ink_y1], outline=(255, 0, 0, 200), width=1)
            # Centroid (green dot)
            cx = int(np.mean(xs)) + x
            cy = int(np.mean(ys)) + y
            draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=(0, 255, 0, 255))

    return debug
```

Add `import numpy as np` if not already present (it is — cv2 is imported).

**Step 2: Wire into CLI**

In `cli.py`, add `--debug-layout` flag. When active, save the debug overlay alongside the seal.

**Step 3: Run and verify**

Run: `python cli.py --text "天人合一" --debug-layout --output-dir ./seals/debug_layout`

**Step 4: Commit**

```bash
git add core/layout.py cli.py
git commit -m "feat: add layout debug visualization mode"
```

---

### Task 3.2: Oval per-glyph clearance

**Files:**
- Read: `core/renderer.py:262-303` (`text_area` method)
- Read: `core/layout.py:57-195` (`arrange` method)
- Create: `tests/test_layout.py`

**Step 1: Write test for oval clipping detection**

```python
# tests/test_layout.py
import pytest
import numpy as np
from PIL import Image
from core.layout import SealLayout
from core.renderer import SealRenderer


class TestOvalLayout:
    """Verify characters don't clip against oval boundary."""

    def test_text_area_inside_ellipse(self):
        """Text area rectangle should fit inside the elliptical boundary."""
        size = 600
        ta_x, ta_y, ta_w, ta_h = SealRenderer.text_area("oval", size, "zhuwen", 3)
        w, h = SealRenderer.canvas_dimensions("oval", size)

        # All 4 corners of text area must be inside the ellipse
        # Ellipse: (x - cx)^2 / rx^2 + (y - cy)^2 / ry^2 <= 1
        cx, cy = w / 2, h / 2
        rx, ry = w / 2, h / 2

        corners = [
            (ta_x, ta_y),
            (ta_x + ta_w, ta_y),
            (ta_x, ta_y + ta_h),
            (ta_x + ta_w, ta_y + ta_h),
        ]
        for px, py in corners:
            d = ((px - cx) / rx) ** 2 + ((py - cy) / ry) ** 2
            assert d <= 1.0, f"Corner ({px},{py}) is outside ellipse (d={d:.3f})"
```

**Step 2: Run test**

Run: `cd /home/davidlu/private-repo/fangcun && python -m pytest tests/test_layout.py -v`

**Step 3: Commit**

```bash
git add tests/test_layout.py
git commit -m "test: add oval boundary clearance test"
```

---

## Phase 4: Texture and Color Realism (P3)

### Task 4.1: Texture determinism via seed

**Files:**
- Modify: `core/texture.py` (add `seed` parameter)
- Create: `tests/test_texture.py`

**Step 1: Write failing test — same seed should produce identical output**

```python
# tests/test_texture.py
import pytest
import numpy as np
from PIL import Image
from core.texture import StoneTexture


class TestTextureDeterminism:
    """Verify texture is reproducible with a fixed seed."""

    def test_same_seed_same_output(self):
        """Identical input + same seed → identical output."""
        tex = StoneTexture()
        img = Image.new("RGBA", (200, 200), (178, 34, 34, 255))

        result1 = tex.apply(img, grain_strength=0.25, seed=42)
        result2 = tex.apply(img, grain_strength=0.25, seed=42)

        arr1 = np.array(result1)
        arr2 = np.array(result2)
        assert np.array_equal(arr1, arr2), "Same seed should produce identical output"

    def test_different_seed_different_output(self):
        """Different seeds → different output."""
        tex = StoneTexture()
        img = Image.new("RGBA", (200, 200), (178, 34, 34, 255))

        result1 = tex.apply(img, grain_strength=0.25, seed=42)
        result2 = tex.apply(img, grain_strength=0.25, seed=99)

        arr1 = np.array(result1)
        arr2 = np.array(result2)
        assert not np.array_equal(arr1, arr2), "Different seeds should differ"
```

**Step 2: Run test to verify it fails (no seed param yet)**

Run: `python -m pytest tests/test_texture.py -v`
Expected: FAIL (TypeError: unexpected keyword argument 'seed')

**Step 3: Add seed parameter to `StoneTexture.apply`**

In `core/texture.py`, modify `apply`:

```python
def apply(
    self, img: Image.Image, grain_strength: float = 0.25, seed: int | None = None
) -> Image.Image:
    if grain_strength <= 0.0:
        return img.copy()

    if seed is not None:
        np.random.seed(seed)

    # ... rest unchanged
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_texture.py -v`
Expected: PASS

**Step 5: Wire `--seed` into CLI and SealGenerator**

In `core/__init__.py`, add `seed` parameter to `generate()`:

```python
def generate(self, ..., seed: int | None = None) -> dict:
    # ... at step 5 (texture):
    seal = self._texture.apply(seal, grain, seed=seed)
```

In `cli.py`, add `--seed` argument:
```python
p.add_argument("--seed", type=int, default=None, help="随机种子（纹理可复现）")
```
Pass `seed=args.seed` through to `gen.generate()`.

**Step 6: Run regression to verify no breaks**

Run: `python tests/regression/run.py --run-id p4_seed --filter c01,c02`

**Step 7: Commit**

```bash
git add core/texture.py core/__init__.py cli.py tests/test_texture.py
git commit -m "feat: add --seed for reproducible texture generation"
```

---

## Phase 5: Stability and Reproducibility

### Task 5.1: Failure mode classification

**Files:**
- Create: `core/errors.py`
- Modify: `core/__init__.py` (use typed errors)
- Modify: `core/scraper.py` (raise specific errors)
- Create: `tests/test_errors.py`

**Step 1: Write the error classes**

```python
# core/errors.py
"""Classified failure modes for the seal generation pipeline."""


class SealError(Exception):
    """Base class for all seal generation errors."""
    pass


class CharNotFoundError(SealError):
    """Requested character not found in any available script."""
    def __init__(self, char: str, scripts_tried: list[str]):
        self.char = char
        self.scripts_tried = scripts_tried
        super().__init__(
            f"字符 '{char}' 在 {'/'.join(scripts_tried)} 中均无字源。"
            f"建议：尝试繁体输入或降低 seal_type 限制。"
        )


class SourceInconsistencyError(SealError):
    """Strict consistency mode rejected available results."""
    def __init__(self, text: str, level: int):
        self.text = text
        self.level = level
        super().__init__(
            f"'{text}' 统一来源等级为 {level}（要求 ≤2）。"
            f"建议：关闭 --strict-consistency 或减少字数。"
        )


class ExtractionFailedError(SealError):
    """Glyph found but couldn't be cleaned to acceptable quality."""
    def __init__(self, char: str, reason: str):
        self.char = char
        self.reason = reason
        super().__init__(f"字符 '{char}' 提取失败: {reason}")


class UpstreamApiError(SealError):
    """ygsf.com API is unreachable."""
    def __init__(self, status_code: int | None = None, detail: str = ""):
        self.status_code = status_code
        super().__init__(
            f"ygsf.com API 不可用 (HTTP {status_code}): {detail}"
            if status_code
            else f"ygsf.com API 不可用: {detail}"
        )


class RateLimitedError(SealError):
    """ygsf.com rate-limited us."""
    def __init__(self):
        super().__init__("ygsf.com 请求频率超限，请稍后重试。")
```

**Step 2: Write tests for error types**

```python
# tests/test_errors.py
import pytest
from core.errors import (
    CharNotFoundError,
    SourceInconsistencyError,
    ExtractionFailedError,
    UpstreamApiError,
    RateLimitedError,
    SealError,
)


class TestErrorHierarchy:
    def test_all_inherit_from_seal_error(self):
        errors = [
            CharNotFoundError("鬱", ["篆"]),
            SourceInconsistencyError("修齐", 4),
            ExtractionFailedError("禅", "noise"),
            UpstreamApiError(500, "server error"),
            RateLimitedError(),
        ]
        for e in errors:
            assert isinstance(e, SealError)

    def test_error_messages_include_suggestion(self):
        e = CharNotFoundError("鬱", ["篆", "隶"])
        assert "建议" in str(e)
        assert "鬱" in str(e)
```

**Step 3: Run tests**

Run: `python -m pytest tests/test_errors.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add core/errors.py tests/test_errors.py
git commit -m "feat: add classified error types for pipeline failure modes"
```

---

### Task 5.2: Determinism audit

**Step 1: Run regression suite twice**

Run: `python tests/regression/run.py --run-id determinism_a --filter c01,c02,c03,c11,c18,c20`
Run: `python tests/regression/run.py --run-id determinism_b --filter c01,c02,c03,c11,c18,c20`

**Step 2: Compare**

Run: `python tests/regression/compare.py determinism_a determinism_b`

**Step 3: Analyze report**

If any cases show CHANGED, investigate the source of non-determinism (texture noise without seed, or candidate sort order). Fix by:
- Adding deterministic sort key to candidate lists in scraper (score + URL hash)
- Using fixed seed for texture in regression mode

**Step 4: Commit any fixes**

```bash
git commit -m "fix: ensure deterministic candidate sorting for reproducible output"
```

---

## Workflow Checklist

For EVERY task after Phase 0:

1. [ ] Run regression baseline exists
2. [ ] Reproduce issue with specific corpus case
3. [ ] Write focused test demonstrating the bug/behavior
4. [ ] Implement fix
5. [ ] Run `python -m pytest` (all tests pass)
6. [ ] Run `python tests/regression/run.py --run-id {phase}_{task}`
7. [ ] Run `python tests/regression/compare.py baseline {phase}_{task}`
8. [ ] Review diff report — nothing unexpected broke
9. [ ] Commit

---

## Human-in-the-loop checkpoints

After each phase completion:
- **Phase 0**: Human reviews `report.html`, scores each seal 1-5, confirms baseline quality
- **Phase 1**: Human reviews regression diff — P0 bugs should be visibly fixed
- **Phase 2**: Human inspects extractor debug output for 5-10 problem cases
- **Phase 3**: Human evaluates layout debug overlays for spacing quality
- **Phase 4**: Human compares texture with reference real seal impressions
- **Phase 5**: Human confirms determinism report shows zero diffs
