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
