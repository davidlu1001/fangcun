# 方寸 · 极客禅印章生成器

Fangcun — Chinese Seal Generator for Geek-Zen. 数字化传统金石艺术，代码刻出的方寸之美。

## Architecture

Modular `core/` package with zero UI dependencies. Two thin entry points share the same pipeline.

```
core/
├── __init__.py      # SealGenerator — unified pipeline orchestrator
├── scraper.py       # ygsf.com API + 5-level source unification + scoring + 3-tier cache
├── extractor.py     # Three-tier polarity normalization + two-phase 印谱 extraction
├── layout.py        # Multi-phase layout: fit → balance → extreme-detect → reverse-construct → stroke-normalize → centroid-place
├── renderer.py      # Baiwen mask paste / zhuwen max-alpha merge, shape-aware text_scale
└── texture.py       # 6-layer texture: pressure, roughness, chipping, grain, ink pooling, color drift
app.py               # Gradio Web UI (主入口)
cli.py               # CLI batch entry with rich progress + cache management + --debug (次入口)
core/errors.py       # Typed pipeline failure modes (SealError + 5 subclasses)
tests/               # 63 unit + regression tests (pytest)
tests/regression/    # 40-case visual regression harness + runner + compare tool
docs/plans/          # Implementation plans + audit reports
```

## Scraper: source intelligence

### 5-level source selection (同源同体)

```
Level 1: 统一来源 n=5    all chars from one source (intersection)
Level 2: 统一来源 n=10   wider candidate pool retry
Level 3: 多数来源 >50%   majority vote, fallback for uncovered chars
Level 4: 最小损失来源     coverage×100 − score_loss ranking
Level 5: 各字独立最优     per-char best (last resort)
```

**Deferred state machine**: 隶書同源 > 篆書異源. Save fallback and continue to next font.

### Image scoring (0–100)

Resolution (0–30) + contrast (0–50) + coverage (0–20), then penalties:
- Structural integrity (R8): CCA max_ratio < 0.40 → -35, < 0.60 → -10
- `DECORATIVE_SOURCES` (鸟虫篆): name=-40 hard-filter, leisure=-40 penalty
- `_INFERIOR_STYLE_SOURCES` (金文/简帛/楚简/说文/千字文 etc.): -25
- 印谱 base: -10; fragmentation hard-reject (light_ratio > 0.15)

### Seal type semantics (金石学 DSL)

- `name` (名章): strict 篆 only, never degrades. Decorative sources hard-filtered. Traditional-first strict break.
- `leisure` (闲章): 篆→隶→楷, deferred state machine.
- `brand` (品牌章): 篆→隶→楷, most flexible.

### Traditional-first fetch (R8-B)

篆書 predates simplified/traditional split by ~2000 years. Try 齊 before 齐, 盧 before 卢.
- name: strict break after traditional candidates found
- leisure/brand: both forms queried, merged pool

### Single/repeated char short-circuit (R9+R10)

When `len(set(text)) == 1` (禅, 朝朝), skip Pass 2 unified source check entirely. Use `_fetch_all_candidates(n=15)` with preferred source tie-break: within ±5 of top score, prefer `PREFERRED_INSCRIPTION_SOURCES` (中国篆刻大字典, 汉印文字征 series, etc.) over calligrapher personal dictionaries.

### Stroke-width sibling match (R12)

When `_try_unified_source_from_candidates` picks a unified source that has multiple variants for one char (e.g. 中国篆刻大字典 ships both a thin and a thick 知), the default "first-per-source" rule can pair a thin 知 with a thick 足 and produce visual imbalance. R12 fixes this: after `best_source` is chosen, collect the relative stroke width (`p70 × 2 / min_dim`) of each **unambiguous** sibling (only one eligible variant within ±5 score of its top) as the anchor. For each ambiguous char, pick the variant whose relative stroke width is closest to the anchor median. Falls back to median of top variants if every sibling is ambiguous. Logged as `[R12] 笔画匹配`.

**R12 anchor extreme-aspect filter**: chars with aspect > 2.5 or < 0.4 (一, 三, etc.) are excluded from the anchor pool. Their `rel_sw ≈ 1.0` — the entire char IS the stroke — would poison the median target. Three-tier fallback: unambiguous + non-extreme → all + non-extreme → all (handles `text="一"` edge case). Constants: `_R12_EXTREME_ASPECT_HI = 2.5`, `_R12_EXTREME_ASPECT_LO = 0.4`. Deviation above `R12_STROKE_DEVIATION_WARN = 0.20` logs WARNING; per-char summary always logged at INFO via `[R12] 笔画匹配总结`.

### Consistency level (1-5)

`SealGenerator.generate()` returns `consistency_level` indicating which fallback path produced the source selection:
- L1: unified source (n=5) or single/repeated-char short-circuit
- L2: unified source (n=10 wider pool)
- L3: majority source (>50% coverage)
- L4: minimum-loss source
- L5: per-char internal assembly (`_force_assemble_single_font` for name) or Pass 2 multi-font assembly

Reset to 0 at the top of `fetch_chars_consistent` so an exception mid-call doesn't surface stale state. CLI flag `--strict-consistency` raises `SourceInconsistencyError` if level > 2.

### Three-tier cache

- Tier 1: API response JSON (`_api/`) — MD5 key, 30d/7d TTL
- Tier 2: Image CDN (`_img/`) — MD5 of URL
- Tier 3: Selected-best (`{char}_{font}_{tab}.png`)
- CLI: `--cache-info`, `--clear-cache`, `--no-api-cache`, `--debug`

### Other scraper decisions

- Tab priority: 字典 (type=3) → 真迹 (type=2), never 字库 (type=1)
- Font consistency: all chars share same script
- Atomic cache writes (tempfile + os.replace)
- Log: `[Pass1]` for scan results, `[Final]` for per-char source attribution

## Extractor: Three-tier polarity + two-phase 印谱 extraction

**Three-tier polarity detection** (`_normalize_to_black_on_white`):
- Tier 1: `KNOWN_YINPU_SOURCES` whitelist → two-phase alpha extraction
- Tier 2: Alpha semantic detection (bright pixel ratio in opaque region)
- Tier 3: Morphological erosion fallback

**Two-phase `_extract_yinpu_strokes`**: per-chunk Alpha CCA → edge inset per chunk. Mid-60% sampling, consecutive-3-row scan, MIN_INSET floor, 2px safety buffer.

**Binarization**: Otsu for 字典/本地; bilateral filter + adaptive for 真迹.

**Ink-ratio validation** (post-denoise): if `ink_ratio > _INK_RATIO_MAX = 0.60`, re-binarize with strong Otsu + denoise. Real seal characters virtually never exceed ~50% ink coverage; this catches binarization failure on noisy/low-contrast scans. Dormant on baseline inputs (defense-in-depth).

**Debug mode**: set `CharExtractor.debug_dir = path` to save per-stage PNGs (`01_normalized.png`, `02_binary.png`, `03_denoised.png`, `04_cropped.png`). For multi-char seals, `generate()` nests per-char subdirs: `{path}/00_{char0}/01_normalized.png`, `{path}/01_{char1}/...`, etc. — same index allows distinguishing duplicate chars (e.g. `朝朝` → `00_朝/`, `01_朝/`). Use `SealGenerator.set_extract_debug_dir(path)` instead of touching the private attribute directly.

## Layout: Multi-phase pipeline

`SealLayout.arrange()` runs six phases:

1. **Normal fit**: Equal-ratio scaling + conditional tall/wide cell stretch (max 1.25x)
1.5. **2-char balance**: Equalize fill ratio when gap > 10% (max 1.25x scale-up)
2. **Extreme detection**: Identify aspect > 2.5 (一二三) or < 0.4, compute sibling median stroke width
3. **Reverse construction**: Extreme chars → resize stroke to target width, center in padded canvas (ink_ratio ≤ 55%)
4. **Stroke-width normalize**: Distance-transform equalization (dilate < 0.75x, erode > 1.35x)
5. **Centroid placement**: Pixel-weighted centroid compensation (0.65 coefficient)

**Dynamic margins**: `MARGIN_TABLE` by (style, shape).

**Zhuwen bleed**: 4% bleed with `margin=0` lets chars cross frame lines.

**Debug visualization**: `SealLayout.debug_render(placements, canvas_size)` returns an RGBA overlay with blue cell rectangles, red ink bounding boxes, green pixel-weighted centroids. Use `SealGenerator.render_layout_debug(text, ...)` instead of touching the private attribute directly.

## Renderer: text_scale by context

| Style + Shape | text_scale | Rationale |
|---|---|---|
| zhuwen square | 0.98 | Near frame for bleed (冲刀破边) |
| zhuwen oval | 0.88 | Oval corners clip — must keep text inside ellipse (R11) |
| baiwen, 1 char | 0.93 | 顶天立地 fill |
| baiwen, 2+ chars | 0.86 | Leave red border for visual weight |

Zhuwen uses max-alpha merge (`np.maximum` of char + frame layers). Baiwen uses mask paste (white cutout on red).

## Texture: style-aware pipeline

`StoneTexture.apply()` accepts `style` so baiwen (opaque bg + PAPER_COLOR cutouts) and zhuwen (transparent bg + red ink) don't share bugs.

**ink_mask construction**: baiwen excludes pixels near `PAPER_COLOR` (RGB L2 distance < 30) so grain/drift/brightness modulation never leaks red noise onto white text cutouts. Zhuwen uses `alpha > 128` — any visible pixel is ink.

**Pressure path**: baiwen uses `_pressure_variation_rgb` (brightness on ink only, keeps alpha=255 — real paper impressions never turn semi-transparent). Zhuwen keeps alpha-based `_pressure_variation` so relief strokes can fade.

**Intersection darkening cap**: `distanceTransform` on a mostly-opaque baiwen mask yields `d_max ≈ 200`, so the old `dist/d_max > 0.3` check darkened the entire image center. `_stroke_intersection_darkening` now caps influence distance at `max(w, h) * 0.03` (≈18px @ 600px) — matches real ink-pooling radius.

**Local RNG**: every layer receives an `rng: np.random.Generator` argument built once in `apply()` (`default_rng(seed)` or unseeded). No more `np.random.seed()` mutation of global state — concurrent callers with different seeds don't interfere. **This changes the texture byte-stream for any given seed** vs pre-1.3.0 output; regenerate regression baselines after upgrade.

## Data Flow

```
SealGenerator.generate()
  → scraper.fetch_chars_consistent()
    → [single-char short-circuit with preferred source tie-break]
    → _query_glyph_list()                  # API cache → web
    → _fetch_all_candidates(n=5/10)       # image CDN cache → download
    → _try_unified_source_from_candidates # Level 1-2 + R12 stroke-width match
    → _majority_source_fallback           # Level 3
    → _min_style_loss_fallback            # Level 4
  → extractor.extract(img, tab, source_name)
  → layout.arrange(style, shape)            # 6-phase pipeline
  → renderer.render()                     # text_scale per context
  → texture.apply(style=style)            # style-aware 6-layer stone effects
  → rotate + preview
```

## Running

```bash
python app.py                   # Web UI → http://localhost:7860
python cli.py --text "禅" --shape oval --style baiwen --type leisure
python cli.py --batch chars.txt --output-dir ./seals/
python cli.py --cache-info              # show cache stats
python cli.py --clear-cache             # purge all cached data
python cli.py --no-api-cache            # bypass cache for this run
python cli.py --debug --text "齐白石" --type name  # verbose diagnostics
python cli.py --text "禅" --debug-extract           # save extractor intermediates
python cli.py --text "天人合一" --debug-layout       # save layout overlay (cells/ink/centroid)
python cli.py --text "禅" --seed 42                  # reproducible texture
python cli.py --text "齐白石" --type name --strict-consistency  # require Level 1-2 source
python cli.py --batch chars.txt --jobs 3                         # 3 concurrent workers (thread-local SealGenerator per worker)
```

## Testing

63 unit + regression tests under `tests/`. Pytest config in `pyproject.toml` uses `filterwarnings = "error"` (with PIL/opencc whitelisted), `--strict-markers`, and a session-scoped `gen` fixture in `tests/regression/conftest.py`.

```bash
uv run python -m pytest tests/ -q                       # all 63 tests
uv run python -m pytest tests/ -m unit                  # fast, no network
uv run python -m pytest tests/ -m regression            # uses cache + SealGenerator
uv run python tests/regression/run.py --run-id baseline # 40-case visual baseline
uv run python tests/regression/compare.py A B           # side-by-side HTML diff
```

The regression runner derives a stable per-case seed via `MD5(test_id + attempt)` so output is byte-reproducible across separate processes (Python's `hash()` is randomized via PYTHONHASHSEED, so it's not used). Determinism audit at `docs/plans/determinism-audit-2026-04-14.md` confirmed 9/9 byte-identical across processes.

## Errors

`core/errors.py` defines typed pipeline failures, all inheriting from `SealError`:

| Class | When | Carries |
|---|---|---|
| `CharNotFoundError` | char missing across all priority scripts | `char`, `scripts_tried` |
| `SourceInconsistencyError` | strict mode rejected level > 2 | `text`, `level` |
| `ExtractionFailedError` | extraction quality validation failed | `char`, `reason` |
| `UpstreamApiError` | ygsf.com HTTP non-200 (non-429) | `status_code`, `detail` |
| `RateLimitedError` | ygsf.com HTTP 429 | `retry_after` (seconds, optional — from `Retry-After` header) |

All inherit from `Exception`, so existing `except Exception` callers keep catching them. Currently raised at: scraper retry exhaustion (`_query_glyph_list`), CLI `--strict-consistency` check.

## Dependencies

`pycryptodome` (AES), `opencv-python` (CV), `opencc-python-reimplemented` (简→繁), `gradio` (UI), `rich` (CLI), `fake-useragent` (scraping).

## Conventions

- Inversion/normalization ONLY in extractor, never scraper
- Cache: `~/.seal_gen/cache/` — `_api/`, `_img/`, `{char}_{font}_{tab}.png`
- Local font fallback: serif preferred (思源宋体 > 黑体)
- Extreme-aspect chars use reverse construction; normal chars use equal-ratio + conditional stretch
- 留红即设计: single-char-per-column whitespace is intentional (traditional 章法)
- Public debug API: use `SealGenerator.set_extract_debug_dir()` and `SealGenerator.render_layout_debug()`, not `gen._extractor.debug_dir = ...`
- New failure modes go through `core.errors` (typed) — `ValueError` reserved for programmer errors (invalid hex color, empty text)
- Tests with fixture `gen` belong under `tests/regression/` (network/cache); pure unit tests at top-level `tests/`
- Pipeline assertion in `core/__init__.py`: baiwen → opaque > 0.5, zhuwen → transparent > 0.2 (post-render, pre-texture). Catches yin/yang mixing bugs in debug mode (`__debug__`).
