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

## Renderer: text_scale by context

| Style + Shape | text_scale | Rationale |
|---|---|---|
| zhuwen square | 0.98 | Near frame for bleed (冲刀破边) |
| zhuwen oval | 0.88 | Oval corners clip — must keep text inside ellipse (R11) |
| baiwen, 1 char | 0.93 | 顶天立地 fill |
| baiwen, 2+ chars | 0.86 | Leave red border for visual weight |

Zhuwen uses max-alpha merge (`np.maximum` of char + frame layers). Baiwen uses mask paste (white cutout on red).

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
  → texture.apply()                       # 6-layer stone effects
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
python cli.py --debug --text "卢修齐" --type name  # verbose diagnostics
```

## Dependencies

`pycryptodome` (AES), `opencv-python` (CV), `opencc-python-reimplemented` (简→繁), `gradio` (UI), `rich` (CLI), `fake-useragent` (scraping).

## Conventions

- Inversion/normalization ONLY in extractor, never scraper
- Cache: `~/.seal_gen/cache/` — `_api/`, `_img/`, `{char}_{font}_{tab}.png`
- Local font fallback: serif preferred (思源宋体 > 黑体)
- Extreme-aspect chars use reverse construction; normal chars use equal-ratio + conditional stretch
- 留红即设计: single-char-per-column whitespace is intentional (traditional 章法)
