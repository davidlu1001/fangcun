# 方寸 · 极客禅印章生成器

Fangcun — Chinese Seal Generator for Geek-Zen. 数字化传统金石艺术，代码刻出的方寸之美。

## Architecture

Modular `core/` package with zero UI dependencies. Two thin entry points share the same pipeline.

```
core/
├── __init__.py      # SealGenerator — unified pipeline orchestrator
├── scraper.py       # ygsf.com API + 5-level source unification + scoring + 3-tier cache
├── extractor.py     # Three-tier polarity normalization + two-phase 印谱 extraction
├── layout.py        # Multi-phase layout: fit → extreme-detect → reverse-construct → stroke-normalize
├── renderer.py      # Baiwen/zhuwen rendering, max-alpha merge, style-aware text_scale
└── texture.py       # 6-layer texture: pressure, roughness, chipping, grain, ink pooling, color drift
app.py               # Gradio Web UI (主入口)
cli.py               # CLI batch entry with rich progress + cache management (次入口)
```

## Scraper: 5-level source selection (同源同体)

Beyond same script style (同印同体), the scraper pursues same *source* (碑帖) for visual consistency. Five levels, each falling back to the next:

```
Level 1: 统一来源 n=5    all chars from one source (intersection)
Level 2: 统一来源 n=10   wider candidate pool retry
Level 3: 多数来源 >50%   majority vote, fallback for uncovered chars
Level 4: 最小损失来源     coverage×100 − score_loss ranking
Level 5: 各字独立最优     per-char best (last resort)
```

**Image scoring** (0–100): resolution (0–30) + contrast (0–50) + coverage (0–20)
- 印谱 fragmentation hard-reject: only when light_ratio > 0.15 in opaque region (prevents false kills on multi-stroke chars like 道/轼)
- 印谱 source base penalty: -10
- Inferior style penalty: -25 (金文/简帛 sources unsuitable for seal carving)
- `KNOWN_YINPU_SOURCES` and `INFERIOR_STYLE_SOURCES` sets in extractor/scraper

**Deferred state machine**: prefers lower-priority font with unified source over higher-priority font with mixed sources (隶書同源 > 篆書異源). When a font covers all chars but no unified source exists, saves a fallback and continues to the next font instead of returning immediately.

**Three-tier cache** (eliminates repeat network requests):
- Tier 1: API response JSON cache (`_api/`) — keyed by MD5 of params dict, 30d/7d TTL
- Tier 2: Image CDN cache (`_img/`) — keyed by MD5 of URL, persists candidates
- Tier 3: Selected-best cache (`{char}_{font}_{tab}.png`) — final selection
- Cache check before `time.sleep`/`_encrypt_params` for zero-latency hits
- CLI: `--cache-info`, `--clear-cache`, `--no-api-cache` flags

**Other scraper decisions**:
- Tab priority: 字典 (type=3) → 真迹 (type=2), never 字库 (type=1)
- Font consistency: all chars must share same script. 闲章: 篆→隶→楷; 名章: 隶→楷
- Simplified→traditional: opencc auto-tries 蘇 when 苏 fails
- Atomic cache writes (tempfile + os.replace)
- AES-ECB encrypted API. Protocol in `scraper.py` header.

## Extractor: Three-tier polarity + two-phase 印谱 extraction

**Three-tier polarity detection** (`_normalize_to_black_on_white`):
- Tier 1: `KNOWN_YINPU_SOURCES` whitelist → two-phase alpha extraction
- Tier 2: Alpha semantic detection (bright pixel ratio in opaque region)
- Tier 3: Morphological erosion fallback

**Two-phase `_extract_yinpu_strokes`** (handles polarity AND frame removal):

Phase 1 — Per-chunk Alpha CCA:
- Find stone surface connected components, filter by area (>15%) and proximity
- Each chunk processed independently (handles cracked stone)
- Mid-60% sampling for edge scan (avoids corner cross-contamination)

Phase 2 — Edge inset per chunk:
- Consecutive-3-row opaque scan (noise-resistant)
- MIN_INSET floor: max(6, short_side × 0.012)
- Safety buffer: 2px
- Only transparent holes (α<128) within inset region become strokes

**Binarization**: Otsu for 字典/本地; bilateral filter + adaptive for 真迹. Never adaptive with small block on clean images (hollow outlines).

## Layout: Multi-phase pipeline

`SealLayout.arrange()` runs five phases:

1. **Normal fit**: Equal-ratio scaling + conditional tall/wide cell stretch (max 1.25x)
2. **Extreme detection**: Identify chars with aspect > 2.5 (一二三) or < 0.4, compute sibling median stroke width
3. **Reverse construction**: Extreme chars rebuilt from scratch — resize stroke to target width, center in padded canvas (ink_ratio ≤ 55%, length = 70% cell)
4. **Stroke-width normalize**: Distance-transform-based equalization for normal chars (dilate < 0.75x median, erode > 1.35x)
5. **Centroid placement**: Pixel-weighted centroid compensation (0.65 coefficient)

**Dynamic margins**: `MARGIN_TABLE` by (style, shape): baiwen/square=0.06, baiwen/oval=0.04, zhuwen/square=0.03, zhuwen/oval=0.01.

**Zhuwen bleed (冲刀破边)**: `text_scale=0.98` puts text area ~6px from frame. 4% bleed with `margin=0` lets chars physically cross the frame line. Verified: 1500+ pixels in each frame zone.

**Single-char fill boost**: `text_scale=0.93` for single-char baiwen (顶天立地); multi-char uses 0.86.

## Aesthetic decisions

- **Texture (6-layer)**: (1) Pressure variation — low-freq alpha modulation. (2) Dual-path frame roughness — thin-line detection routes to gap-breakage; thick areas use shell-confined erosion with contagion expansion across adjacent frame lines. (3) Salt-noise chipping. (4) Ink grain. (4.5) Stroke intersection darkening — distance-transform ink pooling. (4.6) Color temperature drift — per-channel RGB shift. (5) Final blur.
- **Renderer**: Zhuwen uses max-alpha merge (char + frame layers via `np.maximum`) for natural bleed fusion. Baiwen uses mask paste (white cutout on red).
- **No indiscriminate dilate/erode**: Extreme flat chars use reverse construction (canvas-based), not post-hoc dilation which creates rectangles.

## Data Flow

```
SealGenerator.generate()
  → scraper.fetch_chars_consistent()
    → _query_glyph_list()                  # API cache → web, 3 retries
    → _fetch_all_candidates(n=5/10)       # image CDN cache → download
    → _try_unified_source_from_candidates # Level 1-2: source intersection
    → _majority_source_fallback           # Level 3: >50% coverage vote
    → _min_style_loss_fallback            # Level 4: coverage×100 − loss
  → extractor.extract(img, tab, source_name)
    → _normalize_to_black_on_white()      # Tier 1/2/3
      → _extract_yinpu_strokes()          # per-chunk CCA + edge inset
    → _binarize_otsu() / _binarize_adaptive()
    → _denoise() → _crop_bbox()
  → layout.arrange(style, shape)            # 5-phase: fit → detect → construct → normalize → place
  → renderer.render()                     # baiwen paste / zhuwen max-alpha merge
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
```

## Dependencies

`pycryptodome` (AES), `opencv-python` (CV), `opencc-python-reimplemented` (简→繁), `gradio` (UI), `rich` (CLI), `fake-useragent` (scraping).

## Conventions

- Inversion/normalization ONLY in extractor, never scraper
- Cache: `~/.seal_gen/cache/` — `_api/` (JSON), `_img/` (CDN PNGs), `{char}_{font}_{tab}.png` (selected best)
- Local font fallback: serif preferred (思源宋体 > 黑体)
- Conditional stretch max 1.25x for tall/wide cells; extreme-aspect chars use reverse construction
- 留红即设计: single-char-per-column whitespace is intentional (traditional 章法)
