# fangcun (方寸)

A Geek-Zen seal generator. 数字化传统金石艺术，代码刻出的方寸之美。

## Architecture

Modular `core/` package with zero UI dependencies. Two thin entry points share the same pipeline.

```
core/
├── __init__.py      # SealGenerator — unified pipeline orchestrator
├── scraper.py       # ygsf.com API client (AES-ECB encrypted) + disk cache + local font fallback
├── extractor.py     # Three-tier polarity normalization + binarization + frame removal
├── layout.py        # Traditional right-to-left vertical layout (1–4+ chars)
├── renderer.py      # Baiwen/zhuwen rendering, oval/square shapes, double-line frames
└── texture.py       # Stone-carved texture: edge roughness, chipping, ink grain
app.py               # Gradio Web UI (主入口)
cli.py               # CLI batch entry with rich progress (次入口)
```

## Key Design Decisions

- **core/ is UI-agnostic**: only accepts params, returns PIL.Image. Never imports gradio or rich.
- **Scraper uses AES-ECB encryption**: ygsf.com API requires encrypted request/response. Key and protocol details in `core/scraper.py` header comments.
- **Tab priority (字典→真迹, never 字库)**: `type=3` (字典) preferred — clean B&W from seal references. `type=2` (真迹) fallback — noisier. `type=1` (字库) excluded — digital font glyphs unusable.
- **Font consistency (同印同体)**: all characters in a seal must share the same script style. `fetch_chars_consistent()` tries each priority for ALL characters before falling back. No 行书/草书 ever.
- **Top-N candidate scoring**: downloads up to 5 candidates, scores on resolution (0–30), contrast/std (0–50), coverage (0–20). Scraper returns raw images + source_name — no inversion in scraper.
- **Simplified→traditional fallback**: `opencc` auto-tries 蘇 when 苏 fails.

### Extractor: Three-tier polarity + frame removal

The hardest engineering problem in this project. 印谱 images (e.g. 鸟虫篆全书) have inverted polarity: white stroke slots carved into black stone surface, wrapped in transparent padding. Naive compositing/inversion fails because padding and strokes are both transparent.

**Three-tier polarity normalization** (`_normalize_to_black_on_white`):
- Tier 1: `KNOWN_YINPU_SOURCES` whitelist → `_extract_yinpu_strokes()` (alpha-hole extraction)
- Tier 2: Alpha semantic detection — bright pixel ratio (>200) in opaque region. >15% = 印谱, <5% = normal
- Tier 3: Morphological erosion fallback — large surviving blocks after erosion = dark background

**印谱 stroke extraction** (`_extract_yinpu_strokes`):
- Strokes = transparent holes (α<128) WITHIN the opaque block bbox
- Transparent padding OUTSIDE the bbox is excluded (not strokes)
- Produces clean white-bg + black-strokes grayscale for Otsu

**Seal frame removal** (`_remove_seal_frame`):
- 1D pixel projection (NOT CCA — CCA fails when frame and stroke pixels are physically connected/粘连)
- Row/col projection finds lines where pixel count > 50% of width/height
- Only scans outer 15% — center zone absolutely protected (preserves 一)
- Convolve FIRST to dilate thick/double frames, THEN hard-truncate center zone

### Other design decisions

- **Source-aware binarization**: 字典 → Otsu; 真迹 → bilateral filter + adaptive threshold
- **Cache at `~/.seal_gen/cache/`**: keyed by `{char}_{font}_{tab}.png` + `.src` metadata for source_name
- **Local font fallback**: serif fonts preferred (思源宋体 > 黑体) for seal aesthetic

## Data Flow

```
SealGenerator.generate()
  → scraper.fetch_chars_consistent()     # same style for all chars (同印同体)
    → _get_or_fetch(char, font)          # 字典→真迹 tabs, simplified→traditional
      → _fetch_from_web()               # AES-encrypted API
        → _download_best_image()         # top-5 scoring, returns (img, source_name)
  → extractor.extract(img, tab, source_name)
    → _normalize_to_black_on_white()     # Tier 1/2/3 polarity defense
      → _extract_yinpu_strokes()         # 印谱: alpha holes within opaque bbox
    → _binarize_otsu() / _binarize_adaptive()
    → _remove_seal_frame()              # 印谱: 1D projection strips frame lines
    → _denoise() → _crop_bbox()
  → layout.arrange()                    # traditional vertical layout (列优先)
  → renderer.render()                   # baiwen/zhuwen RGBA output
  → texture.apply()                     # stone-carved effects (skip if grain=0)
  → rotate + preview                    # final transparent + white-bg
```

## Running

```bash
# Web UI
python app.py
# → http://localhost:7860

# CLI
python cli.py --text "禅" --shape oval --style baiwen --type leisure
python cli.py --batch chars.txt --output-dir ./seals/
```

## Dependencies

Key external: `pycryptodome` (AES), `opencv-python` (binarization/morphology), `opencc-python-reimplemented` (simplified→traditional), `gradio` (UI), `rich` (CLI), `fake-useragent` (scraping).

## Conventions

- All core functions use type annotations
- Logging via `logging` module, not print statements
- PIL Image modes: scraper returns raw (any mode), extractor outputs "L", renderer outputs "RGBA"
- Inversion/normalization happens ONLY in extractor (never in scraper) — ensures all chars in a seal go through identical processing
