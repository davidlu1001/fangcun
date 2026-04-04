# fangcun (方寸)

A Geek-Zen seal generator. 数字化传统金石艺术，代码刻出的方寸之美。

## Architecture

Modular `core/` package with zero UI dependencies. Two thin entry points share the same pipeline.

```
core/
├── __init__.py      # SealGenerator — unified pipeline orchestrator
├── scraper.py       # ygsf.com API client (AES-ECB encrypted) + disk cache + local font fallback
├── extractor.py     # Source-aware binarization → clean stroke mask (mode "L")
├── layout.py        # Traditional right-to-left vertical layout (1–4+ chars)
├── renderer.py      # Baiwen/zhuwen rendering, oval/square shapes, double-line frames
└── texture.py       # Stone-carved texture: edge roughness, chipping, ink grain
app.py               # Gradio Web UI (主入口)
cli.py               # CLI batch entry with rich progress (次入口)
```

## Key Design Decisions

- **core/ is UI-agnostic**: only accepts params, returns PIL.Image. Never imports gradio or rich.
- **Scraper uses AES-ECB encryption**: ygsf.com API requires encrypted request/response. Key and protocol details are documented in `core/scraper.py` header comments.
- **Tab priority (字典→真迹, never 字库)**: `type=3` (字典/dictionary) is preferred — clean B&W from seal references. `type=2` (真迹/authentic) is fallback — original calligraphy, noisier. `type=1` (字库/font library) is excluded — digital font glyphs with vector edges are unusable for seals.
- **Font consistency**: all characters in a seal must share the same script style (金石学原则). `fetch_chars_consistent()` tries each priority (闲章: 篆→隶→楷; 名章: 隶→楷) for ALL characters before falling back. No 行书/草书 ever.
- **Top-N candidate scoring**: downloads up to 5 candidates per query, scores on resolution (0–30), contrast/std (0–50), coverage (0–20). Selects highest-scoring candidate.
- **Three-tier polarity normalization** (in extractor, NOT scraper):
  - Tier 1: `KNOWN_YINPU_SOURCES` whitelist → direct alpha-hole extraction
  - Tier 2: Alpha semantic detection (bright pixel ratio in opaque region)
  - Tier 3: Morphological erosion fallback (large surviving blocks = dark bg)
  - 印谱 images use `_extract_yinpu_strokes()` — strokes = transparent holes WITHIN the opaque block bbox; transparent padding outside excluded.
  - `_remove_seal_frame()` strips rectangular frame lines via connected component analysis after extraction.
- **Source-aware binarization**: 字典 → Otsu; 真迹 → bilateral filter + adaptive threshold; 本地 → Otsu.
- **Simplified→traditional fallback**: `opencc` auto-tries 蘇 when 苏 fails (ygsf indexes traditional).
- **Cache at `~/.seal_gen/cache/`**: keyed by `{char}_{font}_{tab}.png` + `.src` metadata.
- **Local font fallback**: serif fonts preferred (思源宋体 > 黑体) for seal aesthetic.

## Data Flow

```
SealGenerator.generate()
  → scraper.fetch_chars_consistent()     # consistency-first: same style for all chars
    → _get_or_fetch(char, font)          # 字典→真迹 tabs, simplified→traditional
      → _fetch_from_web()               # AES-encrypted API, returns (img, source_name)
        → _download_best_image()         # top-5 scoring, returns (img, source_name)
  → extractor.extract(img, tab, source_name)
    → _normalize_to_black_on_white()     # Tier 1/2/3 polarity defense
      → _extract_yinpu_strokes()         # for 印谱: alpha holes within opaque bbox
    → _binarize_otsu() / _binarize_adaptive()
    → _remove_seal_frame()              # for 印谱: strip rectangular frame lines
    → _denoise() → _crop_bbox()
  → layout.arrange()                    # traditional vertical layout
  → renderer.render()                   # baiwen/zhuwen RGBA output
  → texture.apply()                     # stone-carved effects (skip if grain=0)
  → rotate + preview                    # final transparent + white-bg
```

## Running

```bash
# Web UI
python app.py

# CLI
python cli.py --text "禅" --shape oval --style baiwen --type leisure
python cli.py --batch chars.txt --output-dir ./seals/
```

## Dependencies

Key external: `pycryptodome` (AES), `opencv-python` (binarization/morphology), `gradio` (UI), `rich` (CLI), `fake-useragent` (scraping).

## Conventions

- All core functions use type annotations
- Immutable dataclasses where applicable (e.g. `Placement`)
- Logging via `logging` module, not print statements
- PIL Image modes: scraper returns any mode, extractor outputs "L", renderer outputs "RGBA"
