# fangcun (方寸)

A Geek-Zen seal generator. 数字化传统金石艺术，代码刻出的方寸之美。

## Architecture

Modular `core/` package with zero UI dependencies. Two thin entry points share the same pipeline.

```
core/
├── __init__.py      # SealGenerator — unified pipeline orchestrator
├── scraper.py       # ygsf.com API client (AES-ECB encrypted) + disk cache + local font fallback
├── extractor.py     # Three-tier polarity normalization + two-phase 印谱 extraction
├── layout.py        # Traditional right-to-left vertical layout (1–4+ chars)
├── renderer.py      # Baiwen/zhuwen rendering, oval/square shapes, double-line frames
└── texture.py       # Stone-carved texture: edge roughness, chipping, ink grain
app.py               # Gradio Web UI (主入口)
cli.py               # CLI batch entry with rich progress (次入口)
```

## Key Design Decisions

- **core/ is UI-agnostic**: only accepts params, returns PIL.Image. Never imports gradio or rich.
- **Scraper uses AES-ECB encryption**: ygsf.com API requires encrypted request/response. Protocol in `core/scraper.py` header.
- **Tab priority (字典→真迹, never 字库)**: `type=3` preferred, `type=2` fallback, `type=1` excluded.
- **Font consistency (同印同体)**: `fetch_chars_consistent()` tries each priority for ALL characters before falling back. No 行书/草书 ever.
- **Top-N candidate scoring**: 5 candidates, resolution (0–30) + contrast (0–50) + coverage (0–20). Raw images + source_name — no inversion in scraper.
- **Simplified→traditional fallback**: `opencc` auto-tries 蘇 when 苏 fails.

### Extractor: The hardest engineering problem

印谱 images (鸟虫篆全书 etc.) have inverted polarity: white stroke slots carved into black stone, wrapped in transparent padding. Multiple failed approaches taught us what works.

**Three-tier polarity detection** (`_normalize_to_black_on_white`):
- Tier 1: `KNOWN_YINPU_SOURCES` whitelist → two-phase alpha extraction
- Tier 2: Alpha semantic detection (bright pixel ratio in opaque region)
- Tier 3: Morphological erosion fallback

**Two-phase 印谱 extraction** (`_extract_yinpu_strokes`) — handles BOTH polarity and frame removal in one step:

Phase 1 — Aggregated Alpha CCA:
- Find largest opaque connected component (stone surface)
- Aggregate nearby large chunks (area > 5%, distance < 2× short side)
- Produces true stone surface bounding box, filtering page scan borders

Phase 2 — CCA bbox edge inset:
- Scan inward from each CCA bbox edge
- Require 3 consecutive rows with opaque ratio >= 50% (noise-resistant)
- Everything before = frame channel, stripped with 1px safety buffer
- Only transparent holes (α<128) within the inset region become strokes

**Failed approaches** (for future reference):
- `_composite_and_invert`: padding and strokes both become black → white block
- CCA frame removal: fails when frame and stroke pixels are 粘连 (connected)
- 1D projection with full-image threshold: fails when bbox << image size (padding)
- Single-row opaque scan: noise pixels in frame channels cause early stop

### Other design decisions

- **Binarization**: Otsu for 字典/本地 (never adaptive with small block — hollow outlines); bilateral filter + adaptive for 真迹
- **Cache**: `~/.seal_gen/cache/{char}_{font}_{tab}.png` + `.src` metadata
- **Local font fallback**: serif preferred (思源宋体 > 黑体)
- **Inversion only in extractor, never scraper** — ensures all chars in a seal get identical processing

## Data Flow

```
SealGenerator.generate()
  → scraper.fetch_chars_consistent()       # same style for all chars (同印同体)
    → _get_or_fetch(char, font)            # 字典→真迹, simplified→traditional
      → _fetch_from_web()                  # AES-encrypted API
        → _download_best_image()           # top-5 scoring → (img, source_name)
  → extractor.extract(img, tab, source_name)
    → _normalize_to_black_on_white()       # Tier 1/2/3 polarity defense
      → _extract_yinpu_strokes()           # two-phase: CCA bbox + edge inset
    → _binarize_otsu() / _binarize_adaptive()
    → _denoise() → _crop_bbox()
  → layout.arrange()                       # traditional vertical layout (列优先)
  → renderer.render()                      # baiwen/zhuwen RGBA output
  → texture.apply()                        # stone-carved effects (skip if grain=0)
  → rotate + preview                       # final transparent + white-bg
```

## Running

```bash
python app.py                   # Web UI → http://localhost:7860
python cli.py --text "禅" --shape oval --style baiwen --type leisure
python cli.py --batch chars.txt --output-dir ./seals/
```

## Dependencies

`pycryptodome` (AES), `opencv-python` (CV), `opencc-python-reimplemented` (简→繁), `gradio` (UI), `rich` (CLI), `fake-useragent` (scraping).
