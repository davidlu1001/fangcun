# fangcun (方寸)

A Geek-Zen seal generator. 数字化传统金石艺术，代码刻出的方寸之美。

## Architecture

Modular `core/` package with zero UI dependencies. Two thin entry points share the same pipeline.

```
core/
├── __init__.py      # SealGenerator — unified pipeline orchestrator
├── scraper.py       # ygsf.com API client (AES-ECB encrypted) + disk cache + local font fallback
├── extractor.py     # Adaptive binarization → clean stroke mask (mode "L")
├── layout.py        # Traditional right-to-left vertical layout (1–4+ chars)
├── renderer.py      # Baiwen/zhuwen rendering, oval/square shapes, double-line frames
└── texture.py       # Stone-carved texture: edge roughness, chipping, ink grain
app.py               # Gradio Web UI (主入口)
cli.py               # CLI batch entry with rich progress (次入口)
```

## Key Design Decisions

- **core/ is UI-agnostic**: only accepts params, returns PIL.Image. Never imports gradio or rich.
- **Scraper uses AES-ECB encryption**: ygsf.com API requires encrypted request/response. Key and protocol details are documented in `core/scraper.py` header comments.
- **Font priority by seal type**: 闲章 → 篆→隶→楷; 名章 → 隶→楷. Strict: no 行书/草书 ever.
- **Cache at `~/.seal_gen/cache/`**: keyed by `{char}_{font_style}.png`.
- **Local font fallback**: if all web fonts fail, renders with system Chinese font (auto-detected via `fc-list`).

## Data Flow

```
SealGenerator.generate()
  → scraper.fetch_char_image()   # per character, with cache + fallback
  → extractor.extract()          # raw image → binary mask (mode "L")
  → layout.arrange()             # masks → placement list [{img, x, y, w, h}]
  → renderer.render()            # placements → RGBA seal image
  → texture.apply()              # add stone-carved effects
  → rotate + create preview      # final output
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
