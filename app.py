"""
方寸 · 极客禅印章生成器 — Gradio Web UI (主入口)

Usage:
    python app.py
    → http://localhost:7860
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import gradio as gr
from PIL import Image

from core import SealGenerator

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

_gen = SealGenerator()

# ── shape / style mapping (Chinese UI ↔ internal codes) ──────

_SHAPE_MAP = {"竖椭圆": "oval", "方章": "square"}
_STYLE_MAP = {"白文": "baiwen", "朱文": "zhuwen"}
_TYPE_MAP = {"名章（强制篆书）": "name", "闲章（允许隶楷）": "leisure", "品牌章（任何字体）": "brand"}


def generate_seal(
    text: str,
    shape_label: str,
    style_label: str,
    seal_type_label: str,
    color_hex: str,
    grain_strength: float,
    rotation: float,
) -> tuple[Image.Image | None, Image.Image | None, str, str | None, str | None]:
    """
    Main generation callback for Gradio.

    Returns:
        (transparent_img, preview_img, status_markdown,
         transparent_download_path, preview_download_path)
    """
    text = text.strip()
    if not text:
        return None, None, "请输入印章文字（1–4字）", None, None

    shape = _SHAPE_MAP.get(shape_label, "oval")
    style = _STYLE_MAP.get(style_label, "baiwen")
    seal_type = _TYPE_MAP.get(seal_type_label, "leisure")

    try:
        result = _gen.generate(
            text=text,
            shape=shape,
            style=style,
            seal_type=seal_type,
            color=color_hex,
            grain=grain_strength,
            rotation=rotation,
        )
    except Exception as exc:
        logging.exception("Generation failed")
        return None, None, f"生成失败: {exc}", None, None

    # Build status text
    if result["font_fallback"]:
        status = f"⚠ 字体已降级 → 使用: {result['font_used']}"
    else:
        status = f"✓ 使用: {result['font_used']}"

    if result["warnings"]:
        status += "\n" + "\n".join(f"⚠ {w}" for w in result["warnings"])

    # Save to temp files for download buttons
    tmp_dir = Path(tempfile.gettempdir()) / "seal_gen"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    transparent_path = str(tmp_dir / f"{text}_transparent.png")
    preview_path = str(tmp_dir / f"{text}_preview.png")
    result["image_transparent"].save(transparent_path, "PNG")
    result["image_preview"].save(preview_path, "PNG")

    return (
        result["image_transparent"],
        result["image_preview"],
        status,
        transparent_path,
        preview_path,
    )


# ── Gradio UI ────────────────────────────────────────────────

with gr.Blocks(title="方寸 · 极客禅印章生成器", theme=gr.themes.Soft()) as demo:

    gr.Markdown("# 🖋 方寸 · 极客禅印章生成器\n*Fangcun — Chinese Seal Generator for Geek-Zen*")

    with gr.Row():
        # ── Left panel: parameters ───────────────────────────
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="📝 印章文字",
                placeholder="输入印章文字（1–4字）",
                value="禅",
                max_lines=1,
            )

            shape_input = gr.Radio(
                choices=["竖椭圆", "方章"],
                label="📐 形制",
                value="竖椭圆",
            )

            style_input = gr.Radio(
                choices=["白文", "朱文"],
                label="🖌 风格",
                value="白文",
            )

            seal_type_input = gr.Radio(
                choices=["名章（强制篆书）", "闲章（允许隶楷）", "品牌章（任何字体）"],
                label="📋 章类",
                value="闲章（允许隶楷）",
            )

            color_input = gr.ColorPicker(
                label="🎨 朱砂颜色",
                value="#B22222",
            )

            grain_input = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                value=0.25,
                label="🪨 质感强度",
            )

            rotation_input = gr.Slider(
                minimum=0.0,
                maximum=5.0,
                step=0.5,
                value=2.0,
                label="🔄 旋转角度（°）",
            )

            generate_btn = gr.Button("✨ 生成印章", variant="primary")

        # ── Right panel: preview ─────────────────────────────
        with gr.Column(scale=1):
            transparent_output = gr.Image(
                label="印章预览（透明底）",
                type="pil",
                interactive=False,
            )

            preview_output = gr.Image(
                label="印章预览（白底）",
                type="pil",
                interactive=False,
            )

            status_output = gr.Markdown(value="等待生成...")

            with gr.Row():
                dl_transparent = gr.DownloadButton(
                    label="⬇ 下载 PNG（透明底）",
                    visible=True,
                )
                dl_preview = gr.DownloadButton(
                    label="⬇ 下载 PNG（白底预览）",
                    visible=True,
                )

    # Hidden state for file paths
    transparent_path_state = gr.State(value=None)
    preview_path_state = gr.State(value=None)

    def on_generate(text, shape, style, seal_type, color, grain, rotation):
        img_t, img_p, status, path_t, path_p = generate_seal(
            text, shape, style, seal_type, color, grain, rotation
        )
        return img_t, img_p, status, path_t, path_p

    generate_btn.click(
        fn=on_generate,
        inputs=[
            text_input,
            shape_input,
            style_input,
            seal_type_input,
            color_input,
            grain_input,
            rotation_input,
        ],
        outputs=[
            transparent_output,
            preview_output,
            status_output,
            dl_transparent,
            dl_preview,
        ],
    )


if __name__ == "__main__":
    is_hf_space = os.environ.get("SPACE_ID") is not None
    demo.launch(server_name="0.0.0.0" if is_hf_space else "127.0.0.1")
