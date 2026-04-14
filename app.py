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


_CONSISTENCY_LABELS = {
    1: "L1 · 统一来源（最佳）",
    2: "L2 · 宽池统一来源",
    3: "L3 · 多数来源（部分字从次优）",
    4: "L4 · 最小损失来源",
    5: "L5 · 各字独立最优（来源混合）",
}


def generate_seal(
    text: str,
    shape_label: str,
    style_label: str,
    seal_type_label: str,
    color_hex: str,
    grain_strength: float,
    rotation: float,
    seed_value: float | None,
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

    # seed_value comes in as float from gr.Number; None or 0 means "random"
    seed: int | None = None
    if seed_value is not None and seed_value > 0:
        seed = int(seed_value)

    try:
        result = _gen.generate(
            text=text,
            shape=shape,
            style=style,
            seal_type=seal_type,
            color=color_hex,
            grain=grain_strength,
            rotation=rotation,
            seed=seed,
        )
    except Exception as exc:
        logging.exception("Generation failed")
        return None, None, f"生成失败: {exc}", None, None

    # Build status text
    if result["font_fallback"]:
        status = f"⚠ 字体已降级 → 使用: {result['font_used']}"
    else:
        status = f"✓ 使用: {result['font_used']}"

    # Consistency level (L1 best → L5 worst)
    level = result.get("consistency_level", 0)
    if level:
        status += f"\n\n**字源一致性**: {_CONSISTENCY_LABELS.get(level, f'L{level}')}"

    if result["warnings"]:
        status += "\n\n" + "\n".join(f"⚠ {w}" for w in result["warnings"])

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

            seed_input = gr.Number(
                label="🎲 随机种子（留空或 0 = 随机；固定数字 = 可复现）",
                value=None,
                precision=0,
                minimum=0,
            )

            generate_btn = gr.Button("✨ 生成印章", variant="primary")

        # ── Right panel: preview ─────────────────────────────
        with gr.Column(scale=1):
            transparent_output = gr.Image(
                label="印章（透明底，用于实际盖章）",
                type="pil",
                interactive=False,
            )

            preview_output = gr.Image(
                label="印章预览（白底，仅供查看效果）",
                type="pil",
                interactive=False,
            )

            status_output = gr.Markdown(value="等待生成...")

            with gr.Row():
                dl_transparent = gr.DownloadButton(
                    label="⬇ 下载 PNG（透明底 · 主用）",
                    visible=True,
                )
                dl_preview = gr.DownloadButton(
                    label="⬇ 下载 PNG（白底预览 · 备用）",
                    visible=True,
                )

    # Hidden state for file paths
    transparent_path_state = gr.State(value=None)
    preview_path_state = gr.State(value=None)

    def on_generate(text, shape, style, seal_type, color, grain, rotation, seed):
        img_t, img_p, status, path_t, path_p = generate_seal(
            text, shape, style, seal_type, color, grain, rotation, seed
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
            seed_input,
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
