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
_POLARITY_MAP = {
    "自动检测": "auto",
    "黑字白底": "black_on_white",
    "白字黑底（印谱拓片）": "white_on_black",
}


def _collect_user_glyphs(
    char_inputs: list[str | None],
    img_inputs: list[Image.Image | None],
) -> tuple[str, list[Image.Image]]:
    """Pair non-empty (char, image) entries in UI order.

    Raises gr.Error on missing pairs or invalid chars. Returns the derived
    text (concatenated chars) and the matching glyph list.
    """
    pairs: list[tuple[str, Image.Image]] = []
    for c, img in zip(char_inputs, img_inputs):
        if not c and img is None:
            continue  # both empty — skip row
        if not c or not c.strip():
            raise gr.Error("已上传图片但未填写对应汉字")
        if img is None:
            raise gr.Error(f"汉字 '{c}' 缺少对应的上传图片")
        char = c.strip()
        if len(char) != 1:
            raise gr.Error(f"每个输入框只能填一个汉字，'{char}' 不符合")
        pairs.append((char, img))

    if not pairs:
        raise gr.Error("请至少为一个字上传图片")

    return "".join(c for c, _ in pairs), [img for _, img in pairs]


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
    user_glyphs: list[Image.Image] | None = None,
    user_glyph_polarity: str = "auto",
) -> tuple[Image.Image | None, Image.Image | None, str, str | None, str | None]:
    """
    Main generation callback for Gradio.

    When `user_glyphs` is provided, the scraper is bypassed. `text` must
    still be passed — its length must equal len(user_glyphs).

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
            user_glyphs=user_glyphs,
            user_glyph_polarity=user_glyph_polarity,
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


def generate_variants(
    text: str,
    shape_label: str,
    style_label: str,
    seal_type_label: str,
    color_hex: str,
    grain_strength: float,
    rotation: float,
) -> tuple[list | None, str]:
    """Generate 3 texture-seed variations sharing identical source selection.

    Returns a list of (image, caption) tuples for gr.Gallery plus a status string.
    """
    text = text.strip()
    if not text:
        return None, "请输入印章文字（1–4字）"

    shape = _SHAPE_MAP.get(shape_label, "oval")
    style = _STYLE_MAP.get(style_label, "baiwen")
    seal_type = _TYPE_MAP.get(seal_type_label, "leisure")

    try:
        results = _gen.generate_variants(
            text=text,
            n=3,
            shape=shape,
            style=style,
            seal_type=seal_type,
            color=color_hex,
            grain=grain_strength,
            rotation=rotation,
            seeds=[1, 2, 3],
        )
    except Exception as exc:
        logging.exception("Variant generation failed")
        return None, f"生成失败: {exc}"

    # Shared metadata across variants — read from first
    first = results[0]
    if first["font_fallback"]:
        status = f"⚠ 字体已降级 → 使用: {first['font_used']}"
    else:
        status = f"✓ 使用: {first['font_used']}"

    level = first.get("consistency_level", 0)
    if level:
        status += f"\n\n**字源一致性**: {_CONSISTENCY_LABELS.get(level, f'L{level}')}"

    status += "\n\n3 个版本仅石质纹理不同，文字/来源/布局完全一致。右键保存心仪版本。"

    if first["warnings"]:
        status += "\n\n" + "\n".join(f"⚠ {w}" for w in first["warnings"])

    # Gallery wants list of (image, caption) tuples
    gallery = [
        (r["image_preview"], f"版本 {r['seed']}（种子 {r['seed']}）")
        for r in results
    ]
    return gallery, status


# ── Gradio UI ────────────────────────────────────────────────

with gr.Blocks(title="方寸 · 极客禅印章生成器", theme=gr.themes.Soft()) as demo:

    gr.Markdown("# 🖋 方寸 · 极客禅印章生成器\n*Fangcun — Chinese Seal Generator for Geek-Zen*")

    with gr.Row():
        # ── Left panel: parameters ───────────────────────────
        with gr.Column(scale=1):
            mode_input = gr.Radio(
                choices=["自动选源", "我自己上传字源"],
                value="自动选源",
                label="🔀 字源模式",
                info="自动选源：从字典抓取并做同源同体匹配。上传字源：跳过抓取，用你自己的图片。",
            )

            # ── Auto mode ────────────────────────────────────
            with gr.Group(visible=True) as auto_group:
                text_input = gr.Textbox(
                    label="📝 印章文字",
                    placeholder="输入印章文字（1–4字）",
                    value="禅",
                    max_lines=1,
                )

            # ── User-upload mode ─────────────────────────────
            with gr.Group(visible=False) as upload_group:
                gr.Markdown(
                    "**上传字源**：每个字独立上传一张单字、已裁剪、对比度清晰的图片。"
                    "字序按上至下排列（印章阅读顺序）。空行会自动忽略。"
                )
                polarity_input = gr.Radio(
                    choices=list(_POLARITY_MAP.keys()),
                    value="自动检测",
                    label="🔅 图片极性",
                    info="印谱拓片（白字黑底）请选 '白字黑底'；其余默认即可。",
                )
                upload_char_inputs: list[gr.Textbox] = []
                upload_img_inputs: list[gr.Image] = []
                for i in range(4):
                    with gr.Row():
                        char_tb = gr.Textbox(
                            label=f"字 {i + 1}",
                            placeholder="单字",
                            max_lines=1,
                            scale=1,
                        )
                        img_up = gr.Image(
                            label=f"图片 {i + 1}",
                            type="pil",
                            sources=["upload"],
                            height=120,
                            scale=3,
                        )
                    upload_char_inputs.append(char_tb)
                    upload_img_inputs.append(img_up)

            def _on_mode_change(mode: str):
                auto_vis = mode == "自动选源"
                return gr.update(visible=auto_vis), gr.update(visible=not auto_vis)

            mode_input.change(
                fn=_on_mode_change,
                inputs=[mode_input],
                outputs=[auto_group, upload_group],
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
            variants_btn = gr.Button("✨ 生成 3 个版本（仅石质纹理差异）")

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

            variants_gallery = gr.Gallery(
                label="3 个版本（右键图片可下载心仪版本）",
                columns=3,
                rows=1,
                height="auto",
                preview=False,
                object_fit="contain",
            )

    # Hidden state for file paths
    transparent_path_state = gr.State(value=None)
    preview_path_state = gr.State(value=None)

    def _resolve_source(
        mode: str,
        auto_text: str,
        polarity_label: str,
        c0, c1, c2, c3,
        i0, i1, i2, i3,
    ) -> tuple[str, list[Image.Image] | None, str]:
        """Collapse the two-mode UI into (text, user_glyphs, polarity)."""
        if mode == "自动选源":
            return auto_text, None, "auto"
        text, glyphs = _collect_user_glyphs([c0, c1, c2, c3], [i0, i1, i2, i3])
        return text, glyphs, _POLARITY_MAP.get(polarity_label, "auto")

    def on_generate(
        mode, auto_text, polarity_label,
        c0, c1, c2, c3, i0, i1, i2, i3,
        shape, style, seal_type, color, grain, rotation, seed,
    ):
        text, glyphs, polarity = _resolve_source(
            mode, auto_text, polarity_label,
            c0, c1, c2, c3, i0, i1, i2, i3,
        )
        return generate_seal(
            text, shape, style, seal_type, color, grain, rotation, seed,
            user_glyphs=glyphs, user_glyph_polarity=polarity,
        )

    generate_btn.click(
        fn=on_generate,
        inputs=[
            mode_input,
            text_input,
            polarity_input,
            upload_char_inputs[0], upload_char_inputs[1],
            upload_char_inputs[2], upload_char_inputs[3],
            upload_img_inputs[0], upload_img_inputs[1],
            upload_img_inputs[2], upload_img_inputs[3],
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

    def on_generate_variants(
        mode, auto_text, polarity_label,
        c0, c1, c2, c3, i0, i1, i2, i3,
        shape, style, seal_type, color, grain, rotation,
    ):
        text, glyphs, polarity = _resolve_source(
            mode, auto_text, polarity_label,
            c0, c1, c2, c3, i0, i1, i2, i3,
        )
        # Inline to thread user_glyphs through; mirrors `generate_variants()`
        # but builds the params dict locally so both modes work.
        shape_code = _SHAPE_MAP.get(shape, "oval")
        style_code = _STYLE_MAP.get(style, "baiwen")
        seal_type_code = _TYPE_MAP.get(seal_type, "leisure")
        try:
            results = _gen.generate_variants(
                text=text.strip(),
                n=3,
                shape=shape_code,
                style=style_code,
                seal_type=seal_type_code,
                color=color,
                grain=grain,
                rotation=rotation,
                seeds=[1, 2, 3],
                user_glyphs=glyphs,
                user_glyph_polarity=polarity,
            )
        except Exception as exc:
            logging.exception("Variant generation failed")
            return None, f"生成失败: {exc}"

        first = results[0]
        status = (
            f"⚠ 字体已降级 → 使用: {first['font_used']}"
            if first["font_fallback"]
            else f"✓ 使用: {first['font_used']}"
        )
        level = first.get("consistency_level", 0)
        if level:
            status += (
                f"\n\n**字源一致性**: "
                f"{_CONSISTENCY_LABELS.get(level, f'L{level}')}"
            )
        status += (
            "\n\n3 个版本仅石质纹理不同，文字/来源/布局完全一致。右键保存心仪版本。"
        )
        if first["warnings"]:
            status += "\n\n" + "\n".join(f"⚠ {w}" for w in first["warnings"])

        gallery = [
            (r["image_preview"], f"版本 {r['seed']}（种子 {r['seed']}）")
            for r in results
        ]
        return gallery, status

    variants_btn.click(
        fn=on_generate_variants,
        inputs=[
            mode_input,
            text_input,
            polarity_input,
            upload_char_inputs[0], upload_char_inputs[1],
            upload_char_inputs[2], upload_char_inputs[3],
            upload_img_inputs[0], upload_img_inputs[1],
            upload_img_inputs[2], upload_img_inputs[3],
            shape_input,
            style_input,
            seal_type_input,
            color_input,
            grain_input,
            rotation_input,
        ],
        outputs=[
            variants_gallery,
            status_output,
        ],
    )


if __name__ == "__main__":
    is_hf_space = os.environ.get("SPACE_ID") is not None
    demo.launch(server_name="0.0.0.0" if is_hf_space else "127.0.0.1")
