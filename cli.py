"""
印章生成器 — CLI 批量入口

Usage:
    # Single seal
    python cli.py --text "禅" --shape oval --style baiwen --type leisure

    # Batch (one word per line in file)
    python cli.py --batch chars.txt --shape oval --style baiwen --type leisure --output-dir ./seals/

chars.txt format:
    禅
    苏轼
    极客禅
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from core import SealGenerator, cache_info, clear_cache
from core.errors import SourceInconsistencyError

console = Console()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="极客禅 · 印章生成器 CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input mode (mutually exclusive; not required when using --cache-info/--clear-cache)
    group = p.add_mutually_exclusive_group(required=False)
    group.add_argument("--text", type=str, help="印章文字（1–4字）")
    group.add_argument("--batch", type=str, help="批量文件路径（每行一个词）")

    # Seal parameters
    p.add_argument(
        "--shape",
        choices=["oval", "square"],
        default="oval",
        help="形制: oval=竖椭圆, square=方章 (default: oval)",
    )
    p.add_argument(
        "--style",
        choices=["baiwen", "zhuwen"],
        default="baiwen",
        help="风格: baiwen=白文, zhuwen=朱文 (default: baiwen)",
    )
    p.add_argument(
        "--type",
        choices=["leisure", "name", "brand"],
        default="leisure",
        dest="seal_type",
        help=(
            "章类: name=名章（强制篆书）, "
            "leisure=闲章（优先篆书，允许隶楷）, "
            "brand=品牌章（任何字体） (default: leisure)"
        ),
    )
    p.add_argument(
        "--color", type=str, default="#B22222", help="朱砂颜色 hex (default: #B22222)"
    )
    p.add_argument(
        "--grain", type=float, default=0.25, help="质感强度 0.0–1.0 (default: 0.25)"
    )
    p.add_argument(
        "--rotation", type=float, default=2.0, help="旋转角度° (default: 2.0)"
    )
    p.add_argument("--size", type=int, default=600, help="短边像素 (default: 600)")
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子（纹理可复现；缺省为每次随机）",
    )

    # Output
    p.add_argument(
        "--output-dir",
        type=str,
        default="./seals",
        help="输出目录 (default: ./seals)",
    )
    p.add_argument(
        "--jobs",
        type=int,
        default=1,
        help=(
            "并发 worker 数 (default: 1, 仅批量模式生效). "
            "推荐值: 2-4. 注意：每个 worker 独立发起上游请求，并发过高易触发限流"
        ),
    )
    p.add_argument(
        "--format",
        choices=["png", "svg"],
        default="png",
        help=(
            "输出格式 (default: png). SVG 为矢量格式，无石质纹理，"
            "适合印刷/矢量编辑 (Illustrator / Inkscape)"
        ),
    )

    # Cache control
    p.add_argument(
        "--no-api-cache",
        action="store_true",
        help="跳过 API/图片缓存，强制网络请求",
    )
    p.add_argument(
        "--clear-cache",
        action="store_true",
        help="清除全部缓存后退出",
    )
    p.add_argument(
        "--cache-info",
        action="store_true",
        help="显示缓存统计后退出",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="启用详细调试日志（候选列表、打分细节）",
    )
    p.add_argument(
        "--strict-consistency",
        action="store_true",
        help="严格一致性模式：仅接受 Level 1-2 统一来源",
    )
    p.add_argument(
        "--debug-extract",
        action="store_true",
        help="保存最后一个字的提取中间步骤至 {output_dir}/{text}_debug/",
    )
    p.add_argument(
        "--debug-layout", action="store_true",
        help="保存版面布局调试图（蓝=cell，红=ink bbox，绿=centroid）至 {output_dir}/{text}_layout.png",
    )

    # User-provided glyphs (bypass scraper)
    p.add_argument(
        "--user-glyph",
        action="append",
        metavar="CHAR=PATH",
        help=(
            "用自己的图片作为字源（跳过自动抓取）。"
            "格式 CHAR=PATH，可重复。例如 --user-glyph 然=./ran.png --user-glyph 苏=./su.png。"
            "字序以 --text 为准；未指定 --text 时按出现顺序拼接。"
        ),
    )
    p.add_argument(
        "--polarity",
        choices=["auto", "black_on_white", "white_on_black"],
        default="auto",
        help=(
            "用户上传图片的极性提示 (default: auto)。"
            "印谱拓片（白字黑底）请用 white_on_black。"
        ),
    )

    return p.parse_args()


def _generate_one(
    gen: SealGenerator,
    text: str,
    args: argparse.Namespace,
    output_dir: Path,
) -> bool:
    """Generate and save one seal. Returns True on success."""
    try:
        if args.format == "svg":
            return _generate_one_svg(gen, text, args, output_dir)

        gen.set_extract_debug_dir(
            output_dir / f"{text}_debug" if args.debug_extract else None
        )

        # When --debug-layout is set, generate() returns the overlay alongside
        # the seal from the SAME prepare pass — avoids a second scraper+extract
        # round-trip compared to calling render_layout_debug separately.
        user_glyphs = getattr(args, "_user_glyphs_for_text", {}).get(text)

        result = gen.generate(
            text=text,
            shape=args.shape,
            style=args.style,
            seal_type=args.seal_type,
            color=args.color,
            grain=args.grain,
            rotation=args.rotation,
            size=args.size,
            seed=args.seed,
            return_debug=args.debug_layout,
            user_glyphs=user_glyphs,
            user_glyph_polarity=getattr(args, "polarity", "auto"),
        )

        if args.strict_consistency and result.get("consistency_level", 0) > 2:
            raise SourceInconsistencyError(text, result["consistency_level"])

        # Save transparent PNG
        filename = f"{text}_{args.style}_{args.shape}.png"
        out_path = output_dir / filename
        result["image_transparent"].save(out_path, "PNG")

        font_info = result["font_used"]
        if result["font_fallback"]:
            console.print(f"  [yellow]⚠ 降级[/yellow] → {font_info}")
        else:
            console.print(f"  [green]✓[/green] 字体: {font_info}")

        for w in result["warnings"]:
            console.print(f"  [yellow]⚠ {w}[/yellow]")

        console.print(f"  [dim]→ {out_path}[/dim]")

        if args.debug_layout and "image_layout_debug" in result:
            debug_path = output_dir / f"{text}_layout.png"
            result["image_layout_debug"].save(debug_path, "PNG")
            console.print(f"  [dim]→ layout debug: {debug_path}[/dim]")

        return True

    except Exception as exc:
        console.print(f"  [red]✗ 生成失败: {exc}[/red]")
        logging.exception("Generation failed for '%s'", text)
        return False


def _generate_one_svg(
    gen: SealGenerator,
    text: str,
    args: argparse.Namespace,
    output_dir: Path,
) -> bool:
    """Generate and save one SVG seal. Returns True on success.

    SVG output intentionally skips texture, rotation, and preview. If the
    user mixed --format svg with raster-only flags, warn but continue.
    """
    if args.seed is not None:
        console.print(
            "  [yellow]⚠ --seed 对 SVG 无效（矢量输出无纹理）[/yellow]"
        )
    if args.debug_extract or args.debug_layout:
        console.print(
            "  [yellow]⚠ --debug-extract / --debug-layout 对 SVG 无效[/yellow]"
        )

    result = gen.generate_svg(
        text=text,
        shape=args.shape,
        style=args.style,
        seal_type=args.seal_type,
        color=args.color,
        size=args.size,
    )

    if args.strict_consistency and result.get("consistency_level", 0) > 2:
        raise SourceInconsistencyError(text, result["consistency_level"])

    filename = f"{text}_{args.style}_{args.shape}.svg"
    out_path = output_dir / filename
    out_path.write_text(result["svg"], encoding="utf-8")

    font_info = result["font_used"]
    if result["font_fallback"]:
        console.print(f"  [yellow]⚠ 降级[/yellow] → {font_info}")
    else:
        console.print(f"  [green]✓[/green] 字体: {font_info}")

    for w in result["warnings"]:
        console.print(f"  [yellow]⚠ {w}[/yellow]")

    console.print(f"  [dim]→ SVG saved: {out_path}[/dim]")
    return True


def main() -> None:
    args = _parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_time=False, show_path=False)],
    )

    # ── Cache management commands ────────────────────────────
    if args.cache_info:
        info = cache_info()
        console.print("[bold]缓存统计[/bold]")
        console.print(f"  API 正缓存: {info['api_positive']} 条")
        console.print(f"  API 负缓存: {info['api_negative']} 条")
        console.print(f"  图片缓存:   {info['img_cached']} 张")
        console.print(f"  总占用:     {info['total_bytes'] / 1024:.1f} KB")
        return

    if args.clear_cache:
        n = clear_cache()
        console.print(f"[green]已清除 {n} 个缓存文件[/green]")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Each worker thread gets its own SealGenerator — the scraper keeps
    # per-call state (_last_consistency_level, _current_seal_type) that
    # would race under shared access. HTTP sessions are cheap to re-create.
    _tls = threading.local()

    def _get_gen() -> SealGenerator:
        if not hasattr(_tls, "gen"):
            _tls.gen = SealGenerator(no_api_cache=args.no_api_cache)
        return _tls.gen

    # ── Parse --user-glyph flags (optional) ───────────────────
    # Maps each text → list[PIL.Image] in char order, attached to args so
    # _generate_one picks it up. Batch mode doesn't support --user-glyph
    # (one glyph set per invocation).
    args._user_glyphs_for_text = {}
    if args.user_glyph:
        if args.batch:
            console.print(
                "[red]--user-glyph 不支持 --batch 模式（每次调用只能提供一组字源）[/red]"
            )
            sys.exit(1)

        from PIL import Image as _PILImage
        glyph_map: dict[str, _PILImage.Image] = {}
        for spec in args.user_glyph:
            if "=" not in spec:
                console.print(
                    f"[red]--user-glyph 需 CHAR=PATH 格式，收到: {spec}[/red]"
                )
                sys.exit(1)
            char, path = spec.split("=", 1)
            if len(char) != 1:
                console.print(
                    f"[red]--user-glyph CHAR 必须是单字，收到: '{char}'[/red]"
                )
                sys.exit(1)
            path_obj = Path(path)
            if not path_obj.exists():
                console.print(f"[red]图片不存在: {path_obj}[/red]")
                sys.exit(1)
            glyph_map[char] = _PILImage.open(path_obj)

        # Derive or validate --text against the glyph map
        if args.text:
            missing = [c for c in args.text if c not in glyph_map]
            if missing:
                console.print(
                    f"[red]缺少 --user-glyph: {missing}[/red]"
                )
                sys.exit(1)
            resolved_text = args.text
        else:
            # No --text: use the order the user passed --user-glyph in
            resolved_text = "".join(glyph_map.keys())
            args.text = resolved_text

        args._user_glyphs_for_text[resolved_text] = [
            glyph_map[c] for c in resolved_text
        ]

    if not args.text and not args.batch:
        console.print("[red]请指定 --text 或 --batch[/red]")
        sys.exit(1)

    if args.text:
        texts = [args.text]
    else:
        batch_path = Path(args.batch)
        if not batch_path.exists():
            console.print(f"[red]文件不存在: {batch_path}[/red]")
            sys.exit(1)
        texts = [
            line.strip()
            for line in batch_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    if not texts:
        console.print("[red]无有效输入文字[/red]")
        sys.exit(1)

    # Clamp --jobs to something sane and to single-item batches.
    jobs = max(1, args.jobs)
    if len(texts) == 1:
        jobs = 1

    console.print(f"\n[bold]极客禅 · 印章生成器[/bold]")
    suffix = f" (并发 {jobs} worker)" if jobs > 1 else ""
    console.print(f"共 {len(texts)} 枚印章待生成{suffix}\n")

    success = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("生成中...", total=len(texts))

        if jobs == 1:
            for text in texts:
                progress.update(task, description=f"[cyan]{text}[/cyan]")
                if _generate_one(_get_gen(), text, args, output_dir):
                    success += 1
                progress.advance(task)
        else:
            def _worker(text: str) -> tuple[str, bool]:
                return text, _generate_one(_get_gen(), text, args, output_dir)

            with ThreadPoolExecutor(max_workers=jobs) as pool:
                futures = [pool.submit(_worker, t) for t in texts]
                for fut in as_completed(futures):
                    text, ok = fut.result()
                    progress.update(task, description=f"[cyan]{text}[/cyan]")
                    if ok:
                        success += 1
                    progress.advance(task)

    console.print(f"\n[bold green]完成:[/bold green] {success}/{len(texts)} 枚成功")
    console.print(f"[dim]输出目录: {output_dir.resolve()}[/dim]\n")


if __name__ == "__main__":
    main()
