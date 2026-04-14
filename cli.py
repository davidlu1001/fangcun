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

    return p.parse_args()


def _generate_one(
    gen: SealGenerator,
    text: str,
    args: argparse.Namespace,
    output_dir: Path,
) -> bool:
    """Generate and save one seal. Returns True on success."""
    try:
        gen.set_extract_debug_dir(
            output_dir / f"{text}_debug" if args.debug_extract else None
        )

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

        if args.debug_layout:
            debug_overlay = gen.render_layout_debug(
                text=text,
                shape=args.shape,
                style=args.style,
                seal_type=args.seal_type,
                size=args.size,
            )
            debug_path = output_dir / f"{text}_layout.png"
            debug_overlay.save(debug_path, "PNG")
            console.print(f"  [dim]→ layout debug: {debug_path}[/dim]")

        return True

    except Exception as exc:
        console.print(f"  [red]✗ 生成失败: {exc}[/red]")
        logging.exception("Generation failed for '%s'", text)
        return False


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

    gen = SealGenerator(no_api_cache=args.no_api_cache)

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

    console.print(f"\n[bold]极客禅 · 印章生成器[/bold]")
    console.print(f"共 {len(texts)} 枚印章待生成\n")

    success = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("生成中...", total=len(texts))

        for text in texts:
            progress.update(task, description=f"[cyan]{text}[/cyan]")
            if _generate_one(gen, text, args, output_dir):
                success += 1
            progress.advance(task)

    console.print(f"\n[bold green]完成:[/bold green] {success}/{len(texts)} 枚成功")
    console.print(f"[dim]输出目录: {output_dir.resolve()}[/dim]\n")


if __name__ == "__main__":
    main()
