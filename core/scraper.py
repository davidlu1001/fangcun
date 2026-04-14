"""
Website scraper for ygsf.com calligraphy character images.

API Discovery Notes (from DevTools analysis):
─────────────────────────────────────────────
Endpoint:  POST https://api.ygsf.com/v2.4/glyph/query
Protocol:  AES-128-ECB encrypted JSON payloads (request & response)
Key:       PkT!ihpN^QkQ62k% (16 bytes, from ed08 JS module)
Encoding:  base64 with custom substitution: + → -, / → _, = → !
Body:      application/x-www-form-urlencoded, field "p" = encrypted params
Image CDN: https://ygsf.cdn.bcebos.com/
  _clear_image → clean B&W version (ideal for extraction)
  _color_image → color/rubbing version
  Remove ?x-bce-process=... suffix for full resolution

Params for /glyph/query:
  key (str)     : character to search, e.g. "永"
  kind (int)    : 1 = brush calligraphy
  type (int)    : tab selector — 3=字典, 2=真迹, 1=字库
  font (str)    : "楷" | "行" | "草" | "隶" | "篆"
  author (str)  : filter by calligrapher, "" = all
  orderby (str) : "hot" for popularity sort
  strict (int)  : 1 = exact match
  loaded (int)  : pagination offset (page size = 120)
  _plat, _channel, _brand, _token : metadata fields

Tab priority (per 金石学 standards):
  字典 (type=3) — dictionary/seal-reference entries, clean B&W, ideal for seals
  真迹 (type=2) — original calligraphy scans, noisier, needs enhanced denoising
  字库 (type=1) — digital font glyphs, NEVER used (vector edges unusable)
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import random
import subprocess
import time
from io import BytesIO
from pathlib import Path

import cv2
from typing import Optional

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

try:
    from Crypto.Cipher import AES
    from Crypto.Util.Padding import pad, unpad
except ImportError:
    from Cryptodome.Cipher import AES
    from Cryptodome.Util.Padding import pad, unpad

try:
    from fake_useragent import UserAgent

    _ua = UserAgent(fallback="Mozilla/5.0")
except Exception:
    _ua = None

try:
    from opencc import OpenCC

    _s2t = OpenCC("s2t")
except Exception:
    _s2t = None

logger = logging.getLogger(__name__)

# HF Spaces: home dir may not be writable; use /tmp fallback
_cache_base = os.environ.get("SEAL_CACHE_DIR", "")
if _cache_base:
    CACHE_DIR = Path(_cache_base)
elif os.access(str(Path.home()), os.W_OK):
    CACHE_DIR = Path.home() / ".seal_gen" / "cache"
else:
    CACHE_DIR = Path("/tmp") / ".seal_gen" / "cache"
API_CACHE_DIR = CACHE_DIR / "_api"   # JSON responses from _query_glyph_list
IMG_CACHE_DIR = CACHE_DIR / "_img"   # Downloaded candidate images from CDN

# Cache TTL: glyph data is near-static; negative caches are shorter
_POSITIVE_TTL_DAYS = 30
_NEGATIVE_TTL_DAYS = 7

# ── 装饰性字源（鸟虫篆/玉箸篆等花体字）────────────────────
# 笔画盘绕扭曲成鸟形/虫形，设计目的是"美观但难辨识"。
# 来自一次实战 bug："卢修齐"名章中"卢"字只有 1 个鸟虫篆候选，
# 89.4 分高分独占，导致整枚名章气质分裂。
#
# 按 seal_type 分层处理：
#   name:    完全排除（_fetch_all_candidates 级别跳过）
#   leisure: _score_image 大幅降权 (-40)，能用但排末位
#   brand:   无限制（装饰性是加分项）
DECORATIVE_SOURCES: frozenset[str] = frozenset({
    "鸟虫篆全书",
})

# ── 主流汉印/印谱字源偏好（R10）────────────────────────
# 单字/重字印章 tie-break 时优先选这些正统字源。
# 不含个人书法家字典（赵之谦、吴昌硕等）——它们带书法个性
# 和飞白，更适合闲章但不是单字印的默认偏好。
PREFERRED_INSCRIPTION_SOURCES: frozenset[str] = frozenset({
    "中国篆刻大字典",
    "汉印文字征",
    "汉印文字征七、八",
    "汉印文字征五、六",
    "汉印文字征九、十",
    "汉印分韵",
    "汉印分韵续集",
    "汉印分韵选集",
    "印谱大字典",
    "篆字汇",
    "六书通",
})

API_BASE = "https://api.ygsf.com/v2.4"
AES_KEY = b"PkT!ihpN^QkQ62k%"

# Tab priority: 字典 first (clean, seal-optimized), 真迹 second (original, noisier).
# 字库 (type=1) is deliberately excluded — digital font glyphs are unusable.
TAB_PRIORITY: list[tuple[str, int]] = [
    ("字典", 3),  # dictionary / seal-reference entries
    ("真迹", 2),  # original calligraphy scans
]

# Fallback font search order: serif (宋体) before sans-serif (黑体).
# Sans-serif looks too "digital" for seal context.
_FONT_SEARCH_PATHS = [
    # Serif / Song — preferred for seal fallback (more traditional feel)
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/opentype/adobe/SourceHanSerif-Regular.ttc",
    "/usr/share/fonts/truetype/source-han-serif/SourceHanSerifSC-Regular.otf",
    "C:\\Windows\\Fonts\\simsun.ttc",
    "/System/Library/Fonts/Songti.ttc",
    # Sans-serif fallback (still better than nothing)
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallback.ttf",
    "/System/Library/Fonts/PingFang.ttc",
    "C:\\Windows\\Fonts\\msyh.ttc",
]

_system_font_path: Optional[str] = None


def _find_system_font() -> Optional[str]:
    """Find a usable Chinese font on the system (cached)."""
    global _system_font_path
    if _system_font_path is not None:
        return _system_font_path

    for p in _FONT_SEARCH_PATHS:
        if Path(p).exists():
            _system_font_path = p
            return p

    # Prefer serif fonts from fc-list (Serif/Song/Ming before Sans)
    try:
        result = subprocess.run(
            ["fc-list", ":lang=zh", "-f", "%{file}\n"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            fonts = result.stdout.strip().split("\n")
            # Prefer serif/song fonts
            for f in fonts:
                fl = f.lower()
                if any(k in fl for k in ("serif", "song", "ming", "宋")):
                    _system_font_path = f
                    return f
            _system_font_path = fonts[0]
            return _system_font_path
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return None


def _to_traditional(char: str) -> Optional[str]:
    """Convert simplified Chinese character to traditional form, or None if same."""
    if _s2t is None:
        return None
    trad = _s2t.convert(char)
    return trad if trad != char else None


def _random_ua() -> str:
    if _ua is not None:
        try:
            return _ua.random
        except Exception:
            pass
    return "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


# ── AES helpers ──────────────────────────────────────────────


def _encrypt_params(params: dict) -> str:
    """Encrypt params dict → custom-base64 string for ygsf API."""
    plaintext = json.dumps(params, ensure_ascii=False).encode("utf-8")
    cipher = AES.new(AES_KEY, AES.MODE_ECB)
    ciphertext = cipher.encrypt(pad(plaintext, AES.block_size))
    b64 = base64.b64encode(ciphertext).decode("ascii")
    return b64.replace("+", "-").replace("/", "_").replace("=", "!")


def _decrypt_response(encrypted_text: str) -> dict:
    """Decrypt ygsf API response → dict."""
    b64 = encrypted_text.replace("-", "+").replace("_", "/").replace("!", "=")
    ciphertext = base64.b64decode(b64)
    cipher = AES.new(AES_KEY, AES.MODE_ECB)
    plaintext = unpad(cipher.decrypt(ciphertext), AES.block_size)
    return json.loads(plaintext.decode("utf-8"))


# ── API / image cache helpers ───────────────────────────────


def _api_cache_key(params: dict) -> str:
    """Stable hash from params dict (pre-encryption)."""
    blob = json.dumps(params, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.md5(blob).hexdigest()


def _api_cache_path(params: dict) -> Path:
    return API_CACHE_DIR / f"{_api_cache_key(params)}.json"


def _img_cache_path(img_url: str) -> Path:
    h = hashlib.md5(img_url.encode()).hexdigest()[:16]
    return IMG_CACHE_DIR / f"{h}.png"


def _img_meta_path(img_url: str) -> Path:
    h = hashlib.md5(img_url.encode()).hexdigest()[:16]
    return IMG_CACHE_DIR / f"{h}.meta"


def _is_cache_fresh(path: Path, ttl_days: int) -> bool:
    """Check if a cache file exists and is within TTL."""
    if not path.exists():
        return False
    age_s = time.time() - path.stat().st_mtime
    return age_s < ttl_days * 86400


def cache_info() -> dict[str, int]:
    """Return cache statistics: counts and total disk usage."""
    api_pos = api_neg = img_count = 0
    total_bytes = 0

    if API_CACHE_DIR.exists():
        for f in API_CACHE_DIR.iterdir():
            if f.suffix != ".json":
                continue
            total_bytes += f.stat().st_size
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if data.get("_empty"):
                    api_neg += 1
                else:
                    api_pos += 1
            except (json.JSONDecodeError, OSError):
                pass

    if IMG_CACHE_DIR.exists():
        for f in IMG_CACHE_DIR.iterdir():
            if f.suffix == ".png":
                img_count += 1
                total_bytes += f.stat().st_size

    return {
        "api_positive": api_pos,
        "api_negative": api_neg,
        "img_cached": img_count,
        "total_bytes": total_bytes,
    }


def clear_cache() -> int:
    """Remove all cache files. Returns number of files removed."""
    count = 0
    for d in (API_CACHE_DIR, IMG_CACHE_DIR):
        if d.exists():
            for f in d.iterdir():
                try:
                    f.unlink()
                    count += 1
                except OSError:
                    pass
    return count


# ── Main class ───────────────────────────────────────────────


class CalligraphyScraper:
    """Fetch calligraphy character images from ygsf.com with local font fallback."""

    # 印章类型决定字体优先级和降级策略。
    #
    # 传统金石学里，印章类型和字体选择有严格对应关系：
    # - 名章（姓名印、落款印）：必须篆书。"篆刻"二字本身就暗示了这一点。
    #   名章绝不降级到隶楷，即便字源不同也要保证字体纯粹。
    # - 闲章（书斋印、座右铭印、引首章）：以篆书为主，
    #   但允许降级到隶楷（某些文人有用隶楷的先例）。
    # - 品牌章（现代商业、社交媒体、内容创作者）：最灵活，
    #   允许任何字体，以视觉效果为优先。
    #
    # fetch_chars_consistent 对 "name" 类型启用严格字体模式，
    # 即便篆书统一源找不到也不降级，而是调用
    # _force_assemble_single_font 在篆书内部强制组装。
    FONT_PRIORITY: dict[str, list[str]] = {
        "name":    ["篆"],
        "leisure": ["篆", "隶", "楷"],
        "brand":   ["篆", "隶", "楷"],
    }

    def __init__(self, no_api_cache: bool = False) -> None:
        self._session = requests.Session()
        self._no_api_cache = no_api_cache
        self._current_seal_type = "leisure"  # set by fetch_chars_consistent
        # Consistency level of the most recent fetch_chars_consistent call.
        # 1 = unified source in n=5 pool (or single/repeated char short-circuit)
        # 2 = unified source in n=10 pool
        # 3 = majority source fallback (>50% coverage from one source)
        # 4 = minimum-style-loss fallback
        # 5 = per-char best / multi-font assembly / strict force-assemble
        # 0 = not yet populated
        self._last_consistency_level: int = 0
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        API_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        IMG_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def fetch_chars_consistent(
        self, text: str, seal_type: str
    ) -> tuple[list[Image.Image], str, bool, list[str], list[str], list[str]]:
        """
        Deferred state machine for font+source selection.

        Key principle: prefer lower-priority font with unified source over
        higher-priority font with mixed sources (隶书同源 > 篆书异源).

        For seal_type="name", strict font mode prevents degradation to
        non-篆 fonts. Even if no unified source exists, the result stays
        in 篆 (via _force_assemble_single_font).

        Returns:
            (images, font_used, was_fallback, tab_sources, source_names, warnings)
        """
        priority = self.FONT_PRIORITY.get(seal_type, ["篆", "隶", "楷"])
        strict_font = (seal_type == "name")
        exclude_deco = (seal_type == "name")
        self._current_seal_type = seal_type
        # Reset to 0 so an exception mid-call doesn't leave stale state from
        # the previous invocation visible to the next reader.
        self._last_consistency_level = 0
        warnings: list[str] = []
        best_fallback = None  # "chars found but no unified source" backup
        best_fallback_level = 0  # 3 = majority, 4 = min-loss

        # ── R9+R10: single/repeated char short-circuit ──────
        # When all chars are the same (禅, 朝朝), skip Pass 2 unified
        # source check entirely. Use _fetch_all_candidates with preferred
        # source tie-break (R10): within ±5 score, prefer 汉印 orthodox
        # sources over calligrapher personal dictionaries.
        unique_chars = set(text)
        if len(unique_chars) == 1:
            char = text[0]
            for font_style in priority:
                cands = self._fetch_all_candidates(
                    char, font_style, n=15,
                    exclude_decorative=exclude_deco,
                )
                if not cands:
                    continue

                top_img, top_score, top_src, top_tab = cands[0]
                selected = cands[0]

                # R10 tie-break: prefer PREFERRED source within ±5 of top
                for c in cands:
                    if c[2] in PREFERRED_INSCRIPTION_SOURCES and c[1] >= top_score - 5.0:
                        selected = c
                        break

                sel_img, sel_score, sel_src, sel_tab = selected
                if sel_src != top_src:
                    logger.info(
                        "[R10] 单字偏好 tie-break: '%s' top=%s(%.1f) → %s(%.1f)",
                        text, top_src, top_score, sel_src, sel_score,
                    )

                images = [sel_img] * len(text)
                tabs = [sel_tab] * len(text)
                src_names = [sel_src] * len(text)
                logger.info(
                    "[R9+R10] 单字短路: text='%s' font=%s src=%s",
                    text, font_style, sel_src,
                )
                self._last_consistency_level = 1
                return self._log_final(
                    text, (images, font_style, False, tabs, src_names, [])
                )
            logger.warning("[R9+R10] 单字短路失败: '%s' 进入 Pass 2", text)

        for idx, font_style in enumerate(priority):
            # Step 1: verify font covers all chars (fast, uses cache)
            images: list[Image.Image] = []
            tabs: list[str] = []
            src_names: list[str] = []
            all_found = True

            for char in text:
                img, tab, src_name = self._get_or_fetch(char, font_style)
                if img is None:
                    all_found = False
                    logger.info("'%s' not available in %s", char, font_style)
                    break
                images.append(img)
                tabs.append(tab)
                src_names.append(src_name)

            if not all_found:
                continue  # font can't cover all chars, try next

            # Step 2: try unified source (n=5, then n=10)
            all_cands = {
                char: self._fetch_all_candidates(
                    char, font_style, exclude_decorative=exclude_deco
                )
                for char in text
            }

            unified = self._try_unified_source_from_candidates(
                text, font_style, all_cands
            )
            if unified is not None:
                u_images, u_tabs, u_srcs, u_source = unified
                logger.info("统一来源: %s (%s/%d字)", u_source, font_style, len(text))
                warnings.append(f"统一来源: {u_source}")
                if idx > 0:
                    warnings.append(f"首选{priority[0]}降级至{font_style}")
                self._last_consistency_level = 1
                return self._log_final(text, (u_images, font_style, idx > 0, u_tabs, u_srcs, warnings))

            logger.debug("n=5 无统一来源, 扩大至 n=10")
            all_cands_wide = {
                char: self._fetch_all_candidates(
                    char, font_style, n=10, exclude_decorative=exclude_deco
                )
                for char in text
            }

            unified = self._try_unified_source_from_candidates(
                text, font_style, all_cands_wide
            )
            if unified is not None:
                u_images, u_tabs, u_srcs, u_source = unified
                logger.info("统一来源(wide): %s (%s/%d字)", u_source, font_style, len(text))
                warnings.append(f"统一来源: {u_source}")
                if idx > 0:
                    warnings.append(f"首选{priority[0]}降级至{font_style}")
                self._last_consistency_level = 2
                return self._log_final(text, (u_images, font_style, idx > 0, u_tabs, u_srcs, warnings))

            # ★ Deferred: chars found but no unified source → save as fallback
            # and continue to next font (隶书同源 > 篆书异源)
            logger.debug(
                "%s 无统一来源, 记录备胎, 继续尝试降级字体", font_style
            )

            if best_fallback is None:
                # Build best non-unified result for this font
                maj = self._majority_source_fallback(text, all_cands_wide)
                if maj is not None:
                    m_images, m_tabs, m_srcs, m_source, fb_chars = maj
                    fb_warnings = list(warnings)
                    if fb_chars:
                        fb_warnings.append(
                            f"主来源 {m_source}，「{'」「'.join(fb_chars)}」使用次优"
                        )
                    else:
                        fb_warnings.append(f"统一来源: {m_source}")
                    best_fallback = (
                        m_images, font_style, idx > 0, m_tabs, m_srcs, fb_warnings
                    )
                    best_fallback_level = 3
                else:
                    msl = self._min_style_loss_fallback(text, all_cands_wide)
                    if msl is not None:
                        s_images, s_tabs, s_srcs, s_source, s_fb = msl
                        fb_warnings = list(warnings)
                        if s_fb:
                            fb_warnings.append(
                                f"最小损失来源 {s_source}，「{'」「'.join(s_fb)}」使用次优"
                            )
                        best_fallback = (
                            s_images, font_style, idx > 0, s_tabs, s_srcs, fb_warnings
                        )
                        best_fallback_level = 4

            # continue to next font — don't return!

        # All fonts tried, no perfect unified source
        if best_fallback is not None:
            logger.warning("所有字体无完美统一来源, 启用最优备胎: %s", best_fallback[1])
            # best_fallback_level was set alongside best_fallback (3 or 4)
            self._last_consistency_level = best_fallback_level or 4
            return self._log_final(text, best_fallback)

        # ── Pass 2: no single style covers all — find best coverage ──

        # Strict font mode (name): never degrade across fonts
        if strict_font:
            logger.warning(
                "严格字体模式 (%s): 在 %s 内部强制组装", seal_type, priority
            )
            # Per-char-internal assembly within a single font — sources may
            # differ per char, so this is level 5.
            self._last_consistency_level = 5
            return self._log_final(text, self._force_assemble_single_font(text, priority[0]))

        logger.warning("No single font style covers all chars in '%s'", text)

        # Build availability matrix
        availability: dict[str, dict[str, tuple[Optional[Image.Image], str, str]]] = {}
        for font_style in priority:
            availability[font_style] = {}
            for char in text:
                availability[font_style][char] = self._get_or_fetch(char, font_style)

        # Pick the style that covers the most characters
        best_style = priority[0]
        best_count = 0
        for font_style in priority:
            count = sum(
                1 for img, _, _ in availability[font_style].values() if img is not None
            )
            if count > best_count:
                best_count = count
                best_style = font_style

        # Assemble: best_style where possible, fill gaps
        images, tabs, src_names = [], [], []
        mixed_chars: list[str] = []

        for char in text:
            img, tab, src_name = availability[best_style][char]
            if img is not None:
                images.append(img)
                tabs.append(tab)
                src_names.append(src_name)
                continue

            found = False
            for alt_style in priority:
                if alt_style == best_style:
                    continue
                alt_img, alt_tab, alt_src = availability[alt_style][char]
                if alt_img is not None:
                    images.append(alt_img)
                    tabs.append(alt_tab)
                    src_names.append(alt_src)
                    mixed_chars.append(f"'{char}'→{alt_style}")
                    found = True
                    break

            if not found:
                images.append(self._render_local_fallback(char))
                tabs.append("本地")
                src_names.append("")
                mixed_chars.append(f"'{char}'→本地字体")

        warnings.append(
            f"书体不统一：主体{best_style}，但 {', '.join(mixed_chars)}"
        )
        # Multi-font assembly: worst case — different scripts, different sources
        self._last_consistency_level = 5
        return self._log_final(text, (images, best_style, True, tabs, src_names, warnings))

    # ── unified source selection ────────────────────────────

    def _try_unified_source_from_candidates(
        self,
        text: str,
        font_style: str,
        all_cands: dict[str, list[tuple[Image.Image, float, str, str]]],
    ) -> Optional[tuple[list[Image.Image], list[str], list[str], str]]:
        """Find a single source covering ALL characters. Returns tuple or None.

        R12: When a source has multiple variants for a char (e.g. 中国篆刻大字典
        has both thin and thick 知), pick the variant whose relative stroke
        width best matches the sibling median. This prevents visual imbalance
        when siblings share a source but differ in line weight (知足 case).
        """
        # Group ALL variants per source per char (score-desc order preserved)
        char_by_source: dict[
            str, dict[str, list[tuple[Image.Image, float, str]]]
        ] = {}
        for char in text:
            by_src: dict[str, list[tuple[Image.Image, float, str]]] = {}
            for img, score, src, tab in all_cands.get(char, []):
                by_src.setdefault(src, []).append((img, score, tab))
            if not by_src:
                return None
            char_by_source[char] = by_src

        # Intersection of sources across all characters
        source_sets = [set(cs.keys()) for cs in char_by_source.values()]
        common = source_sets[0]
        for s in source_sets[1:]:
            common &= s

        if not common:
            logger.info("无统一来源 (字: %s)", ", ".join(text))
            return None

        # Best common source by average top-variant score
        best_source = max(
            common,
            key=lambda src: sum(
                char_by_source[c][src][0][1] for c in text
            ) / len(text),
        )

        # ── R12: stroke-width sibling tie-break ─────────────────
        # When a char has multiple variants within ±5 score of its top
        # (ambiguous: e.g. 知 has both thin sw=22 and thick sw=88 in
        # 中国篆刻大字典), pick the variant whose relative stroke width best
        # matches an anchor computed from UNAMBIGUOUS siblings. Unambiguous
        # = only one eligible variant in best_source, so its stroke width
        # is a trustworthy anchor. If all siblings are ambiguous, fall
        # back to median of top variants.
        eligible_per_char: list[list[tuple[Image.Image, float, str]]] = []
        for char in text:
            variants = char_by_source[char][best_source]
            top_score = variants[0][1]
            elig = [v for v in variants if v[1] >= top_score - 5.0]
            eligible_per_char.append(elig)

        anchor_sws = [
            self._relative_stroke_width(elig[0][0])
            for elig in eligible_per_char
            if len(elig) == 1
        ]
        if not anchor_sws:
            anchor_sws = [
                self._relative_stroke_width(elig[0][0])
                for elig in eligible_per_char
            ]
        target_sw = float(np.median(anchor_sws)) if anchor_sws else 0.0

        images, tabs_, src_names = [], [], []
        for char, eligible in zip(text, eligible_per_char):
            if len(eligible) > 1 and target_sw > 0:
                chosen = min(
                    eligible,
                    key=lambda v: abs(
                        self._relative_stroke_width(v[0]) - target_sw
                    ),
                )
                if chosen is not eligible[0]:
                    logger.info(
                        "[R12] 笔画匹配 '%s' (%s): top rel_sw=%.3f → 选 rel_sw=%.3f (target=%.3f)",
                        char, best_source,
                        self._relative_stroke_width(eligible[0][0]),
                        self._relative_stroke_width(chosen[0]),
                        target_sw,
                    )
            else:
                chosen = eligible[0]

            img, score, tab = chosen
            images.append(img)
            tabs_.append(tab)
            src_names.append(best_source)
            self._save_cache(char, font_style, tab, img, best_source)

        # Per-char stroke deviation summary (only for multi-char + non-zero target)
        if target_sw > 0 and len(text) > 1:
            deviations = []
            for char, img in zip(text, images):
                char_sw = self._relative_stroke_width(img)
                dev = abs(char_sw - target_sw) / target_sw
                deviations.append((char, char_sw, dev))
                if dev > 0.20:
                    logger.warning(
                        "[R12] STROKE_DEVIATION '%s': rel_sw=%.3f target=%.3f deviation=%.0f%%",
                        char, char_sw, target_sw, dev * 100,
                    )
            logger.info(
                "[R12] 笔画匹配总结 source=%s target_sw=%.3f: %s",
                best_source, target_sw,
                ", ".join(f"'{c}'={sw:.3f}({d:.0%})" for c, sw, d in deviations),
            )

        return images, tabs_, src_names, best_source

    @staticmethod
    def _relative_stroke_width(img: Image.Image) -> float:
        """Stroke width as a fraction of the image short side.

        Uses distance-transform p70 × 2 on dark pixels (strokes are dark in
        raw scraper images). Normalizing by short side makes the metric
        resolution-independent so thin/thick variants can be compared across
        candidates that arrive at different pixel dimensions.
        """
        arr = np.array(img.convert("L"))
        binary = (arr < 128).astype(np.uint8)
        if binary.sum() < 10:
            return 0.0
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        vals = dist[binary > 0]
        if len(vals) == 0:
            return 0.0
        sw = float(np.percentile(vals, 70)) * 2.0
        short_side = min(img.width, img.height)
        if short_side <= 0:
            return 0.0
        # Clip pathological overflow (seen on tiny/degenerate masks)
        return min(sw, float(short_side)) / float(short_side)

    def _majority_source_fallback(
        self,
        text: str,
        all_cands: dict[str, list[tuple[Image.Image, float, str, str]]],
    ) -> Optional[tuple[list[Image.Image], list[str], list[str], str, list[str]]]:
        """
        Pick the source covering the most characters (majority vote).
        Tie-break by average score. Returns (images, tabs, src_names, source, fallback_chars).
        """
        # Build coverage map: {source: [(char, score, img, tab), ...]}
        source_cov: dict[str, list[tuple[str, float, Image.Image, str]]] = {}
        for char in text:
            seen: set[str] = set()
            for img, score, src, tab in all_cands.get(char, []):
                if src not in seen:
                    source_cov.setdefault(src, []).append((char, score, img, tab))
                    seen.add(src)

        if not source_cov:
            return None

        # Rank: coverage count first, average score second
        def _rank(src: str) -> tuple[int, float]:
            entries = source_cov[src]
            return (len({c for c, *_ in entries}), sum(s for _, s, *_ in entries) / len(entries))

        majority_src = max(source_cov, key=_rank)
        covered = {c: (img, tab) for c, _, img, tab in source_cov[majority_src]}

        # If majority source covers at most half, defer to min-style-loss
        if len(covered) <= len(text) // 2:
            logger.warning(
                "多数来源 %s 覆盖率不足 50%% (%d/%d), 退化至最小损失",
                majority_src, len(covered), len(text),
            )
            return None

        logger.info(
            "多数来源: %s 覆盖 %d/%d 字",
            majority_src, len(covered), len(text),
        )

        images, tabs_, src_names, fb_chars = [], [], [], []
        for char in text:
            if char in covered:
                img, tab = covered[char]
                images.append(img)
                tabs_.append(tab)
                src_names.append(majority_src)
            else:
                # Fallback: best available from any source
                cands = all_cands.get(char, [])
                if cands:
                    img, _, src, tab = cands[0]
                    images.append(img)
                    tabs_.append(tab)
                    src_names.append(src)
                else:
                    images.append(self._render_local_fallback(char))
                    tabs_.append("本地")
                    src_names.append("")
                fb_chars.append(char)
                logger.warning("主来源 %s 无「%s」, 回退次优", majority_src, char)

        return images, tabs_, src_names, majority_src, fb_chars

    # ── minimum style loss fallback ─────────────────────────

    def _min_style_loss_fallback(
        self,
        text: str,
        all_cands: dict[str, list[tuple[Image.Image, float, str, str]]],
    ) -> Optional[tuple[list[Image.Image], list[str], list[str], str, list[str]]]:
        """
        When no source covers >50%: find the source with minimum total
        score loss compared to per-char best. Prioritizes coverage count,
        then minimizes quality sacrifice.

        Ranking: coverage_count × 100 − total_score_loss
        """
        # Build per-source stats
        source_info: dict[str, dict] = {}  # {src: {chars: {char: (img, score, tab)}}}

        for char in text:
            seen: set[str] = set()
            for img, score, src, tab in all_cands.get(char, []):
                if src not in seen:
                    if src not in source_info:
                        source_info[src] = {"chars": {}}
                    source_info[src]["chars"][char] = (img, score, tab)
                    seen.add(src)

        if not source_info:
            return None

        # Per-char best scores for loss calculation
        best_scores = {}
        for char in text:
            cands = all_cands.get(char, [])
            best_scores[char] = cands[0][1] if cands else 0.0

        # Rank sources: coverage × 100 − total loss
        def _rank(src: str) -> float:
            chars_map = source_info[src]["chars"]
            coverage = len(chars_map)
            total_loss = sum(
                best_scores[c] - chars_map[c][1]
                for c in chars_map
            )
            return coverage * 100.0 - total_loss

        best_src = max(source_info, key=_rank)
        covered_chars = source_info[best_src]["chars"]

        logger.info(
            "最小损失来源: %s 覆盖 %d/%d 字",
            best_src, len(covered_chars), len(text),
        )

        images, tabs_, src_names, fb_chars = [], [], [], []
        for char in text:
            if char in covered_chars:
                img, score, tab = covered_chars[char]
                images.append(img)
                tabs_.append(tab)
                src_names.append(best_src)
            else:
                cands = all_cands.get(char, [])
                if cands:
                    img, _, src, tab = cands[0]
                    images.append(img)
                    tabs_.append(tab)
                    src_names.append(src)
                else:
                    images.append(self._render_local_fallback(char))
                    tabs_.append("本地")
                    src_names.append("")
                fb_chars.append(char)

        return images, tabs_, src_names, best_src, fb_chars

    # ── log helper ──────────────────────────────────────────

    @staticmethod
    def _log_final(
        text: str,
        result: tuple[list[Image.Image], str, bool, list[str], list[str], list[str]],
    ) -> tuple[list[Image.Image], str, bool, list[str], list[str], list[str]]:
        """Log final source decision before returning from fetch_chars_consistent."""
        _, font_style, _, _, src_names, _ = result
        summary = "、".join(
            f"'{c}'→{src or '本地'}" for c, src in zip(text, src_names)
        )
        logger.info("[Final] 最终字源: %s (font=%s)", summary, font_style)
        return result

    # ── strict font force-assembly ───────────────────────────

    def _force_assemble_single_font(
        self, text: str, font_style: str
    ) -> tuple[list[Image.Image], str, bool, list[str], list[str], list[str]]:
        """Force-assemble all chars in a single font, even if sources differ.

        Used by strict font mode (name seals): better to have mixed-source
        篆書 than to degrade to 隸書/楷書.
        """
        images: list[Image.Image] = []
        tabs: list[str] = []
        src_names: list[str] = []
        warnings: list[str] = []
        sources_used: list[str] = []

        for char in text:
            img, tab, src_name = self._get_or_fetch(char, font_style)
            if img is None:
                logger.warning("强制组装: '%s' 在 %s 中不可得, 本地兜底", char, font_style)
                images.append(self._render_local_fallback(char))
                tabs.append("本地")
                src_names.append("")
            else:
                images.append(img)
                tabs.append(tab)
                src_names.append(src_name)
                if src_name:
                    sources_used.append(src_name)

        unique = set(sources_used)
        if len(unique) > 1:
            warnings.append(f"严格字体({font_style})：{len(unique)} 个不同字源")
        elif len(unique) == 1:
            warnings.append(f"统一来源: {list(unique)[0]}")

        return images, font_style, False, tabs, src_names, warnings

    # ── cache-then-web helper ───────────────────────────────

    def _get_or_fetch(
        self, char: str, font_style: str
    ) -> tuple[Optional[Image.Image], str, str]:
        """
        Try each tab (字典→真迹) via cache then web.

        R8-B: Traditional-first — tries 蘇 before 苏, 齊 before 齐.
        Cache key uses original char (simplified) for stability.

        Returns: (image, tab_source, source_name) or (None, '', '')
        """
        # R8-B: 繁優先 — traditional form first for all seal types
        trad = _to_traditional(char)
        chars_to_try: list[str] = []
        if trad is not None:
            chars_to_try.append(trad)
        chars_to_try.append(char)

        for try_char in chars_to_try:
            for tab_name, tab_type in TAB_PRIORITY:
                # Cache check (keyed by original char for stability)
                cached = self._load_cache(char, font_style, tab_name)
                if cached is not None:
                    src_name = self._load_cache_meta(char, font_style, tab_name)
                    logger.info("[Pass1] Cache hit: '%s' in %s/%s", char, font_style, tab_name)
                    return cached, tab_name, src_name

                # Web fetch — returns (image, source_name)
                img, src_name = self._fetch_from_web(try_char, font_style, tab_type)
                if img is not None:
                    self._save_cache(char, font_style, tab_name, img, src_name)
                    if try_char != char:
                        logger.info(
                            "[Pass1] Fetched '%s' (繁体'%s') in %s/%s from=%s",
                            char, try_char, font_style, tab_name, src_name,
                        )
                    else:
                        logger.info(
                            "[Pass1] Fetched '%s' in %s/%s from=%s",
                            char, font_style, tab_name, src_name,
                        )
                    return img, tab_name, src_name

            if try_char != char:
                logger.debug("Traditional '%s' also not found in %s", try_char, font_style)

        return None, "", ""

    # ── web fetch ────────────────────────────────────────────

    def _query_glyph_list(
        self, char: str, font_style: str, tab_type: int
    ) -> list[dict]:
        """Query ygsf API, return raw glyph list (no image download).

        Cache layer (inserted before sleep/encrypt for zero-latency hits):
          - Positive cache: 30-day TTL (glyph data rarely changes)
          - Negative cache: 7-day TTL (cold chars won't suddenly appear)
          - Key: MD5 of stable params dict (not encrypted blob)
        """
        params = {
            "key": char,
            "kind": 1,
            "type": tab_type,
            "font": font_style,
            "author": "",
            "orderby": "hot",
            "strict": 1,
            "loaded": 0,
            "_plat": "web",
            "_channel": "pc",
            "_brand": "",
            "_token": "",
        }

        # ── Cache check (before sleep/encrypt) ──────────────
        if not self._no_api_cache:
            cp = _api_cache_path(params)
            pos_fresh = _is_cache_fresh(cp, _POSITIVE_TTL_DAYS)
            neg_fresh = _is_cache_fresh(cp, _NEGATIVE_TTL_DAYS)
            if pos_fresh or neg_fresh:
                try:
                    cached = json.loads(cp.read_text(encoding="utf-8"))
                    if cached.get("_empty"):
                        if neg_fresh:
                            logger.debug("API negative cache hit: %s/%s/tab%d", char, font_style, tab_type)
                            return []
                    else:
                        if pos_fresh:
                            logger.debug("API cache hit: %s/%s/tab%d (%d items)", char, font_style, tab_type, len(cached.get("list", [])))
                            return cached.get("list", [])
                except (json.JSONDecodeError, OSError):
                    cp.unlink(missing_ok=True)

        # ── Network fetch ───────────────────────────────────
        encrypted = _encrypt_params(params)

        for attempt in range(3):
            try:
                time.sleep(random.uniform(0.5, 2.0))

                resp = self._session.post(
                    f"{API_BASE}/glyph/query",
                    data={"p": encrypted},
                    headers={
                        "User-Agent": _random_ua(),
                        "Content-Type": "application/x-www-form-urlencoded",
                        "Origin": "https://web.ygsf.com",
                        "Referer": "https://web.ygsf.com/",
                    },
                    timeout=10,
                )
                resp.raise_for_status()

                data = _decrypt_response(resp.text)

                if data.get("stat") != 0:
                    # Negative cache: record empty result
                    self._write_api_cache(params, None)
                    return []

                glyph_list = data.get("data", {}).get("list", [])
                self._write_api_cache(params, glyph_list)
                return glyph_list

            except requests.RequestException as exc:
                wait = (2**attempt) + random.uniform(0, 1)
                logger.warning(
                    "Request failed (attempt %d/3): %s — retrying in %.1fs",
                    attempt + 1,
                    exc,
                    wait,
                )
                time.sleep(wait)

        return []

    @staticmethod
    def _write_api_cache(params: dict, glyph_list: Optional[list[dict]]) -> None:
        """Atomic write of API response cache (positive or negative)."""
        import tempfile as _tf

        cp = _api_cache_path(params)
        if glyph_list is None or len(glyph_list) == 0:
            payload = {"_empty": True, "ts": time.time()}
        else:
            payload = {"list": glyph_list, "ts": time.time()}
        tmp: Optional[str] = None
        try:
            fd, tmp = _tf.mkstemp(dir=cp.parent, suffix=".tmp")
            os.close(fd)
            Path(tmp).write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
            os.replace(tmp, cp)
        except OSError as exc:
            logger.debug("API cache write failed: %s", exc)
            if tmp is not None:
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    def _fetch_from_web(
        self, char: str, font_style: str, tab_type: int
    ) -> tuple[Optional[Image.Image], str]:
        """Query API + download best image. Returns (img, source_name) or (None, '')."""
        glyph_list = self._query_glyph_list(char, font_style, tab_type)
        if not glyph_list:
            return None, ""
        return self._download_best_image(glyph_list)

    def _fetch_all_candidates(
        self,
        char: str,
        font_style: str,
        n: int = 5,
        exclude_decorative: bool = False,
    ) -> list[tuple[Image.Image, float, str, str]]:
        """
        Fetch up to n scored candidates for a char across tabs.

        R8-B: Traditional-first strategy. Serious seal dictionaries (中国篆刻大字典,
        汉印文字征) index by traditional forms. 篆書 predates the simplified/traditional
        split by ~2000 years — traditional IS the native form.

        - name type: strict trad-first. Break after traditional candidates found.
        - leisure/brand: flexible. Query both forms, merge candidate pool.

        Returns [(img, score, source_name, tab), ...] sorted by score desc.
        """
        all_candidates: list[tuple[Image.Image, float, str, str]] = []

        # R8-B: 繁優先 — traditional form first
        trad = _to_traditional(char)
        chars_to_try: list[str] = []
        if trad is not None:
            chars_to_try.append(trad)
        chars_to_try.append(char)

        strict_trad_first = (self._current_seal_type == "name")

        for try_char in chars_to_try:
            for tab_name, tab_type in TAB_PRIORITY:
                glyph_list = self._query_glyph_list(try_char, font_style, tab_type)
                if not glyph_list:
                    continue

                # Name seals: hard-filter decorative sources at glyph level
                if exclude_decorative:
                    glyph_list = [
                        g for g in glyph_list
                        if g.get("_from", "") not in DECORATIVE_SOURCES
                    ]
                    if not glyph_list:
                        continue

                scored = self._download_scored_candidates(glyph_list, max_n=n)
                for img, score, src in scored:
                    if score > 0:
                        all_candidates.append((img, score, src, tab_name))

            if all_candidates:
                if strict_trad_first and try_char != char:
                    logger.info(
                        "[R8-B] name 严格繁优先: '%s'→'%s' 拿到 %d 候选, 跳过简体",
                        char, try_char, len(all_candidates),
                    )
                break

        all_candidates.sort(key=lambda c: c[1], reverse=True)
        return all_candidates

    # ── image selection (top-N scoring) ────────────────────

    _MAX_CANDIDATES = 5
    _MIN_RESOLUTION = 150

    def _download_scored_candidates(
        self, glyph_list: list[dict], max_n: int = 5
    ) -> list[tuple[Image.Image, float, str]]:
        """Download up to max_n candidates, score each.

        Image CDN cache: each URL → MD5-hashed .png + .meta (source_name).
        Cache hit skips HTTP GET entirely.
        Returns [(img, score, source_name), ...] sorted by score desc.
        """
        candidates: list[tuple[Image.Image, float, str]] = []

        for glyph in glyph_list:
            if len(candidates) >= max_n:
                break

            img_url = glyph.get("_clear_image", "")
            if not img_url:
                continue

            if "x-bce-process=" in img_url:
                img_url = img_url.split("?")[0]

            src = glyph.get("_from", "?")

            # ── Image CDN cache check ───────────────────────
            icp = _img_cache_path(img_url)
            if not self._no_api_cache and icp.exists():
                try:
                    img = Image.open(icp)
                    img.load()
                    # Read cached source_name if available
                    imp = _img_meta_path(img_url)
                    if imp.exists():
                        src = imp.read_text(encoding="utf-8").strip() or src

                    if min(img.width, img.height) < self._MIN_RESOLUTION:
                        continue

                    score = self._score_image(img, src)
                    candidates.append((img, score, src))
                    logger.debug(
                        "IMG cache hit: %dx%d score=%.1f from=%s",
                        img.width, img.height, score, src,
                    )
                    continue
                except (OSError, IOError):
                    icp.unlink(missing_ok=True)

            # ── HTTP download ───────────────────────────────
            try:
                img_resp = self._session.get(
                    img_url,
                    headers={"User-Agent": _random_ua()},
                    timeout=10,
                )
                img_resp.raise_for_status()

                img = Image.open(BytesIO(img_resp.content))

                if min(img.width, img.height) < self._MIN_RESOLUTION:
                    continue

                # Save to image cache (atomic)
                self._write_img_cache(img_url, img, src)

                score = self._score_image(img, src)
                candidates.append((img, score, src))
                logger.debug(
                    "Candidate: %dx%d score=%.1f from=%s",
                    img.width, img.height, score, src,
                )

            except (requests.RequestException, OSError) as exc:
                logger.debug("Image download failed: %s", exc)
                continue

        candidates.sort(key=lambda c: c[1], reverse=True)
        return candidates

    @staticmethod
    def _write_img_cache(img_url: str, img: Image.Image, src: str) -> None:
        """Atomic write of image CDN cache."""
        import tempfile as _tf

        icp = _img_cache_path(img_url)
        tmp: Optional[str] = None
        try:
            fd, tmp = _tf.mkstemp(dir=icp.parent, suffix=".tmp")
            os.close(fd)
            img.save(tmp, "PNG")
            os.replace(tmp, icp)
            # Write source metadata
            imp = _img_meta_path(img_url)
            imp.write_text(src, encoding="utf-8")
        except OSError as exc:
            logger.debug("IMG cache write failed: %s", exc)
            if tmp is not None:
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    def _download_best_image(
        self, glyph_list: list[dict]
    ) -> tuple[Optional[Image.Image], str]:
        """Download top-5, return best. Returns (image, source_name) or (None, '')."""
        candidates = self._download_scored_candidates(glyph_list)
        if not candidates:
            return None, ""
        best_img, best_score, best_src = candidates[0]
        logger.info(
            "Selected image score=%.1f from=%s (%d candidates)",
            best_score, best_src, len(candidates),
        )
        return best_img, best_src

    @staticmethod
    def _score_image(img: Image.Image, source_name: str = "") -> float:
        """
        Score a candidate image (0–100).

          Resolution  (0–30) + Contrast (0–50) + Coverage (0–20)
          - 印谱 base penalty (-10)
          - Physical fragmentation: hard reject (return 0)
            Only triggers when largest opaque block > 15% of image area
            (prevents 「八」「川」false kills — their strokes are < 5%)
        """
        gray = np.array(img.convert("L"), dtype=np.float64)
        short_side = min(img.width, img.height)

        res_score = min(short_side / 600.0, 1.0) * 30.0
        contrast_score = min(float(np.std(gray)) / 120.0, 1.0) * 50.0

        # ── Physical fragmentation hard reject (印谱 only) ────
        # Only applies to 印谱 images (opaque area has bright stroke slots).
        # Dictionary images have opaque=strokes (all dark) — skip entirely.
        # Prevents false kills on 道(7 components), 轼(4 components) etc.
        if img.mode in ("RGBA", "LA"):
            alpha_arr = np.array(img.split()[-1])
            fully_transparent = float((alpha_arr < 10).sum()) / alpha_arr.size

            if fully_transparent > 0.05:
                opaque = (alpha_arr >= 128).astype(np.uint8)
                n_labels, _, cc_stats, _ = cv2.connectedComponentsWithStats(
                    opaque, connectivity=8
                )
                if n_labels > 1:
                    cc_max = int(cc_stats[1:, cv2.CC_STAT_AREA].max())
                    total_area = img.width * img.height

                    if cc_max > total_area * 0.15:
                        # Check if this is actually 印谱 (bright slots in opaque)
                        # vs normal strokes (opaque area all dark)
                        bg = Image.new("RGB", img.size, (255, 255, 255))
                        bg.paste(img, mask=img.split()[-1])
                        comp_gray = np.array(bg.convert("L"))
                        opaque_gray = comp_gray[opaque.astype(bool)]
                        light_ratio = float(
                            (opaque_gray > 200).sum()
                        ) / max(len(opaque_gray), 1)

                        # Only 印谱: light_ratio > 0.15 (white slots in stone)
                        if light_ratio > 0.15:
                            major_blocks = sum(
                                1
                                for j in range(1, n_labels)
                                if cc_stats[j, cv2.CC_STAT_AREA] > cc_max * 0.15
                            )
                            if major_blocks >= 2:
                                logger.warning(
                                    "印谱碎裂 %d 大块 (light=%.2f), 淘汰: src=%s",
                                    major_blocks,
                                    light_ratio,
                                    source_name,
                                )
                                return 0.0

        # ── Coverage ─────────────────────────────────────────
        binary = gray < 128
        coverage = float(binary.mean())
        coverage = min(coverage, 1.0 - coverage)

        if 0.15 <= coverage <= 0.60:
            coverage_score = 20.0
        elif coverage < 0.15:
            coverage_score = (coverage / 0.15) * 20.0
        else:
            coverage_score = max(0.0, (1.0 - coverage) / 0.40) * 20.0

        base = res_score + contrast_score + coverage_score

        # ── Source penalties ──────────────────────────────────
        from .extractor import KNOWN_YINPU_SOURCES

        if source_name in KNOWN_YINPU_SOURCES:
            base = max(0.0, base - 10.0)

        # 金文/简帛 sources: archaic glyph forms unsuitable for seal carving
        # R9-P0-2: archaic/obscure sources unsuitable for seal carving.
        # Kept as -25 penalty (not hard-excluded) to preserve fallback for rare chars.
        _INFERIOR_STYLE_SOURCES = {
            # 金文/简帛系
            "常用金文书法字典",
            "汉语古文字字形表",
            "睡虎地秦简文字编",
            "马王堆简帛",
            "马王堆帛书书法大字典",
            "金文书法集萃",
            "金文名品",
            "甲骨文字典",
            # 古奥偏门系
            "楚简名品",
            "许氏说文序",
            "三坟记",
            "说文解字真本",
            "说文解字",
            "说文部目",
            "说文广义",
            "篆书千字文",
            "千字文",
        }
        if source_name in _INFERIOR_STYLE_SOURCES:
            base = max(0.0, base - 25.0)

        # 装饰性字源（鸟虫篆等）：大幅降权，排在正统篆书之后
        if source_name in DECORATIVE_SOURCES:
            base = max(0.0, base - 40.0)

        # ── 结构完整度惩罚（R8-A）─────────────────────────
        # 防止图像质量高但结构碎片化的"抽象古文字"被误判高分。
        # 合理的篆书字应有 1 个主导连通块占总墨量 > 60%。
        #
        # 阈值来自 2026-04 实测：
        #   赵之谦蝌蚪齐: max_ratio=0.29 → 重罚
        #   赵之谦三竖齐: max_ratio=0.73 → 不动
        #   中国篆刻大字典齊: max_ratio=1.00 → 不动
        binary_cc = (gray < 128).astype(np.uint8)
        ink_total = int(binary_cc.sum())

        if ink_total > 100:
            n_cc, _, cc_stats, _ = cv2.connectedComponentsWithStats(
                binary_cc, connectivity=8
            )
            if n_cc > 1:
                areas = cc_stats[1:, cv2.CC_STAT_AREA]
                if len(areas) > 0:
                    max_ratio = int(areas.max()) / ink_total
                    if max_ratio < 0.40:
                        base = max(0.0, base - 35.0)
                        logger.info(
                            "[R8-A] 结构碎片重罚 -35: max_ratio=%.2f, #cc=%d, src=%s",
                            max_ratio, n_cc - 1, source_name,
                        )
                    elif max_ratio < 0.60:
                        base = max(0.0, base - 10.0)
                        logger.debug(
                            "[R8-A] 结构碎片轻罚 -10: max_ratio=%.2f, #cc=%d, src=%s",
                            max_ratio, n_cc - 1, source_name,
                        )

        return base

    # ── local fallback ───────────────────────────────────────

    @staticmethod
    def _render_local_fallback(char: str) -> Image.Image:
        """Render character with local system Chinese font as last-resort fallback."""
        size = 300
        img = Image.new("L", (size, size), 255)
        draw = ImageDraw.Draw(img)

        font_path = _find_system_font()
        font_size = int(size * 0.72)

        if font_path:
            try:
                font = ImageFont.truetype(font_path, font_size)
            except (OSError, IOError):
                font = ImageFont.load_default()
        else:
            font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), char, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (size - tw) // 2 - bbox[0]
        y = (size - th) // 2 - bbox[1]
        draw.text((x, y), char, fill=0, font=font)

        return img

    # ── cache ────────────────────────────────────────────────

    @staticmethod
    def _cache_path(char: str, font_style: str, tab_name: str) -> Path:
        return CACHE_DIR / f"{char}_{font_style}_{tab_name}.png"

    @staticmethod
    def _cache_meta_path(char: str, font_style: str, tab_name: str) -> Path:
        return CACHE_DIR / f"{char}_{font_style}_{tab_name}.src"

    def _load_cache(
        self, char: str, font_style: str, tab_name: str
    ) -> Optional[Image.Image]:
        path = self._cache_path(char, font_style, tab_name)
        if path.exists():
            try:
                img = Image.open(path)
                img.load()
                return img
            except (OSError, IOError):
                path.unlink(missing_ok=True)
        return None

    def _load_cache_meta(
        self, char: str, font_style: str, tab_name: str
    ) -> str:
        meta = self._cache_meta_path(char, font_style, tab_name)
        if meta.exists():
            return meta.read_text(encoding="utf-8").strip()
        return ""

    @staticmethod
    def _save_cache(
        char: str, font_style: str, tab_name: str,
        img: Image.Image, source_name: str = "",
    ) -> None:
        """Atomic cache write — temp file + os.replace prevents corrupt PNGs."""
        import tempfile as _tf

        path = CalligraphyScraper._cache_path(char, font_style, tab_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp: Optional[str] = None
        try:
            fd, tmp = _tf.mkstemp(dir=path.parent, suffix=".tmp")
            os.close(fd)
            img.save(tmp, "PNG")
            os.replace(tmp, path)
            if source_name:
                meta = CalligraphyScraper._cache_meta_path(char, font_style, tab_name)
                meta.write_text(source_name, encoding="utf-8")
        except (OSError, IOError) as exc:
            logger.warning("Cache write failed: %s", exc)
            if tmp is not None:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
