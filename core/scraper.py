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
import json
import logging
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

CACHE_DIR = Path.home() / ".seal_gen" / "cache"
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


# ── Main class ───────────────────────────────────────────────


class CalligraphyScraper:
    """Fetch calligraphy character images from ygsf.com with local font fallback."""

    FONT_PRIORITY: dict[str, list[str]] = {
        "leisure": ["篆", "隶", "楷"],
        "name": ["隶", "楷"],
    }

    def __init__(self) -> None:
        self._session = requests.Session()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def fetch_chars_consistent(
        self, text: str, seal_type: str
    ) -> tuple[list[Image.Image], str, bool, list[str], list[str], list[str]]:
        """
        Fetch images for ALL characters, enforcing same-style consistency.

        金石学原则：同一方印章内所有字必须同一书体。

        Returns:
            (images, font_used, was_fallback, tab_sources, source_names, warnings)
        """
        priority = self.FONT_PRIORITY.get(seal_type, ["篆", "隶", "楷"])
        warnings: list[str] = []

        # ── Pass 1: find a style that covers ALL characters ──
        for idx, font_style in enumerate(priority):
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

            if all_found:
                if idx > 0:
                    warnings.append(
                        f"首选{priority[0]}中部分字缺失，全部统一使用{font_style}"
                    )

                # ── Try source unification (同源同体) ────────
                # Round 1: n=5 candidates per char
                all_cands = {
                    char: self._fetch_all_candidates(char, font_style)
                    for char in text
                }

                # Level 1: fully unified source
                unified = self._try_unified_source_from_candidates(
                    text, font_style, all_cands
                )
                if unified is not None:
                    u_images, u_tabs, u_srcs, u_source = unified
                    logger.info("统一来源: %s (%d字)", u_source, len(text))
                    warnings.append(f"统一来源: {u_source}")
                    return u_images, font_style, idx > 0, u_tabs, u_srcs, warnings

                # Round 2: widen pool to n=10 and retry
                logger.info("n=5 无统一来源, 扩大至 n=10 重试")
                all_cands_wide = {
                    char: self._fetch_all_candidates(char, font_style, n=10)
                    for char in text
                }

                unified = self._try_unified_source_from_candidates(
                    text, font_style, all_cands_wide
                )
                if unified is not None:
                    u_images, u_tabs, u_srcs, u_source = unified
                    logger.info("统一来源(wide): %s (%d字)", u_source, len(text))
                    warnings.append(f"统一来源: {u_source}")
                    return u_images, font_style, idx > 0, u_tabs, u_srcs, warnings

                # Level 2: majority source from wide pool
                maj = self._majority_source_fallback(text, all_cands_wide)
                if maj is not None:
                    m_images, m_tabs, m_srcs, m_source, fb_chars = maj
                    logger.info(
                        "多数来源: %s (%d/%d字)",
                        m_source, len(text) - len(fb_chars), len(text),
                    )
                    if fb_chars:
                        warnings.append(
                            f"主来源 {m_source}，「{'」「'.join(fb_chars)}」使用次优来源"
                        )
                    else:
                        warnings.append(f"统一来源: {m_source}")
                    return m_images, font_style, idx > 0, m_tabs, m_srcs, warnings

                # Level 3: minimum style loss fallback
                msl = self._min_style_loss_fallback(text, all_cands_wide)
                if msl is not None:
                    s_images, s_tabs, s_srcs, s_source, s_fb = msl
                    if s_fb:
                        warnings.append(
                            f"最小损失来源 {s_source}，「{'」「'.join(s_fb)}」使用次优"
                        )
                    else:
                        warnings.append(f"统一来源: {s_source}")
                    return s_images, font_style, idx > 0, s_tabs, s_srcs, warnings

                # Level 4: per-char best (last resort)
                unique_sources = list(dict.fromkeys(src_names))
                if len(unique_sources) > 1:
                    warnings.append(
                        f"无统一来源，各字分别取最优: {', '.join(unique_sources)}"
                    )
                return images, font_style, idx > 0, tabs, src_names, warnings

        # ── Pass 2: no single style covers all — find best coverage ──
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
        return images, best_style, True, tabs, src_names, warnings

    # ── unified source selection ────────────────────────────

    def _try_unified_source_from_candidates(
        self,
        text: str,
        font_style: str,
        all_cands: dict[str, list[tuple[Image.Image, float, str, str]]],
    ) -> Optional[tuple[list[Image.Image], list[str], list[str], str]]:
        """Find a single source covering ALL characters. Returns tuple or None."""
        # Group by source per character
        char_by_source: dict[str, dict[str, tuple[Image.Image, float, str]]] = {}
        for char in text:
            by_src: dict[str, tuple[Image.Image, float, str]] = {}
            for img, score, src, tab in all_cands.get(char, []):
                if src not in by_src:
                    by_src[src] = (img, score, tab)
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

        # Best common source by average score
        best_source = max(
            common,
            key=lambda src: sum(char_by_source[c][src][1] for c in text) / len(text),
        )

        images, tabs_, src_names = [], [], []
        for char in text:
            img, score, tab = char_by_source[char][best_source]
            images.append(img)
            tabs_.append(tab)
            src_names.append(best_source)
            self._save_cache(char, font_style, tab, img, best_source)

        return images, tabs_, src_names, best_source

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

    # ── cache-then-web helper ───────────────────────────────

    def _get_or_fetch(
        self, char: str, font_style: str
    ) -> tuple[Optional[Image.Image], str, str]:
        """
        Try each tab (字典→真迹) via cache then web.
        Also tries traditional form (苏→蘇) if original not found.

        Returns: (image, tab_source, source_name) or (None, '', '')
        """
        chars_to_try = [char]
        trad = _to_traditional(char)
        if trad is not None:
            chars_to_try.append(trad)

        for try_char in chars_to_try:
            for tab_name, tab_type in TAB_PRIORITY:
                # Cache check (keyed by original char)
                cached = self._load_cache(char, font_style, tab_name)
                if cached is not None:
                    src_name = self._load_cache_meta(char, font_style, tab_name)
                    logger.info("Cache hit: '%s' in %s/%s", char, font_style, tab_name)
                    return cached, tab_name, src_name

                # Web fetch — returns (image, source_name)
                img, src_name = self._fetch_from_web(try_char, font_style, tab_type)
                if img is not None:
                    self._save_cache(char, font_style, tab_name, img, src_name)
                    if try_char != char:
                        logger.info(
                            "Fetched '%s' (繁体'%s') in %s/%s from=%s",
                            char, try_char, font_style, tab_name, src_name,
                        )
                    else:
                        logger.info(
                            "Fetched '%s' in %s/%s from=%s",
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
        """Query ygsf API, return raw glyph list (no image download)."""
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
                    return []

                return data.get("data", {}).get("list", [])

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

    def _fetch_from_web(
        self, char: str, font_style: str, tab_type: int
    ) -> tuple[Optional[Image.Image], str]:
        """Query API + download best image. Returns (img, source_name) or (None, '')."""
        glyph_list = self._query_glyph_list(char, font_style, tab_type)
        if not glyph_list:
            return None, ""
        return self._download_best_image(glyph_list)

    def _fetch_all_candidates(
        self, char: str, font_style: str, n: int = 5
    ) -> list[tuple[Image.Image, float, str, str]]:
        """
        Fetch up to n scored candidates for a char across tabs.
        Returns [(img, score, source_name, tab), ...] sorted by score desc.
        Used for unified source selection.
        """
        all_candidates: list[tuple[Image.Image, float, str, str]] = []

        # Try traditional form too
        chars_to_try = [char]
        trad = _to_traditional(char)
        if trad is not None:
            chars_to_try.append(trad)

        for try_char in chars_to_try:
            for tab_name, tab_type in TAB_PRIORITY:
                glyph_list = self._query_glyph_list(try_char, font_style, tab_type)
                if not glyph_list:
                    continue
                scored = self._download_scored_candidates(glyph_list, max_n=n)
                for img, score, src in scored:
                    if score > 0:
                        all_candidates.append((img, score, src, tab_name))

            if all_candidates:
                break

        all_candidates.sort(key=lambda c: c[1], reverse=True)
        return all_candidates

    # ── image selection (top-N scoring) ────────────────────

    _MAX_CANDIDATES = 5
    _MIN_RESOLUTION = 150

    def _download_scored_candidates(
        self, glyph_list: list[dict], max_n: int = 5
    ) -> list[tuple[Image.Image, float, str]]:
        """Download up to max_n candidates, score each. Returns [(img, score, source_name), ...]."""
        candidates: list[tuple[Image.Image, float, str]] = []

        for glyph in glyph_list:
            if len(candidates) >= max_n:
                break

            img_url = glyph.get("_clear_image", "")
            if not img_url:
                continue

            if "x-bce-process=" in img_url:
                img_url = img_url.split("?")[0]

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

                src = glyph.get("_from", "?")
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
        _INFERIOR_STYLE_SOURCES = {
            "常用金文书法字典",
            "汉语古文字字形表",
            "睡虎地秦简文字编",
            "马王堆简帛",
            "马王堆帛书书法大字典",
        }
        if source_name in _INFERIOR_STYLE_SOURCES:
            base = max(0.0, base - 25.0)

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
        path = CalligraphyScraper._cache_path(char, font_style, tab_name)
        try:
            img.save(path, "PNG")
            if source_name:
                meta = CalligraphyScraper._cache_meta_path(char, font_style, tab_name)
                meta.write_text(source_name, encoding="utf-8")
        except (OSError, IOError) as exc:
            logger.warning("Cache write failed: %s", exc)
