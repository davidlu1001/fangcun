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

    def _fetch_from_web(
        self, char: str, font_style: str, tab_type: int
    ) -> Optional[Image.Image]:
        """Query ygsf API for one (char, font, tab). Returns PIL Image or None."""
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
                    logger.warning("API error for '%s' %s: %s", char, font_style, data)
                    return None, ""

                glyph_list = data.get("data", {}).get("list", [])
                if not glyph_list:
                    return None, ""

                return self._download_best_image(glyph_list)

            except requests.RequestException as exc:
                wait = (2**attempt) + random.uniform(0, 1)
                logger.warning(
                    "Request failed (attempt %d/3): %s — retrying in %.1fs",
                    attempt + 1,
                    exc,
                    wait,
                )
                time.sleep(wait)

        return None, ""

    # ── image selection (top-N scoring) ────────────────────

    _MAX_CANDIDATES = 5
    _MIN_RESOLUTION = 150

    def _download_best_image(
        self, glyph_list: list[dict]
    ) -> tuple[Optional[Image.Image], str]:
        """
        Download up to top-5 candidate images, score each, return the best.
        Returns (image, source_name) or (None, '').
        """
        candidates: list[tuple[Image.Image, float, str]] = []

        for glyph in glyph_list:
            if len(candidates) >= self._MAX_CANDIDATES:
                break

            img_url = glyph.get("_clear_image", "")
            if not img_url:
                continue

            # Full resolution
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

                # Hard filter: minimum resolution
                if min(img.width, img.height) < self._MIN_RESOLUTION:
                    logger.debug(
                        "Below min resolution (%dx%d), skipping",
                        img.width,
                        img.height,
                    )
                    continue

                # Score raw image — normalization is done in extractor
                score = self._score_image(img)
                src = glyph.get("_from", "?")
                candidates.append((img, score, src))
                logger.debug(
                    "Candidate: %dx%d score=%.1f from=%s",
                    img.width,
                    img.height,
                    score,
                    src,
                )

            except (requests.RequestException, OSError) as exc:
                logger.debug("Image download failed: %s", exc)
                continue

        if not candidates:
            return None, ""

        candidates.sort(key=lambda c: c[1], reverse=True)
        best_img, best_score, best_src = candidates[0]
        logger.info(
            "Selected image score=%.1f from=%s (%d candidates)",
            best_score,
            best_src,
            len(candidates),
        )
        return best_img, best_src

    @staticmethod
    def _score_image(img: Image.Image) -> float:
        """
        Score a candidate image (0–100).

          Resolution  (0–30): short-side relative to 600px (gate)
          Contrast    (0–50): grayscale std relative to 120 (main discriminator)
          Coverage    (0–20): binary stroke ratio in 15%–60% sweet spot

        Higher contrast denominator (120 vs 80) spreads scores among
        already-clean images — std=93 → 38.8 vs std=119 → 49.6.
        """
        gray = np.array(img.convert("L"), dtype=np.float64)
        short_side = min(img.width, img.height)

        # Resolution: 0–30 (gate — most images pass easily)
        res_score = min(short_side / 600.0, 1.0) * 30.0

        # Contrast: 0–50 (main quality signal for calligraphy)
        std = float(np.std(gray))
        contrast_score = min(std / 120.0, 1.0) * 50.0

        # Coverage: 0–20 (use minority color as stroke proxy —
        # works regardless of black-on-white or white-on-black)
        binary = gray < 128
        coverage = float(binary.mean())
        coverage = min(coverage, 1.0 - coverage)  # minority = stroke

        if 0.15 <= coverage <= 0.60:
            coverage_score = 20.0
        elif coverage < 0.15:
            coverage_score = (coverage / 0.15) * 20.0
        else:  # > 0.60
            coverage_score = max(0.0, (1.0 - coverage) / 0.40) * 20.0

        base_score = res_score + contrast_score + coverage_score

        # ── Alpha structure penalty ──────────────────────────
        # Complex alpha (multiple opaque fragments) = harder to extract.
        # Prefer clean single-block images over multi-fragment 印谱 scans.
        alpha_penalty = 0.0
        if img.mode == "RGBA":
            alpha_arr = np.array(img.split()[3])
            opaque_mask = (alpha_arr >= 128).astype(np.uint8)

            if opaque_mask.any():
                n_labels, _, cc_stats, _ = cv2.connectedComponentsWithStats(
                    opaque_mask, connectivity=8
                )
                if n_labels > 1:
                    cc_max = int(cc_stats[1:, cv2.CC_STAT_AREA].max())
                    fragment_count = sum(
                        1
                        for j in range(1, n_labels)
                        if cc_stats[j, cv2.CC_STAT_AREA] > cc_max * 0.05
                    )
                    if fragment_count == 2:
                        alpha_penalty = 15.0
                    elif fragment_count >= 3:
                        alpha_penalty = 35.0

        return max(0.0, base_score - alpha_penalty)

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
