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
  type (int)    : 2 (default)
  font (str)    : "楷" | "行" | "草" | "隶" | "篆"
  author (str)  : filter by calligrapher, "" = all
  orderby (str) : "hot" for popularity sort
  strict (int)  : 1 = exact match
  loaded (int)  : pagination offset (page size = 120)
  _plat, _channel, _brand, _token : metadata fields
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

logger = logging.getLogger(__name__)

CACHE_DIR = Path.home() / ".seal_gen" / "cache"
API_BASE = "https://api.ygsf.com/v2.4"
AES_KEY = b"PkT!ihpN^QkQ62k%"

_FONT_SEARCH_PATHS = [
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallback.ttf",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "C:\\Windows\\Fonts\\simsun.ttc",
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

    try:
        result = subprocess.run(
            ["fc-list", ":lang=zh", "-f", "%{file}\n"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            _system_font_path = result.stdout.strip().split("\n")[0]
            return _system_font_path
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return None


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

    def fetch_char_image(
        self, char: str, seal_type: str
    ) -> tuple[Image.Image, str, bool]:
        """
        Fetch calligraphy image for one character.

        Returns:
            (image, font_name_used, was_fallback)
            was_fallback is True if the font is not the first-priority choice.
        """
        priority = self.FONT_PRIORITY.get(seal_type, ["篆", "隶", "楷"])

        for idx, font_style in enumerate(priority):
            cached = self._load_cache(char, font_style)
            if cached is not None:
                logger.info("Cache hit: '%s' in %s", char, font_style)
                return cached, font_style, idx > 0

            img = self._fetch_from_web(char, font_style)
            if img is not None:
                self._save_cache(char, font_style, img)
                return img, font_style, idx > 0

            logger.warning("'%s' not found in %s, trying next font...", char, font_style)

        logger.warning("All web fonts exhausted for '%s', using local fallback", char)
        img = self._render_local_fallback(char)
        return img, f"本地字体(兜底)", True

    # ── web fetch ────────────────────────────────────────────

    def _fetch_from_web(self, char: str, font_style: str) -> Optional[Image.Image]:
        """Query ygsf API, download first suitable clear_image. Returns PIL Image or None."""
        params = {
            "key": char,
            "kind": 1,
            "type": 2,
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
                    return None

                glyph_list = data.get("data", {}).get("list", [])
                if not glyph_list:
                    return None

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

        return None

    def _download_best_image(self, glyph_list: list[dict]) -> Optional[Image.Image]:
        """Try to download the first clear_image with resolution >= 100px."""
        for glyph in glyph_list:
            img_url = glyph.get("_clear_image", "")
            if not img_url:
                continue

            # Remove CDN processing for full resolution
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
                if img.width < 100 or img.height < 100:
                    logger.debug("Image too small (%dx%d), skipping", img.width, img.height)
                    continue

                return img

            except (requests.RequestException, OSError) as exc:
                logger.debug("Image download failed: %s", exc)
                continue

        return None

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
    def _cache_path(char: str, font_style: str) -> Path:
        return CACHE_DIR / f"{char}_{font_style}.png"

    def _load_cache(self, char: str, font_style: str) -> Optional[Image.Image]:
        path = self._cache_path(char, font_style)
        if path.exists():
            try:
                img = Image.open(path)
                img.load()
                return img
            except (OSError, IOError):
                path.unlink(missing_ok=True)
        return None

    @staticmethod
    def _save_cache(char: str, font_style: str, img: Image.Image) -> None:
        path = CalligraphyScraper._cache_path(char, font_style)
        try:
            img.save(path, "PNG")
        except (OSError, IOError) as exc:
            logger.warning("Cache write failed: %s", exc)
