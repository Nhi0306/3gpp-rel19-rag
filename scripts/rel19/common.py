"""Tiện ích dùng chung: HTTP, JSON, log, đọc tên file 3GPP."""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from pathlib import Path

import requests

from config import CFG

log = logging.getLogger("3gpp")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# --- HTTP ------------------------------------------------------------------
_tls = threading.local()


def session() -> requests.Session:
    s = getattr(_tls, "session", None)
    if s is None:
        s = requests.Session()
        s.headers.update(HEADERS)
        _tls.session = s
    return s


def http_get(url: str, *, timeout: int | None = None, retries: int = 4) -> requests.Response:
    timeout = timeout or CFG["HTTP_TIMEOUT"]
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = session().get(url, timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.5 * (attempt + 1))
            continue
        if r.status_code == 200:
            return r
        if r.status_code in (429, 500, 502, 503, 504):
            last = requests.HTTPError(f"HTTP {r.status_code}")
            time.sleep(1.5 * (attempt + 1))
            continue
        r.raise_for_status()  # 4xx -> fail fast
    raise RuntimeError(f"GET {url} failed: {last}")


def cached_text(url: str, path: Path, force: bool = False) -> str:
    if path.exists() and not force and path.stat().st_size > 1000:
        return path.read_text(encoding="utf-8", errors="replace")
    path.parent.mkdir(parents=True, exist_ok=True)
    txt = http_get(url).text
    path.write_text(txt, encoding="utf-8")
    log.info("cached %s (%d KB)", url, len(txt) // 1024)
    return txt


# --- JSON ------------------------------------------------------------------
def jdump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def load_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


# --- tên file 3GPP ---------------------------------------------------------
FILENAME_RE = re.compile(
    r"^(?P<a>\d{2})(?P<b>\d{3})"
    r"(?:-(?P<part>\d{1,2})(?:-(?P<part2>\d{1,2}))?)?"
    r"-(?P<rel>[0-9a-z])(?P<minor>[0-9a-z])(?P<patch>[0-9a-z])\.zip$",
    re.I,
)


def base36(ch: str) -> int:
    return int(ch, 36)


def parse_filename(name: str) -> dict | None:
    """`23501-j90.zip` -> {'spec': '23.501', 'version': '19.9.0'}."""
    m = FILENAME_RE.match(name)
    if not m:
        return None
    if m.group("rel").lower() != "j":
        return None
    spec = f"{m.group('a')}.{m.group('b')}"
    if m.group("part"):
        spec += f"-{m.group('part')}"
        if m.group("part2"):
            spec += f"-{m.group('part2')}"
    version = f"19.{base36(m.group('minor'))}.{base36(m.group('patch'))}"
    return {"spec": spec, "version": version}


def parse_size(raw: str) -> int:
    m = re.match(r"([\d.,]+)\s*(B|KB|MB|GB)\s*$", raw.strip(), re.I)
    if not m:
        return 0
    num, unit = m.group(1), m.group(2).upper()
    if "," in num:
        num = num.replace(".", "").replace(",", ".")
    try:
        val = float(num)
    except ValueError:
        return 0
    return int(val * {"B": 1, "KB": 1024, "MB": 1024 ** 2, "GB": 1024 ** 3}[unit])
