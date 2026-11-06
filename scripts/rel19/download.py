"""Bước 2: tải toàn bộ file zip Rel-19 (series 21..38)."""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from config import CFG
from rel19.common import jdump, load_json, log, session
from rel19.paths import INDEX, MANIFEST


def download_one(rec: dict, force: bool) -> tuple[str, dict, str]:
    dst = Path(rec["local_zip"])
    if dst.exists() and not force:
        if rec.get("size_bytes") and dst.stat().st_size == rec["size_bytes"]:
            return "skip", rec, ""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".zip.part")
    try:
        with session().get(rec["url"], stream=True, timeout=CFG["HTTP_TIMEOUT"]) as r:
            r.raise_for_status()
            expected = int(r.headers.get("Content-Length") or 0)
            written = 0
            with tmp.open("wb") as fh:
                for block in r.iter_content(1 << 16):
                    fh.write(block)
                    written += len(block)
        if expected and written != expected:
            raise IOError(f"size mismatch {written} != {expected}")
        os.replace(tmp, dst)
        return "ok", rec, ""
    except Exception as exc:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        return "fail", rec, str(exc)


def cmd_download(args) -> None:
    files = load_json(MANIFEST, [])
    if not files:
        raise SystemExit("run `manifest` first")
    if args.series:
        files = [f for f in files if f["series"] in args.series]
    if args.limit:
        files = files[: args.limit]

    ok = skip = fail = 0
    failed: list[dict] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=CFG["DOWNLOAD_WORKERS"]) as pool:
        futures = {pool.submit(download_one, f, args.force): f for f in files}
        for i, fut in enumerate(as_completed(futures), 1):
            status, rec, err = fut.result()
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                fail += 1
                failed.append({"file": rec["filename"], "url": rec["url"], "error": err})
                log.warning("FAIL %s: %s", rec["filename"], err)
            if i % 50 == 0 or i == len(files):
                log.info("[%d/%d] ok=%d skip=%d fail=%d %.0fs", i, len(files), ok, skip, fail, time.time() - t0)

    jdump(INDEX / "failed-downloads.json", failed)
    log.info("download done: ok=%d skip=%d fail=%d in %.0fs", ok, skip, fail, time.time() - t0)
