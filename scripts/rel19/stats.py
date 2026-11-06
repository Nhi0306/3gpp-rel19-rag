"""Thống kê nhanh tiến độ từng bước."""
from __future__ import annotations

from pathlib import Path

from rel19.common import load_json
from rel19.paths import CHUNKS_FILE, MANIFEST, PARSED


def cmd_stats(args) -> None:
    files = load_json(MANIFEST, [])
    print(f"manifest entries : {len(files)}")
    if not files:
        return
    print(f"series           : {min(f['series'] for f in files)}..{max(f['series'] for f in files)}")
    print(f"total size       : {sum(f['size_bytes'] for f in files) / 1024 ** 3:.2f} GB")
    print(f"downloaded       : {sum(1 for f in files if Path(f['local_zip']).exists())}")
    print(f"extracted        : {sum(1 for f in files if Path(f['local_docx']).exists())}")
    parsed = sum(
        1
        for f in files
        if (PARSED / f"{f['series']}_series" / (Path(f["filename"]).stem + ".json")).exists()
    )
    print(f"parsed           : {parsed}")
    if CHUNKS_FILE.exists():
        with CHUNKS_FILE.open("r", encoding="utf-8") as fh:
            print(f"chunks           : {sum(1 for _ in fh)}")
