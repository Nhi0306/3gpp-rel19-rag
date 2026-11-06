#!/usr/bin/env python3
"""3GPP Rel-19 pipeline - điểm vào CLI.

Các giai đoạn (mỗi giai đoạn là một module trong `rel19/`):

    manifest  - crawl status report + FTP listings -> index/manifest.json + CSV/XLSX
    download  - tải toàn bộ zip Rel-19 (series 21..38)
    extract   - giải nén .docx
    parse     - docx -> JSON (sections/blocks) + markdown
    chunk     - cắt theo heading rồi theo token -> JSONL
    embed     - embed chunk và upsert lên Qdrant
    stats     - in thống kê tiến độ

Mọi giai đoạn đều resume được: output đã có sẽ được bỏ qua, trừ khi có --force.

Ví dụ:
    python scripts/3gpp_rel19_pipeline.py --jobs 5 parse
    python scripts/3gpp_rel19_pipeline.py chunk
    python scripts/3gpp_rel19_pipeline.py embed
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CFG  # noqa: E402
from rel19 import chunk, download, embed, extract, manifest, parse, stats, update  # noqa: E402
from rel19.paths import CHUNKS, DOCS, INDEX, PARSED, REL19_ROOT, TEXT  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="3GPP Rel-19 pipeline")
    p.add_argument("--force", action="store_true", help="rebuild / re-download everything")
    p.add_argument("--limit", type=int, default=0, help="limit number of files (download stage)")
    p.add_argument("--series", type=int, nargs="*", help="only these series, e.g. 23 29")
    p.add_argument("--jobs", type=int, default=1, help="parallel worker processes for parse")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("manifest", "download", "extract", "parse", "chunk", "embed", "stats", "all"):
        sub.add_parser(name)
    up = sub.add_parser("update", help="kiem tra du lieu moi va cap nhat Qdrant")
    up.add_argument("--specs", nargs="*", help="chi refresh cac spec nay, vd 24.229-70")
    return p


def main() -> None:
    args = build_parser().parse_args()

    logging.basicConfig(
        level=getattr(logging, CFG["LOG_LEVEL"], logging.INFO),
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    for d in (REL19_ROOT, DOCS, PARSED, TEXT, CHUNKS, INDEX):
        d.mkdir(parents=True, exist_ok=True)

    if args.cmd in ("manifest", "all"):
        manifest.cmd_manifest(args)
    if args.cmd in ("download", "all"):
        download.cmd_download(args)
    if args.cmd in ("extract", "all"):
        extract.cmd_extract(args)
    if args.cmd in ("parse", "all"):
        parse.cmd_parse(args)
    if args.cmd in ("chunk", "all"):
        chunk.cmd_chunk(args)
    if args.cmd in ("embed", "all"):
        embed.cmd_embed(args)
    if args.cmd == "update":
        update.cmd_update(args)
    if args.cmd == "stats":
        stats.cmd_stats(args)


if __name__ == "__main__":
    main()
