#!/usr/bin/env python3
"""Xay sparse BM25 vector cho RAG (collection rieng, chay nen, resume duoc).

    python scripts\\rag_sparse_index.py init      # tao collection sparse + payload index
    python scripts\\rag_sparse_index.py stats     # quet JSONL -> bm25-stats.json
    python scripts\\rag_sparse_index.py run       # nap sparse vector (4 part, resume)
    python scripts\\rag_sparse_index.py status
    python scripts\\rag_sparse_index.py verify
    python scripts\\rag_sparse_index.py all       # init + stats + run
    python scripts\\rag_sparse_index.py reset     # xoa collection sparse (nap lai tu dau)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CFG, validate  # noqa: E402
from rag.bm25 import BM25Stats, build_stats  # noqa: E402
from rag.paths import (  # noqa: E402
    BM25_STATS, COLLECTION, SPARSE_COLLECTION, SPARSE_CURSOR, SPARSE_PARTS_DIR,
    VECTOR_NAME,
)
from rel19.common import jdump, load_json, log  # noqa: E402
from rel19.paths import CHUNKS_FILE  # noqa: E402


def make_client():
    from qdrant_client import QdrantClient

    return QdrantClient(
        url=CFG["QDRANT_URL"], api_key=CFG["QDRANT_API_KEY"],
        timeout=CFG["QDRANT_TIMEOUT"], prefer_grpc=CFG["QDRANT_PREFER_GRPC"],
    )


def _ensure_indexes(fields: list[tuple[str, str]], collection: str) -> None:
    """Tao payload index qua REST (gRPC hay loi voi index tren collection cu)."""
    from qdrant_client import QdrantClient

    from qdrant_client.models import PayloadSchemaType

    rest = QdrantClient(
        url=CFG["QDRANT_URL"], api_key=CFG["QDRANT_API_KEY"],
        timeout=CFG["QDRANT_TIMEOUT"], prefer_grpc=False,
    )
    try:
        for field, kind in fields:
            try:
                rest.create_payload_index(
                    collection, field,
                    field_schema=getattr(PayloadSchemaType, kind.upper()),
                )
                log.info("payload index %s.%s", collection, field)
            except Exception as exc:  # noqa: BLE001
                log.info("payload index %s.%s: %s", collection, field, exc)
    finally:
        rest.close()


def _ensure_collection(client) -> None:
    from qdrant_client.models import SparseIndexParams, SparseVectorParams

    # index cho collection dense (filter + neighbor expand)
    _ensure_indexes([("section_path", "keyword"), ("chunk_index", "integer")], COLLECTION)

    if client.collection_exists(SPARSE_COLLECTION):
        info = client.get_collection(SPARSE_COLLECTION)
        log.info("collection sparse đã có: %s (%d point)", SPARSE_COLLECTION, info.points_count)
        return
    client.create_collection(
        collection_name=SPARSE_COLLECTION,
        vectors_config={},
        sparse_vectors_config={
            VECTOR_NAME: SparseVectorParams(index=SparseIndexParams(on_disk=False))
        },
    )
    _ensure_indexes([("spec", "keyword"), ("section_path", "keyword"), ("version", "keyword")],
                    SPARSE_COLLECTION)
    log.info("đã tạo %s (sparse '%s')", SPARSE_COLLECTION, VECTOR_NAME)


# ---------------------------------------------------------------- init / reset
def cmd_init(_args) -> None:
    validate("qdrant")
    client = make_client()
    try:
        _ensure_collection(client)
    finally:
        client.close()


def cmd_reset(_args) -> None:
    validate("qdrant")
    client = make_client()
    try:
        if client.collection_exists(SPARSE_COLLECTION):
            client.delete_collection(SPARSE_COLLECTION)
            log.info("đã xoá %s", SPARSE_COLLECTION)
        for f in SPARSE_PARTS_DIR.glob("part*.json"):
            f.unlink()
        if SPARSE_CURSOR.exists():
            SPARSE_CURSOR.unlink()
        _ensure_collection(client)
    finally:
        client.close()


# ---------------------------------------------------------------- stats
def cmd_stats(args) -> None:
    if BM25_STATS.exists() and not args.force:
        log.info("đã có %s (dùng --force để tính lại)", BM25_STATS)
        return
    started = time.time()
    log.info("quét %s để tính BM25 stats ...", CHUNKS_FILE)

    def progress(n: int, terms: int) -> None:
        log.info("  %d chunks, %d term", n, terms)

    stats = build_stats(CHUNKS_FILE, progress=progress)
    stats.save(BM25_STATS)
    log.info(
        "stats xong: %d chunk, %d term, avgdl=%.1f, %.0fs",
        stats.n_docs, len(stats.terms), stats.avgdl, time.time() - started,
    )


# ---------------------------------------------------------------- run
def split_bounds(size: int, parts: int) -> list[tuple[int, int]]:
    starts = [0]
    with CHUNKS_FILE.open("rb") as fh:
        for i in range(1, parts):
            fh.seek(size * i // parts)
            fh.readline()
            starts.append(fh.tell())
    starts = sorted(set(starts))
    ends = starts[1:] + [size]
    return list(zip(starts, ends))


def _upsert_with_retry(client, points, tries: int = 4) -> None:
    for attempt in range(tries):
        try:
            client.upsert(SPARSE_COLLECTION, points=points, wait=False)
            return
        except Exception as exc:  # noqa: BLE001
            if attempt == tries - 1:
                raise
            log.warning("upsert lỗi (%s), thử lại sau %ds", exc, 2 ** attempt)
            time.sleep(2 ** attempt)


def run_part(i: int, start: int, end: int, stats: BM25Stats, force: bool) -> None:
    from qdrant_client.models import PointStruct, SparseVector

    prog = SPARSE_PARTS_DIR / f"part{i}.json"
    state = {} if force else load_json(prog, {})
    pos = max(int(state.get("pos", 0)), start)
    done = int(state.get("points", 0))
    batch = CFG["SPARSE_UPSERT_BATCH"]
    client = make_client()
    try:
        with CHUNKS_FILE.open("rb") as fh:
            if pos <= start:
                fh.seek(start)
                if start > 0:
                    fh.readline()
            else:
                fh.seek(pos)
            pos = fh.tell()
            while pos < end:
                points = []
                for _ in range(batch):
                    line = fh.readline()
                    if not line:
                        break
                    rec = json.loads(line.decode("utf-8"))
                    inds, vals = stats.doc_vector(rec["text"])
                    if not inds:
                        continue
                    points.append(
                        PointStruct(
                            id=str(uuid.uuid5(uuid.NAMESPACE_URL, rec["id"])),
                            vector={VECTOR_NAME: SparseVector(indices=inds, values=vals)},
                            payload={
                                "spec": rec["spec"],
                                "version": rec["version"],
                                "section_path": rec.get("section_path", ""),
                                "chunk_index": rec.get("chunk_index", 0),
                            },
                        )
                    )
                if not points:
                    break
                _upsert_with_retry(client, points)
                pos = fh.tell()
                done += len(points)
                jdump(prog, {"pos": pos, "points": done})
                if done % 2048 < batch:
                    log.info("part%d: %d point", i, done)
    finally:
        client.close()


def cmd_run(args) -> None:
    validate("qdrant")
    if not BM25_STATS.exists():
        raise SystemExit("chưa có bm25-stats.json; chạy `stats` trước")
    if not CHUNKS_FILE.exists():
        raise SystemExit(f"không thấy {CHUNKS_FILE}")

    cmd_init(args)
    stats = BM25Stats.load(BM25_STATS)
    size = CHUNKS_FILE.stat().st_size
    parts = CFG["SPARSE_PARTS"]
    SPARSE_PARTS_DIR.mkdir(parents=True, exist_ok=True)
    bounds = split_bounds(size, parts)

    log.info("sparse index: %d part, %d term, %.0f MB", parts, len(stats.terms), size / 1024 ** 2)
    threads = [
        threading.Thread(target=run_part, args=(i, bounds[i][0], bounds[i][1], stats, args.force),
                         daemon=False)
        for i in range(parts)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    total = sum(
        load_json(SPARSE_PARTS_DIR / f"part{i}.json", {}).get("points", 0)
        for i in range(parts)
    )
    jdump(SPARSE_CURSOR, {"pos": size, "points": total})
    log.info("sparse index xong: %d point", total)


# ---------------------------------------------------------------- status
def cmd_status(_args) -> None:
    total = 0
    for i in range(CFG["SPARSE_PARTS"]):
        st = load_json(SPARSE_PARTS_DIR / f"part{i}.json", {})
        total += int(st.get("points", 0))
        log.info("part%d: %s point, pos=%s", i, st.get("points", 0), st.get("pos", 0))
    log.info("tổng đã nạp (cursor): %d point", total)
    try:
        client = make_client()
        if client.collection_exists(SPARSE_COLLECTION):
            info = client.get_collection(SPARSE_COLLECTION)
            log.info("collection %s: %d point", SPARSE_COLLECTION, info.points_count)
        else:
            log.info("chưa có collection %s", SPARSE_COLLECTION)
        client.close()
    except Exception as exc:  # noqa: BLE001
        log.warning("không đọc được collection: %s", exc)


# ---------------------------------------------------------------- verify
def cmd_verify(_args) -> None:
    from qdrant_client.models import PointIdsList

    size = CHUNKS_FILE.stat().st_size
    samples: list[str] = []
    with CHUNKS_FILE.open("rb") as fh:
        for _ in range(5):
            fh.seek(random.randrange(0, max(1, size - 1)))
            fh.readline()
            line = fh.readline()
            if line:
                samples.append(json.loads(line.decode("utf-8"))["id"])

    client = make_client()
    try:
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, s)) for s in samples]
        points = client.retrieve(
            SPARSE_COLLECTION, ids=PointIdsList(points=ids), with_vectors=True,
        )
        for p in points:
            vectors = p.vector if isinstance(p.vector, dict) else {}
            sparse = vectors.get(VECTOR_NAME)
            n = len(sparse.indices) if sparse else 0
            log.info("%s: sparse=%s (%d term), spec=%s",
                     p.id, bool(sparse), n, (p.payload or {}).get("spec"))
        log.info("kiểm tra %d/%d mẫu", len(points), len(samples))
    finally:
        client.close()


def main() -> None:
    import logging

    logging.basicConfig(
        level=getattr(logging, CFG["LOG_LEVEL"], logging.INFO),
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    st = sub.add_parser("stats")
    st.add_argument("--force", action="store_true")
    rn = sub.add_parser("run")
    rn.add_argument("--force", action="store_true")
    sub.add_parser("status")
    sub.add_parser("verify")
    sub.add_parser("all")
    sub.add_parser("reset")

    args = p.parse_args()
    if args.cmd == "reset":
        cmd_reset(args)
    if args.cmd in ("init", "all"):
        cmd_init(args)
    if args.cmd in ("stats", "all"):
        cmd_stats(args)
    if args.cmd in ("run", "all"):
        cmd_run(args)
    if args.cmd == "status":
        cmd_status(args)
    if args.cmd == "verify":
        cmd_verify(args)


if __name__ == "__main__":
    main()
