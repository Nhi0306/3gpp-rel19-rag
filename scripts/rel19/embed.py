"""Bước 6: embed chunk và upsert lên Qdrant (gRPC, chia nhiều phần song song)."""
from __future__ import annotations

import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Thread

import numpy as np
import requests

from config import CFG, validate
from rel19.common import jdump, load_json, log
from rel19.paths import CHUNKS_FILE, EMBED_CURSOR, EMBED_PARTS_DIR

DATATYPES = {"float32": "float32", "float16": "float16", "uint8": "uint8"}

EMBED_URLS = {
    "openrouter": "https://openrouter.ai/api/v1/embeddings",
    "openai": "https://api.openai.com/v1/embeddings",
}


def embed_batch(
    texts: list[str], provider: str, model: str, api_key: str, retries: int = 5
) -> list[list[float]]:
    url = EMBED_URLS[provider]
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if provider == "openrouter":
        headers["HTTP-Referer"] = CFG["OPENROUTER_SITE_URL"] or "https://localhost"
        headers["X-Title"] = CFG["OPENROUTER_APP_NAME"]
    payload = {"model": model, "input": texts}
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=180)
            if r.status_code == 429:
                time.sleep(min(2 ** attempt, 30))
                continue
            r.raise_for_status()
            data = r.json()["data"]
            data.sort(key=lambda d: d.get("index", 0))
            return [d["embedding"] for d in data]
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"embed failed: {last}")


def qdrant_upsert(client, coll: str, points: list, batch: int = 32, retries: int = 6) -> None:
    """Upsert có retry/backoff; Qdrant Cloud có thể timeout lúc collection mới tạo."""
    for start in range(0, len(points), batch):
        part = points[start:start + batch]
        last: Exception | None = None
        for attempt in range(retries):
            try:
                client.upsert(coll, points=part, wait=False)
                last = None
                break
            except Exception as exc:  # noqa: BLE001
                last = exc
                log.warning("upsert retry %d/%d: %s", attempt + 1, retries, exc)
                time.sleep(min(2 ** attempt, 30))
        if last is not None:
            raise RuntimeError(f"upsert failed: {last}")


def embed_range(
    start: int, end: int, progress: Path | None = None, workers: int = 2
) -> int:
    """Embed cac chunk trong khoang byte [start, end) va upsert len Qdrant.

    Dung cho cap nhat tang dan: chi xu ly phan moi duoc append vao cuoi JSONL.
    """
    validate("embed")
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Datatype, Distance, PayloadSchemaType, PointStruct, VectorParams,
    )

    provider = CFG["EMBED_PROVIDER"]
    api_key = CFG["OPENROUTER_API_KEY"] if provider == "openrouter" else CFG["OPENAI_API_KEY"]
    model = CFG["OPENROUTER_EMBED_MODEL"] if provider == "openrouter" else CFG["EMBED_MODEL"]
    dim = CFG["EMBED_DIM"]
    coll = CFG["QDRANT_COLLECTION"]
    dtype_name = DATATYPES.get(CFG["QDRANT_DATATYPE"], "float32")
    dtype = getattr(Datatype, dtype_name.upper(), Datatype.FLOAT32)
    dist = {"cosine": Distance.COSINE, "dot": Distance.DOT, "euclid": Distance.EUCLID}[
        CFG["QDRANT_DISTANCE"].lower()
    ]
    bs = CFG["EMBED_BATCH_SIZE"]
    reconnect_every = max(1, CFG["QDRANT_RECONNECT_EVERY"])

    def make_client() -> QdrantClient:
        return QdrantClient(
            url=CFG["QDRANT_URL"], api_key=CFG["QDRANT_API_KEY"],
            timeout=CFG["QDRANT_TIMEOUT"], prefer_grpc=CFG["QDRANT_PREFER_GRPC"],
        )

    def to_vector(vec: list[float]):
        return np.asarray(vec, dtype=np.float16) if dtype is Datatype.FLOAT16 else vec

    client = make_client()
    if not client.collection_exists(coll):
        client.create_collection(
            coll, vectors_config=VectorParams(size=dim, distance=dist, datatype=dtype)
        )
        log.info("created collection %s (dim=%d, %s)", coll, dim, dtype_name)
        client.create_payload_index(coll, "spec", field_schema=PayloadSchemaType.KEYWORD)

    state = load_json(progress, {}) if progress else {}
    pos = max(int(state.get("pos", 0)), start)
    base_chunks = int(state.get("chunks", 0))
    done = 0
    pool = ThreadPoolExecutor(max_workers=workers)
    groups = 0
    try:
        with CHUNKS_FILE.open("rb") as fh:
            fh.seek(pos)
            if pos > 0:
                fh.seek(pos - 1)
                if fh.read(1) != b"\n":  # start giua dong -> bo dong bi cat
                    fh.readline()
            pos = fh.tell()
            while pos < end:
                group = []
                for _ in range(bs * workers):
                    line = fh.readline()
                    if not line:
                        break
                    group.append(json.loads(line.decode("utf-8")))
                if not group:
                    break
                batches = [group[k:k + bs] for k in range(0, len(group), bs)]
                results = list(pool.map(
                    lambda b: embed_batch([x["text"] for x in b], provider, model, api_key),
                    batches,
                ))
                points = [
                    PointStruct(
                        id=str(uuid.uuid5(uuid.NAMESPACE_URL, b["id"])),
                        vector=to_vector(vec),
                        payload={k: v for k, v in b.items() if k != "id"},
                    )
                    for batch, vectors in zip(batches, results)
                    for b, vec in zip(batch, vectors)
                ]
                qdrant_upsert(client, coll, points, batch=CFG["QDRANT_UPSERT_BATCH"])
                pos = fh.tell()
                done += len(group)
                if progress:
                    jdump(progress, {"pos": pos, "chunks": base_chunks + done})
                if done % 512 < len(group):
                    log.info("update: %d chunks uploaded", done)
                groups += 1
                if groups % reconnect_every == 0:
                    try:
                        client.close()
                    except Exception:  # noqa: BLE001
                        pass
                    client = make_client()
    finally:
        pool.shutdown(wait=True)
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass
    return done


def cmd_embed(args) -> None:
    validate("embed")
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Datatype, Distance, PayloadSchemaType, PointStruct, VectorParams,
    )

    provider = CFG["EMBED_PROVIDER"]
    api_key = CFG["OPENROUTER_API_KEY"] if provider == "openrouter" else CFG["OPENAI_API_KEY"]
    model = CFG["OPENROUTER_EMBED_MODEL"] if provider == "openrouter" else CFG["EMBED_MODEL"]
    dim = CFG["EMBED_DIM"]

    client = QdrantClient(
        url=CFG["QDRANT_URL"],
        api_key=CFG["QDRANT_API_KEY"],
        timeout=CFG["QDRANT_TIMEOUT"],
        prefer_grpc=CFG["QDRANT_PREFER_GRPC"],
    )

    def make_client() -> QdrantClient:
        return QdrantClient(
            url=CFG["QDRANT_URL"],
            api_key=CFG["QDRANT_API_KEY"],
            timeout=CFG["QDRANT_TIMEOUT"],
            prefer_grpc=CFG["QDRANT_PREFER_GRPC"],
        )
    coll = CFG["QDRANT_COLLECTION"]
    dist = {"cosine": Distance.COSINE, "dot": Distance.DOT, "euclid": Distance.EUCLID}[
        CFG["QDRANT_DISTANCE"].lower()
    ]
    dtype_name = DATATYPES.get(CFG["QDRANT_DATATYPE"], "float32")
    dtype = getattr(Datatype, dtype_name.upper(), Datatype.FLOAT32)
    if not client.collection_exists(coll):
        client.create_collection(
            coll, vectors_config=VectorParams(size=dim, distance=dist, datatype=dtype)
        )
        log.info("created collection %s (dim=%d, %s)", coll, dim, dtype_name)
        client.create_payload_index(coll, "spec", field_schema=PayloadSchemaType.KEYWORD)
    else:
        info = client.get_collection(coll)
        existing = str(getattr(info.config.params.vectors, "datatype", "float32")).lower()
        if dtype_name not in existing:
            raise SystemExit(
                f"collection {coll} dang la {existing}, nhung QDRANT_DATATYPE={dtype_name}. "
                "Doi lai .env hoac xoa collection roi chay lai."
            )

    def to_vector(vec: list[float]):
        return np.asarray(vec, dtype=np.float16) if dtype is Datatype.FLOAT16 else vec

    if not CHUNKS_FILE.exists():
        raise SystemExit("no chunks file; run `chunk` first")

    parts = max(1, CFG["EMBED_PARTS"])
    bs = CFG["EMBED_BATCH_SIZE"]
    workers = max(1, CFG["EMBED_CONCURRENCY"])
    group_size = bs * workers

    EMBED_PARTS_DIR.mkdir(parents=True, exist_ok=True)
    size = CHUNKS_FILE.stat().st_size
    # chia file theo byte, mỗi phần tự căn về đầu dòng
    bounds = [size * i // parts for i in range(parts)] + [size]

    def run_part(i: int) -> None:
        start_b, end_b = bounds[i], bounds[i + 1]
        if start_b >= end_b:
            return
        prog = EMBED_PARTS_DIR / f"part{i}.json"
        state = {} if args.force else load_json(prog, {})
        pos = max(int(state.get("pos", 0)), start_b)
        done = int(state.get("chunks", 0))
        pool = ThreadPoolExecutor(max_workers=workers)
        # channel gRPC de lau bi cham dan -> dinh ky tao lai
        part_client = make_client()
        reconnect_every = max(1, CFG["QDRANT_RECONNECT_EVERY"])
        groups_done = 0
        try:
            with CHUNKS_FILE.open("rb") as fh:
                if pos <= start_b:
                    fh.seek(start_b)
                    if start_b > 0:
                        fh.readline()  # bỏ dòng bị cắt giữa 2 phần
                else:
                    fh.seek(pos)
                pos = fh.tell()
                while pos < end_b:
                    group = []
                    for _ in range(group_size):
                        line = fh.readline()
                        if not line:
                            break
                        group.append(json.loads(line.decode("utf-8")))
                    if not group:
                        break
                    batches = [group[k:k + bs] for k in range(0, len(group), bs)]
                    results = list(pool.map(
                        lambda b: embed_batch([x["text"] for x in b], provider, model, api_key),
                        batches,
                    ))
                    points = [
                        PointStruct(
                            id=str(uuid.uuid5(uuid.NAMESPACE_URL, b["id"])),
                            vector=to_vector(vec),
                            payload={k: v for k, v in b.items() if k != "id"},
                        )
                        for batch, vectors in zip(batches, results)
                        for b, vec in zip(batch, vectors)
                    ]
                    qdrant_upsert(part_client, coll, points, batch=CFG["QDRANT_UPSERT_BATCH"])
                    pos = fh.tell()
                    done += len(group)
                    jdump(prog, {"pos": pos, "chunks": done})
                    log.info("part%d: %d chunks uploaded", i, done)
                    groups_done += 1
                    if groups_done % reconnect_every == 0:
                        try:
                            part_client.close()
                        except Exception:  # noqa: BLE001
                            pass
                        part_client = make_client()
        finally:
            pool.shutdown(wait=True)
            try:
                part_client.close()
            except Exception:  # noqa: BLE001
                pass

    log.info("embed: %d parts x %d workers, %d bytes", parts, workers, size)
    threads = [Thread(target=run_part, args=(i,), daemon=False) for i in range(parts)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    total = sum(load_json(EMBED_PARTS_DIR / f"part{i}.json", {}).get("chunks", 0) for i in range(parts))
    jdump(EMBED_CURSOR, {"pos": CHUNKS_FILE.stat().st_size, "chunks": total})
    log.info("embed done: %d chunks -> %s", total, coll)
