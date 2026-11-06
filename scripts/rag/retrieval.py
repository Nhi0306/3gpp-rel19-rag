"""Truy xuat hybrid: dense (70%) + BM25 sparse (30%), hop nhat + mo rong lan can."""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition, Filter, MatchAny, MatchValue, SparseVector,
)

from config import CFG
from rag.bm25 import BM25Stats
from rag.embeddings import embed_query
from rag.paths import BM25_STATS, COLLECTION, SPARSE_COLLECTION, VECTOR_NAME
from rel19.common import log

_client: QdrantClient | None = None
_stats: BM25Stats | None = None
_lock = threading.Lock()


def client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(
            url=CFG["QDRANT_URL"], api_key=CFG["QDRANT_API_KEY"],
            timeout=CFG["QDRANT_TIMEOUT"], prefer_grpc=CFG["QDRANT_PREFER_GRPC"],
        )
    return _client


def stats() -> BM25Stats:
    global _stats
    if _stats is None:
        if not BM25_STATS.exists():
            raise RuntimeError("chưa có bm25-stats.json; chạy rag_sparse_index.py stats")
        _stats = BM25Stats.load(BM25_STATS)
    return _stats


def _spec_filter(specs: list[str] | None) -> Filter | None:
    if not specs:
        return None
    return Filter(must=[FieldCondition(key="spec", match=MatchAny(any=list(specs)))])


def dense_search(client: QdrantClient, vector: list[float], limit: int,
                 specs: list[str] | None) -> list[dict]:
    res = client.query_points(
        COLLECTION, query=vector, limit=limit,
        query_filter=_spec_filter(specs), with_payload=True,
    )
    return [
        {"id": str(p.id), "score": float(p.score), "payload": p.payload or {}, "source": "dense"}
        for p in res.points
    ]


def sparse_search(client: QdrantClient, question: str, limit: int,
                  specs: list[str] | None) -> list[dict]:
    inds, vals = stats().query_vector(question)
    if not inds:
        return []
    res = client.query_points(
        SPARSE_COLLECTION, query=SparseVector(indices=inds, values=vals),
        using=VECTOR_NAME, limit=limit, query_filter=_spec_filter(specs), with_payload=False,
    )
    return [
        {"id": str(p.id), "score": float(p.score), "payload": {}, "source": "sparse"}
        for p in res.points
    ]


def _fetch_payloads(client: QdrantClient, ids: list[str]) -> dict[str, dict]:
    if not ids:
        return {}
    out: dict[str, dict] = {}
    for i in range(0, len(ids), 256):
        chunk = ids[i:i + 256]
        points = client.retrieve(
            COLLECTION, ids=chunk, with_payload=True, with_vectors=False,
        )
        for p in points:
            out[str(p.id)] = p.payload or {}
    return out


def fuse(dense: list[dict], sparse: list[dict], top_k: int, section: str = "") -> list[dict]:
    """70% dense (cosine) + 30% BM25 (chia theo max), dedupe theo id."""
    wd = CFG["RAG_DENSE_WEIGHT"]
    ws = CFG["RAG_SPARSE_WEIGHT"]
    merged: dict[str, dict] = {}

    for h in dense:
        merged[h["id"]] = {
            "id": h["id"], "payload": h["payload"],
            "dense": max(0.0, h["score"]), "sparse": 0.0, "source": "dense",
        }

    smax = max((h["score"] for h in sparse), default=0.0) or 1.0
    for h in sparse:
        item = merged.setdefault(h["id"], {
            "id": h["id"], "payload": h["payload"], "dense": 0.0, "sparse": 0.0, "source": "sparse",
        })
        item["sparse"] = h["score"] / smax
        if item["source"] == "dense":
            item["source"] = "both"

    for item in merged.values():
        item["score"] = wd * item["dense"] + ws * item["sparse"]
        payload = item.get("payload") or {}
        if section and str(payload.get("section", "")).startswith(section):
            item["score"] += 0.03

    ranked = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
    return ranked[:top_k]


def expand_neighbors(client: QdrantClient, hits: list[dict], window: int,
                     expand_top: int = 6) -> list[dict]:
    """Lay them chunk lan can cung section de giu mach van ban."""
    if window <= 0 or not hits:
        return hits
    seen = {h["id"] for h in hits}
    extra: list[dict] = []
    for h in hits[:expand_top]:
        payload = h.get("payload") or {}
        spec = payload.get("spec")
        path = payload.get("section_path")
        idx = payload.get("chunk_index")
        if spec is None or path is None or idx is None:
            continue
        wanted = [idx + d for d in range(-window, window + 1) if d != 0]
        flt = Filter(must=[
            FieldCondition(key="spec", match=MatchValue(value=spec)),
            FieldCondition(key="section_path", match=MatchValue(value=path)),
            FieldCondition(key="chunk_index", match=MatchAny(any=wanted)),
        ])
        try:
            points = client.scroll(
                COLLECTION, scroll_filter=flt, limit=window * 2,
                with_payload=True, with_vectors=False,
            )[0]
        except Exception as exc:  # noqa: BLE001
            log.warning("expand lỗi (bỏ qua): %s", exc)
            continue
        for p in points:
            pid = str(p.id)
            if pid in seen:
                continue
            seen.add(pid)
            extra.append({
                "id": pid, "payload": p.payload or {},
                "dense": 0.0, "sparse": 0.0,
                "score": h["score"] - 0.01, "source": "neighbor",
            })
    return hits + extra


def search(question: str, specs: list[str] | None = None, section: str = "",
           dense_top: int | None = None, sparse_top: int | None = None,
           top_k: int | None = None, neighbors: int | None = None) -> list[dict]:
    """Chay dense || sparse, hop nhat 70/30, mo rong lan can."""
    dense_top = dense_top or CFG["RAG_DENSE_TOP"]
    sparse_top = sparse_top or CFG["RAG_SPARSE_TOP"]
    top_k = top_k or CFG["RAG_TOP_K"]
    neighbors = CFG["RAG_NEIGHBOR_WINDOW"] if neighbors is None else neighbors

    vector = embed_query(question)
    cli = client()
    with ThreadPoolExecutor(max_workers=2) as pool:
        fd = pool.submit(dense_search, cli, vector, dense_top, specs)
        fs = pool.submit(sparse_search, cli, question, sparse_top, specs)
        dense = fd.result()
        try:
            sparse = fs.result()
        except Exception as exc:  # noqa: BLE001  (collection sparse chua xong)
            log.warning("sparse search lỗi, chỉ dùng dense: %s", exc)
            sparse = []

    hits = fuse(dense, sparse, top_k, section=section)

    missing = [h["id"] for h in hits if not h.get("payload")]
    if missing:
        payloads = _fetch_payloads(cli, missing)
        for h in hits:
            if not h.get("payload"):
                h["payload"] = payloads.get(h["id"], {})

    hits = [h for h in hits if h.get("payload")]
    hits = expand_neighbors(cli, hits, neighbors)
    hits.sort(key=lambda x: x["score"], reverse=True)
    return hits


def snippet(hit: dict, max_chars: int = 320) -> str:
    text = " ".join(str((hit.get("payload") or {}).get("text", "")).split())
    return text if len(text) <= max_chars else text[:max_chars].rstrip() + " ..."


def citation_of(hit: dict, rank: int) -> dict:
    p = hit.get("payload") or {}
    return {
        "rank": rank,
        "chunk_id": p.get("id", hit["id"]),
        "spec": p.get("spec", ""),
        "version": p.get("version", ""),
        "section": p.get("section", ""),
        "section_path": p.get("section_path", ""),
        "title": p.get("title", ""),
        "score": round(float(hit.get("score", 0)), 4),
        "source": hit.get("source", ""),
        "snippet": snippet(hit, 500),
    }


def close() -> None:
    global _client
    if _client is not None:
        try:
            _client.close()
        finally:
            _client = None
