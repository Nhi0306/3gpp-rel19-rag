#!/usr/bin/env python3
"""Truy van nhanh collection Qdrant da nap.

    python scripts\\search.py "network slicing admission control" -k 5
    python scripts\\search.py "NG-RAN paging" -k 5 --spec 38.304 --section 7
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CFG, validate  # noqa: E402


def embed(text: str) -> list[float]:
    provider = CFG["EMBED_PROVIDER"]
    url = (
        f"{CFG['OPENROUTER_BASE_URL']}/embeddings"
        if provider == "openrouter"
        else f"{CFG['OPENAI_BASE_URL']}/embeddings"
    )
    key = CFG["OPENROUTER_API_KEY"] if provider == "openrouter" else CFG["OPENAI_API_KEY"]
    model = CFG["OPENROUTER_EMBED_MODEL"] if provider == "openrouter" else CFG["EMBED_MODEL"]
    r = requests.post(
        url,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": model, "input": [text]},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("query")
    p.add_argument("-k", "--top", type=int, default=5)
    p.add_argument("--spec", help="filter theo so spec, vd 23.501")
    p.add_argument("--section", help="filter section bat dau bang, vd 4.2")
    p.add_argument("--full", action="store_true", help="in toan bo text")
    args = p.parse_args()

    validate("embed")
    from qdrant_client import QdrantClient
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    client = QdrantClient(
        url=CFG["QDRANT_URL"], api_key=CFG["QDRANT_API_KEY"], timeout=CFG["QDRANT_TIMEOUT"]
    )

    must = []
    if args.spec:
        must.append(FieldCondition(key="spec", match=MatchValue(value=args.spec)))
    if args.section:
        must.append(FieldCondition(key="section", match=MatchValue(value=args.section)))
    flt = Filter(must=must) if must else None

    hits = client.query_points(
        CFG["QDRANT_COLLECTION"],
        query=embed(args.query),
        limit=args.top,
        query_filter=flt,
        with_payload=True,
    ).points

    for i, h in enumerate(hits, 1):
        pl = h.payload or {}
        print(f"\n[{i}] score={h.score:.4f}  {pl.get('spec_type')} {pl.get('spec')} "
              f"v{pl.get('version')}  {pl.get('section_path')}")
        print(f"    {pl.get('title', '')[:100]}")
        text = pl.get("text", "")
        print("    " + (text if args.full else text[:500]).replace("\n", "\n    "))


if __name__ == "__main__":
    main()
