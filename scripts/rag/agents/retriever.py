"""Agent 2: Retriever - dense || sparse cho tung sub-query, va buoc merge."""
from __future__ import annotations

from rag import retrieval
from rag.agents.common import trace
from rag.state import RAGState
from rel19.common import log


def retrieve(state: RAGState) -> dict:
    query = state.get("sub_query") or state.get("question", "")
    specs = state.get("specs") or []
    section = state.get("section") or ""
    rnd = int(state.get("iterations", 0))
    try:
        hits = retrieval.search(query, specs=specs, section=section)
    except Exception as exc:  # noqa: BLE001
        log.warning("retrieve lỗi (%s): %s", query, exc)
        hits = []
    for h in hits:
        h["round"] = rnd
        h["sub_query"] = query
    log.info("retrieve: %d hit cho %r", len(hits), query)
    return {
        "candidates": hits,
        "trace": [trace("retriever", query=query, hits=len(hits), round=rnd)],
    }


def merge(state: RAGState) -> dict:
    """Gop ket qua cac sub-query cua vong hien tai, dedupe, lay top 24."""
    cands = state.get("candidates") or []
    if not cands:
        return {"evidence": [], "trace": [trace("merge", evidence=0)]}
    latest = max(int(c.get("round", 0)) for c in cands)
    cands = [c for c in cands if int(c.get("round", 0)) == latest]

    best: dict[str, dict] = {}
    for c in cands:
        cid = c["id"]
        if cid not in best or c["score"] > best[cid]["score"]:
            best[cid] = c
    ranked = sorted(best.values(), key=lambda x: x["score"], reverse=True)[:24]
    return {
        "evidence": ranked,
        "trace": [trace("merge", evidence=len(ranked), round=latest)],
    }
