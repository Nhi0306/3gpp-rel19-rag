"""Ham dung chung cho cac agent."""
from __future__ import annotations

import json
from dataclasses import asdict

from rag.llm import Usage


def usage_dict(usage: Usage) -> dict:
    d = asdict(usage)
    d["status"] = "ok"
    return d


def schema_hint(schema) -> str:
    return json.dumps(schema.model_json_schema(), ensure_ascii=False)


def format_evidence(hits: list[dict], max_chars: int = 2200) -> str:
    blocks: list[str] = []
    for i, h in enumerate(hits, 1):
        p = h.get("payload") or {}
        text = " ".join(str(p.get("text", "")).split())
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + " ..."
        blocks.append(
            f"[{i}] {p.get('spec', '')} v{p.get('version', '')} §{p.get('section', '')} "
            f"— {p.get('title', '')}\n{text}"
        )
    return "\n\n".join(blocks)


def trace(agent: str, **data) -> dict:
    return {"agent": agent, **data}
