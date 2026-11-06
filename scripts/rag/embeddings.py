"""Embed cau hoi (cung model voi luc ingest)."""
from __future__ import annotations

import requests

from config import CFG


def embed_query(text: str) -> list[float]:
    provider = CFG["EMBED_PROVIDER"]
    if provider == "openrouter":
        url = f"{CFG['OPENROUTER_BASE_URL']}/embeddings"
        key = CFG["OPENROUTER_API_KEY"]
        model = CFG["OPENROUTER_EMBED_MODEL"]
    else:
        url = f"{CFG['OPENAI_BASE_URL']}/embeddings"
        key = CFG["OPENAI_API_KEY"]
        model = CFG["EMBED_MODEL"]
    r = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": CFG["OPENROUTER_SITE_URL"] or "http://localhost",
            "X-Title": CFG["OPENROUTER_APP_NAME"],
        },
        json={"model": model, "input": [text]},
        timeout=CFG["HTTP_TIMEOUT"],
    )
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]
