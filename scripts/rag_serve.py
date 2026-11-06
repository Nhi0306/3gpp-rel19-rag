#!/usr/bin/env python3
"""Chay Web API + giao dien chat cho he thong RAG 3GPP Rel-19.

    python scripts\\rag_serve.py                 # http://127.0.0.1:8000
    python scripts\\rag_serve.py --port 8080
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import uvicorn  # noqa: E402

from rag.api import app  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Web server RAG 3GPP Rel-19")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
