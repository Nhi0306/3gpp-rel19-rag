#!/usr/bin/env python3
"""Chat voi he thong RAG 3GPP Rel-19.

    python scripts\\rag_chat.py
    python scripts\\rag_chat.py --session <id>
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
