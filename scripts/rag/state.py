"""Trang thai dung chung cua graph."""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class RAGState(TypedDict, total=False):
    session_id: str
    turn_id: int
    question: str
    started_at: float
    history: list[dict]
    history_text: str
    language: str
    intent: str
    specs: list[str]
    section: str
    sub_query: str
    sub_queries: list[str]
    refine_hint: str
    iterations: int
    candidates: Annotated[list[dict], operator.add]
    evidence: list[dict]
    sufficient: bool
    answer: str
    citations: list[dict]
    verify_attempts: int
    verdict: dict
    trace: Annotated[list[dict], operator.add]
    usage: Annotated[list[dict], operator.add]
