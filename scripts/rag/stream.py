"""Chia se callback nhan token cho CLI/API (contextvar, an toan theo thread)."""
from __future__ import annotations

import contextvars
from collections.abc import Callable

token_sink: contextvars.ContextVar[Callable[[str], None] | None] = contextvars.ContextVar(
    "token_sink", default=None,
)


def emit(text: str) -> None:
    sink = token_sink.get()
    if sink and text:
        sink(text)
