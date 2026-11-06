"""Goi LLM qua OpenRouter (OpenAI-compatible) + structured output."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import TypeVar

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from config import CFG
from rel19.common import log

T = TypeVar("T", bound=BaseModel)

_AGENT_KEY = {
    "supervisor": "PLANNER_MODEL",
    "grader": "GRADER_MODEL",
    "synthesizer": "SYNTH_MODEL",
    "verifier": "VERIFIER_MODEL",
}

_FENCE = re.compile(r"```(?:json)?", re.I)


@dataclass
class Usage:
    agent: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0


def model_name(agent: str) -> str:
    return CFG[_AGENT_KEY.get(agent, "SYNTH_MODEL")]


def chat_model(
    agent: str, temperature: float | None = None, max_tokens: int | None = None,
    model: str | None = None,
) -> ChatOpenAI:
    if max_tokens is None:
        max_tokens = CFG["SYNTH_MAX_TOKENS"] if agent == "synthesizer" else CFG["LLM_MAX_TOKENS"]
    return ChatOpenAI(
        model=model or model_name(agent),
        api_key=CFG["OPENROUTER_API_KEY"],
        base_url=CFG["OPENROUTER_BASE_URL"],
        temperature=CFG["LLM_TEMPERATURE"] if temperature is None else temperature,
        max_tokens=max_tokens,
        timeout=CFG["LLM_TIMEOUT"],
        max_retries=2,
        stream_usage=True,
        default_headers={
            "HTTP-Referer": CFG["OPENROUTER_SITE_URL"] or "http://localhost",
            "X-Title": CFG["OPENROUTER_APP_NAME"],
        },
    )


def _usage(msg, agent: str, started: float, model: str | None = None) -> Usage:
    meta = getattr(msg, "usage_metadata", None) or {}
    return Usage(
        agent=agent,
        model=model or model_name(agent),
        input_tokens=int(meta.get("input_tokens", 0) or 0),
        output_tokens=int(meta.get("output_tokens", 0) or 0),
        latency_ms=int((time.time() - started) * 1000),
    )


def _extract_json(text: str) -> str:
    t = _FENCE.sub("", text or "").replace("```", "").strip()
    if t.startswith("{"):
        return t
    m = re.search(r"\{.*\}", t, re.S)
    return m.group(0) if m else t


def call_json(
    agent: str, system: str, user: str, schema: type[T],
    temperature: float = 0.0, retries: int = 2,
) -> tuple[T, Usage]:
    """Goi LLM va parse JSON theo schema; tu retry khi model tra loi hong."""
    llm = chat_model(agent, temperature=temperature)
    last: Exception | None = None
    for attempt in range(retries + 1):
        started = time.time()
        try:
            msg = llm.invoke([("system", system), ("human", user)])
            usage = _usage(msg, agent, started)
            data = json.loads(_extract_json(str(msg.content)))
            return schema.model_validate(data), usage
        except Exception as exc:  # noqa: BLE001
            last = exc
            log.warning("llm %s: JSON lỗi (lần %d) - %s", agent, attempt + 1, exc)
            time.sleep(1 + attempt)
    raise RuntimeError(f"{agent}: không parse được JSON sau {retries + 1} lần: {last}")


def call_text(
    agent: str, system: str, user: str, temperature: float | None = None,
    model: str | None = None, retries: int = 1,
) -> tuple[str, Usage]:
    """Goi LLM tra text; neu model tra rong (reasoning an het token) thi thu lai."""
    llm = chat_model(agent, temperature=temperature, model=model)
    text, usage = "", None
    for attempt in range(retries + 1):
        started = time.time()
        msg = llm.invoke([("system", system), ("human", user)])
        usage = _usage(msg, agent, started, model)
        text = str(msg.content or "").strip()
        if text:
            return text, usage
        log.warning("llm %s: nội dung rỗng (lần %d/%d, out=%d token)",
                    agent, attempt + 1, retries + 1, usage.output_tokens)
        time.sleep(1)
    return text, usage or Usage(agent=agent, model=model or model_name(agent))


def call_text_stream(
    agent: str, system: str, user: str, temperature: float | None = None,
    model: str | None = None,
) -> tuple[str, Usage]:
    """Goi LLM dang stream, day tung token ra token_sink (neu co)."""
    from rag.stream import emit

    llm = chat_model(agent, temperature=temperature, model=model)
    started = time.time()
    parts: list[str] = []
    tin = tout = 0
    for chunk in llm.stream([("system", system), ("human", user)]):
        if chunk.content:
            text = str(chunk.content)
            parts.append(text)
            emit(text)
        meta = getattr(chunk, "usage_metadata", None) or {}
        tin = int(meta.get("input_tokens", tin) or tin)
        tout = int(meta.get("output_tokens", tout) or tout)
    return "".join(parts).strip(), Usage(
        agent=agent, model=model or model_name(agent),
        input_tokens=tin, output_tokens=tout,
        latency_ms=int((time.time() - started) * 1000),
    )
