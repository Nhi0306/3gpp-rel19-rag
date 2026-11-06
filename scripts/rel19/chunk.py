"""Bước 5: chunk theo heading trước, rồi theo token -> JSONL."""
from __future__ import annotations

import json
import re
from pathlib import Path

from config import CFG
from rel19.common import load_json, log
from rel19.paths import CHUNKED_DOCS, CHUNKS_FILE, MANIFEST, PARSED

_ENCODER = None


def encoder():
    global _ENCODER
    if _ENCODER is None:
        import tiktoken

        _ENCODER = tiktoken.get_encoding("cl100k_base")
    return _ENCODER


def ntokens(text: str) -> int:
    return len(encoder().encode(text, disallowed_special=()))


def hard_split_text(text: str, max_t: int, overlap: int) -> list[str]:
    """Cắt cứng theo cửa sổ token cho block không có dấu câu."""
    enc = encoder()
    ids = enc.encode(text, disallowed_special=())
    if len(ids) <= max_t:
        return [text]
    step = max(1, max_t - overlap)
    out: list[str] = []
    for start in range(0, len(ids), step):
        window = ids[start:start + max_t]
        if not window:
            break
        out.append(enc.decode(window))
        if start + max_t >= len(ids):
            break
    return out


def split_oversized(blocks: list[dict], max_t: int, overlap: int) -> list[list[dict]]:
    parts: list[list[dict]] = []
    cur: list[dict] = []
    cur_t = 0
    for block in blocks:
        bt = ntokens(block["text"]) + 1
        if cur and cur_t + bt > max_t:
            parts.append(cur)
            tail, tail_t = [], 0
            for prev in reversed(cur):
                pt = ntokens(prev["text"]) + 1
                if tail_t + pt > overlap:
                    break
                tail.insert(0, prev)
                tail_t += pt
            cur, cur_t = tail[:], tail_t
        if bt > max_t:  # block quá lớn -> cắt theo câu, rồi cắt cứng theo token
            sentences = re.split(r"(?<=[.;:!?])\s+", block["text"])
            for sent in sentences:
                st = ntokens(sent) + 1
                if st > max_t:
                    if cur:
                        parts.append(cur)
                        cur, cur_t = [], 0
                    for piece in hard_split_text(sent, max_t, overlap):
                        parts.append([{"type": "p", "text": piece}])
                    continue
                if cur and cur_t + st > max_t:
                    parts.append(cur)
                    cur, cur_t = [], 0
                cur.append({"type": "p", "text": sent})
                cur_t += st
            continue
        cur.append(block)
        cur_t += bt
    if cur:
        parts.append(cur)
    return parts


def chunk_files(files: list[dict], force: bool = False) -> int:
    """Chunk cac file docx da parse, append vao JSONL. Tra ve so chunk moi."""
    max_t = CFG["CHUNK_MAX_TOKENS"]
    overlap = CFG["CHUNK_OVERLAP"]
    min_t = CFG["CHUNK_MIN_TOKENS"]
    skip_re = re.compile(CFG["CHUNK_SKIP_PATTERN"], re.I) if CFG["CHUNK_SKIP_PATTERN"] else None

    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    seen = (
        set(CHUNKED_DOCS.read_text(encoding="utf-8").splitlines())
        if CHUNKED_DOCS.exists() and not force
        else set()
    )

    total = 0
    append = CHUNKS_FILE.exists() and not force
    with CHUNKS_FILE.open("a" if append else "w", encoding="utf-8") as fout, \
            CHUNKED_DOCS.open("a" if append else "w", encoding="utf-8") as fseen:
        for rec in files:
            src = PARSED / f"{rec['series']}_series" / (Path(rec["local_zip"]).stem + ".json")
            if not src.exists() or rec["spec"] in seen:
                continue
            data = json.loads(src.read_text(encoding="utf-8"))
            units: list[dict] = []
            buf: dict | None = None
            for sec in data["sections"]:
                if skip_re and skip_re.search(sec["title"] or ""):
                    continue
                body = "\n\n".join(b["text"] for b in sec["blocks"])
                if not body.strip():
                    continue
                t = ntokens(body)
                if (
                    buf
                    and t < min_t
                    and buf["parent"] == sec["parent"]
                    and buf["tokens"] + t <= max_t
                ):
                    buf["blocks"].extend(sec["blocks"])
                    buf["tokens"] += t
                else:
                    if buf:
                        units.append(buf)
                    buf = {"sec": sec, "blocks": list(sec["blocks"]), "tokens": t,
                           "parent": sec["parent"], "path": sec["path"]}
            if buf:
                units.append(buf)

            prefix_base = f"{data.get('nature') or 'TS'} {data['spec']} v{data['version']}"
            idx = 0
            for unit in units:
                sec = unit["sec"]
                path = unit["path"] or sec["title"]
                prefix = f"{prefix_base} | {path}".strip(" |")
                if unit["tokens"] <= max_t:
                    groups = [unit["blocks"]]
                else:
                    groups = split_oversized(unit["blocks"], max_t, overlap)
                for gi, group in enumerate(groups):
                    body = "\n\n".join(b["text"] for b in group)
                    text = f"{prefix}\n{body}"
                    chunk_id = f"{data['spec']}_{data['version']}_{idx:05d}"
                    chunk_id = re.sub(r"[^0-9A-Za-z._-]+", "-", chunk_id)
                    fout.write(json.dumps({
                        "id": chunk_id,
                        "spec": data["spec"],
                        "spec_type": data.get("nature") or "TS",
                        "version": data["version"],
                        "release": 19,
                        "title": data.get("title", ""),
                        "wg": data.get("wg", ""),
                        "status": data.get("status", ""),
                        "section": sec["number"],
                        "section_title": sec["title"],
                        "section_level": sec["level"],
                        "section_path": path,
                        "parent_section": sec["parent"],
                        "chunk_index": gi,
                        "chunk_of_section": len(groups),
                        "tokens": ntokens(text),
                        "text": text,
                        "source_docx": data["docx"],
                    }, ensure_ascii=False) + "\n")
                    idx += 1
                    total += 1
            fseen.write(rec["spec"] + "\n")
            fseen.flush()
    log.info("chunk done: %d chunks written to %s", total, CHUNKS_FILE)
    return total


def cmd_chunk(args) -> None:
    files = load_json(MANIFEST, [])
    if args.series:
        files = [f for f in files if f["series"] in args.series]
    chunk_files(files, force=args.force)
