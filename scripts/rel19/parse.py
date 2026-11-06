"""Bước 4: docx -> JSON (sections/blocks) + markdown."""
from __future__ import annotations

import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from rel19.common import jdump, load_json, log
from rel19.paths import INDEX, MANIFEST, PARSED, TEXT

HEADING_RE = re.compile(r"^heading\s*([1-9])$", re.I)
NUMBER_RE = re.compile(r"^(\d+(?:\.\d+)*)\s+(\S.*)$")


def heading_level(par) -> int:
    name = (par.style.name or "").strip()
    m = HEADING_RE.match(name)
    if m:
        return int(m.group(1))
    m = re.match(r"^H([1-9])$", name)
    return int(m.group(1)) if m else 0


def split_number(text: str) -> tuple[str, str]:
    m = NUMBER_RE.match(text)
    if m:
        return m.group(1), m.group(2)
    return "", text


def clean(text: str) -> str:
    return " ".join(text.replace("\u00a0", " ").split())


def table_to_md(tbl) -> str:
    rows = []
    for row in tbl.rows:
        cells = [clean(c.text) for c in row.cells]
        if any(cells):
            rows.append(cells)
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out)


def parse_docx(path: Path) -> dict:
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = Document(str(path))
    sections: list[dict] = []
    stack: list[tuple[int, str]] = []
    cur: dict | None = None

    def new_section(level: int, number: str, title: str) -> dict:
        parent_path = " > ".join(n for _, n in stack if n)
        path_str = " > ".join([p for p in (parent_path, number) if p])
        return {
            "number": number,
            "title": title,
            "level": level,
            "parent": parent_path,
            "path": path_str,
            "blocks": [],
        }

    for child in doc.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            par = Paragraph(child, doc)
            raw = par.text.strip()
            if not raw:
                continue
            style = (par.style.name or "").lower()
            if style.startswith("toc") or style == "table of contents":
                continue
            level = heading_level(par)
            if level:
                text = clean(raw)
                number, title = split_number(text)
                while stack and stack[-1][0] >= level:
                    stack.pop()
                cur = new_section(level, number, title)
                stack.append((level, number or title))
                sections.append(cur)
            else:
                if cur is None:
                    cur = new_section(0, "", "Document")
                    sections.append(cur)
                cur["blocks"].append({"type": "p", "text": raw})
        elif tag == "tbl":
            md = table_to_md(Table(child, doc))
            if md:
                if cur is None:
                    cur = new_section(0, "", "Document")
                    sections.append(cur)
                cur["blocks"].append({"type": "table", "text": md})

    sections = [s for s in sections if s["blocks"] or s["title"]]
    return {"docx": path.name, "sections": sections}


def section_markdown(sec: dict) -> str:
    head = f"{'#' * min(max(sec['level'], 1), 6)} "
    head += " ".join(x for x in (sec["number"], sec["title"]) if x)
    body = [b["text"] for b in sec["blocks"]]
    return head + "\n\n" + "\n\n".join(body) + "\n"


def parse_job(rec: dict, force: bool = False) -> tuple[str, str, str]:
    """Parse một docx. Chạy trong worker process nên phải ở top-level."""
    src = Path(rec["local_docx"])
    if not src.exists():
        return "missing", src.name, ""
    series = rec["series"]
    out_json = PARSED / f"{series}_series" / (src.stem + ".json")
    out_md = TEXT / f"{series}_series" / (src.stem + ".md")
    if not force and out_json.exists() and out_json.stat().st_mtime >= src.stat().st_mtime:
        return "skip", src.name, ""
    try:
        data = parse_docx(src)
        meta = {
            "spec": rec["spec"],
            "nature": rec.get("nature") or ("TR" if "TR" in (rec.get("title") or "")[:4] else "TS"),
            "version": rec["version"],
            "title": rec.get("title", ""),
            "wg": rec.get("wg", ""),
            "status": rec.get("status", ""),
            "docx": src.name,
        }
        data.update(meta)
        jdump(out_json, data)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        md = [f"# {meta['nature']} {meta['spec']} v{meta['version']} — {meta['title']}\n"]
        md.extend(section_markdown(sec) for sec in data["sections"])
        out_md.write_text("\n".join(md), encoding="utf-8")
        return "ok", src.name, ""
    except Exception as exc:  # noqa: BLE001
        return "fail", src.name, str(exc)


def cmd_parse(args) -> None:
    files = load_json(MANIFEST, [])
    if args.series:
        files = [f for f in files if f["series"] in args.series]
    jobs = max(1, int(getattr(args, "jobs", 1) or 1))
    done = skipped = failed = 0
    errors: list[str] = []
    total = len(files)
    t0 = time.time()

    def handle(status: str, name: str, err: str) -> None:
        nonlocal done, skipped, failed
        if status == "ok":
            done += 1
        elif status == "skip":
            skipped += 1
        elif status == "missing":
            return
        else:
            failed += 1
            errors.append(f"{name}: {err}")
            if len(errors) <= 30:
                log.warning("parse fail %s: %s", name, err)
        if status == "ok" and done % 100 == 0:
            rate = done / max(time.time() - t0, 1)
            log.info("parsed %d docs (%d/%d, %.1f docs/s) ...", done, done + skipped, total, rate)

    if jobs > 1:
        log.info("parse: %d workers, %d files", jobs, total)
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for status, name, err in pool.map(
                parse_job, files, [args.force] * len(files), chunksize=1
            ):
                handle(status, name, err)
    else:
        for rec in files:
            handle(*parse_job(rec, args.force))

    jdump(INDEX / "failed-parse.json", errors)
    log.info("parse done: ok=%d skip=%d fail=%d in %.0fs", done, skipped, failed, time.time() - t0)
