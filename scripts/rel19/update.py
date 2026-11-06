"""Cap nhat du lieu: kiem tra 3GPP co spec moi/cap nhat roi nap phan thay doi.

Dung cho chay tay:
    python scripts/3gpp_rel19_pipeline.py update
    python scripts/3gpp_rel19_pipeline.py update --specs 24.229 33.511

Hoac chay tu dong hang tuan bang Task Scheduler (xem scripts/install_weekly_task.ps1).
"""
from __future__ import annotations

import json
import time

from config import validate
from rel19 import download, extract, manifest, parse
from rel19.chunk import chunk_files
from rel19.common import jdump, load_json, log
from rel19.embed import embed_range
from rel19.paths import CHUNKED_DOCS, CHUNKS_FILE, EMBED_CURSOR, INDEX, MANIFEST

UPDATE_REPORT = INDEX / "last-update.json"


def _forget_specs(specs: set[str]) -> None:
    """Bo cac spec khoi chunked-docs de chunk lai."""
    if not CHUNKED_DOCS.exists():
        return
    lines = [x for x in CHUNKED_DOCS.read_text(encoding="utf-8").splitlines() if x not in specs]
    CHUNKED_DOCS.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _delete_specs(specs: set[str]) -> None:
    """Xoa point cu cua cac spec sap duoc nap lai."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import FieldCondition, Filter, MatchAny, PayloadSchemaType

    from config import CFG

    client = QdrantClient(
        url=CFG["QDRANT_URL"], api_key=CFG["QDRANT_API_KEY"],
        timeout=CFG["QDRANT_TIMEOUT"], prefer_grpc=CFG["QDRANT_PREFER_GRPC"],
    )
    coll = CFG["QDRANT_COLLECTION"]
    try:
        try:
            client.create_payload_index(coll, "spec", field_schema=PayloadSchemaType.KEYWORD)
        except Exception:  # noqa: BLE001  (index da ton tai)
            pass
        client.delete(
            coll,
            points_selector=Filter(
                must=[FieldCondition(key="spec", match=MatchAny(any=sorted(specs)))]
            ),
        )
        log.info("update: da xoa point cu cua %d spec", len(specs))
    finally:
        client.close()


def _specs_in_range(start: int, end: int) -> set[str]:
    """Cac spec co chunk trong khoang byte [start, end)."""
    specs: set[str] = set()
    if not CHUNKS_FILE.exists() or start >= end:
        return specs
    with CHUNKS_FILE.open("rb") as fh:
        fh.seek(start)
        if start > 0:
            fh.seek(start - 1)
            if fh.read(1) != b"\n":
                fh.readline()
        while fh.tell() < end:
            line = fh.readline()
            if not line:
                break
            try:
                specs.add(json.loads(line.decode("utf-8"))["spec"])
            except (json.JSONDecodeError, KeyError):
                continue
    return specs


def _embed_pending() -> int:
    """Nap not phan da chunk nhung chua kip embed (do lan chay truoc bi loi)."""
    if not CHUNKS_FILE.exists():
        return 0
    size = CHUNKS_FILE.stat().st_size
    start = int(load_json(EMBED_CURSOR, {}).get("pos", 0))
    if start >= size:
        return 0
    log.info("update: con %.1f MB chua embed, nap tiep ...", (size - start) / 1024 ** 2)
    specs = _specs_in_range(start, size)
    if specs:
        _delete_specs(specs)
    return embed_range(start, size, progress=EMBED_CURSOR)


def cmd_update(args) -> None:
    validate("embed")
    started = time.time()
    _embed_pending()
    old_files = load_json(MANIFEST, [])
    if not old_files:
        raise SystemExit("chua co manifest; chay `manifest` truoc")
    old = {f["spec"]: f for f in old_files}

    specs_arg = list(getattr(args, "specs", None) or [])
    removed: list[str] = []

    if specs_arg:
        wanted = set(specs_arg)
        changed = [f for f in old_files if f["spec"] in wanted]
        missing = wanted - {f["spec"] for f in changed}
        if missing:
            log.warning("update: khong thay spec trong manifest: %s", ", ".join(sorted(missing)))
        log.info("update: refresh thu cong %d spec", len(changed))
    else:
        log.info("update: crawl lai status report + FTP listing ...")
        args.force = True
        manifest.cmd_manifest(args)
        new_files = load_json(MANIFEST, [])
        changed = [
            f for f in new_files
            if f["spec"] not in old or old[f["spec"]]["version"] != f["version"]
        ]
        new_specs = {f["spec"] for f in new_files}
        removed = [s for s in old if s not in new_specs]
        if removed:
            log.info("update: %d spec khong con trong Rel-19: %s", len(removed),
                     ", ".join(removed[:10]))
        if not changed:
            log.info("update: khong co du lieu moi (%d spec)", len(new_files))
            jdump(UPDATE_REPORT, {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "changed": [], "removed": removed, "chunks": 0,
                "elapsed_s": round(time.time() - started, 1),
            })
            return

    log.info("update: %d spec moi/cap nhat: %s", len(changed),
             ", ".join(f"{f['spec']} v{f['version']}" for f in changed[:15]))

    # 1) tai + giai nen + parse
    chunked = (
        set(CHUNKED_DOCS.read_text(encoding="utf-8").splitlines())
        if CHUNKED_DOCS.exists() else set()
    )
    updated: list[dict] = []
    for rec in changed:
        old_ver = old.get(rec["spec"], {}).get("version")
        already_chunked = rec["spec"] in chunked and old_ver == rec["version"]
        status, _, err = download.download_one(rec, force=False)
        if status not in ("ok", "skip"):
            log.warning("update: download fail %s: %s", rec["filename"], err)
            continue
        status, err = extract.extract_one(rec, force=False)
        if status not in ("ok", "skip"):
            log.warning("update: extract fail %s: %s", rec["filename"], err)
            continue
        status, name, err = parse.parse_job(rec, force=True)
        if status != "ok":
            log.warning("update: parse fail %s: %s", name, err)
            continue
        if already_chunked:
            # refresh parse thoi, giu nguyen chunk/point da co (tranh trung lap)
            log.info("update: %s v%s da co chunk, chi refresh parse", rec["spec"], rec["version"])
            continue
        updated.append(rec)

    if not updated:
        log.info("update: khong co spec nao can chunk lai")
        jdump(UPDATE_REPORT, {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "changed": [], "removed": removed, "chunks": 0,
            "elapsed_s": round(time.time() - started, 1),
        })
        return

    specs = {f["spec"] for f in updated}

    # 2) chunk lai (append vao JSONL)
    start_byte = CHUNKS_FILE.stat().st_size if CHUNKS_FILE.exists() else 0
    _forget_specs(specs)
    added = chunk_files(updated)

    # 3) xoa point cu cua cac spec nay, roi embed phan moi append
    _delete_specs(specs)
    end_byte = CHUNKS_FILE.stat().st_size
    start = min(int(load_json(EMBED_CURSOR, {}).get("pos", start_byte)), start_byte)
    uploaded = embed_range(start, end_byte, progress=EMBED_CURSOR) if end_byte > start else 0

    report = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "changed": [{"spec": f["spec"], "version": f["version"]} for f in updated],
        "removed": removed,
        "chunks": uploaded,
        "chunks_added": added,
        "elapsed_s": round(time.time() - started, 1),
    }
    jdump(UPDATE_REPORT, report)
    log.info("update done: %d spec, %d chunks moi, %.0fs", len(updated), uploaded,
             time.time() - started)
