"""Bước 3: giải nén .docx từ các file zip."""
from __future__ import annotations

import os
import subprocess
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rel19.common import jdump, load_json, log
from rel19.paths import INDEX, MANIFEST

PS = Path(__file__).resolve().parent.parent / "_word_convert_one.ps1"


def _convert_legacy_doc(doc: Path, docx: Path, timeout: int = 300) -> bool:
    """Chuyen .doc -> .docx bang Word COM (moi file mot process rieng)."""
    subprocess.run(["taskkill", "/F", "/IM", "WINWORD.EXE"], capture_output=True)
    proc = subprocess.Popen(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
         "-File", str(PS), str(doc), str(docx)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
    )
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
    subprocess.run(["taskkill", "/F", "/IM", "WINWORD.EXE"], capture_output=True)
    return docx.exists() and docx.stat().st_size > 0


def extract_one(rec: dict, force: bool) -> tuple[str, str]:
    src = Path(rec["local_zip"])
    dst = Path(rec["local_docx"])
    if not src.exists():
        return "missing", rec["filename"]
    if dst.exists() and not force and dst.stat().st_size > 0:
        return "skip", ""
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(src) as zf:
            members = [m for m in zf.namelist() if m.lower().endswith(".docx")]
            if not members:
                # file legacy .doc -> giai nen roi convert bang Word
                docs = [m for m in zf.namelist() if m.lower().endswith(".doc")]
                if not docs:
                    return "fail", f"{src.name}: no docx"
                member = max(docs, key=lambda m: zf.getinfo(m).file_size)
                doc = dst.with_suffix(".doc")
                with zf.open(member) as fin, doc.open("wb") as fout:
                    while True:
                        block = fin.read(1 << 20)
                        if not block:
                            break
                        fout.write(block)
                if _convert_legacy_doc(doc, dst):
                    doc.unlink(missing_ok=True)
                    return "ok", ""
                return "fail", f"{src.name}: convert .doc fail"
            member = max(members, key=lambda m: zf.getinfo(m).file_size)
            tmp = dst.with_suffix(".docx.part")
            with zf.open(member) as fin, tmp.open("wb") as fout:
                while True:
                    chunk = fin.read(1 << 20)
                    if not chunk:
                        break
                    fout.write(chunk)
            os.replace(tmp, dst)
        return "ok", ""
    except Exception as exc:  # noqa: BLE001
        return "fail", f"{src.name}: {exc}"


def cmd_extract(args) -> None:
    files = load_json(MANIFEST, [])
    if args.series:
        files = [f for f in files if f["series"] in args.series]
    ok = skip = fail = 0
    errors = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for status, err in pool.map(lambda f: extract_one(f, args.force), files):
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                fail += 1
                errors.append(err)
    jdump(INDEX / "failed-extract.json", errors)
    log.info("extract done: ok=%d skip=%d fail=%d", ok, skip, fail)
