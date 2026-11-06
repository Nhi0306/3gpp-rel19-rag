"""Bước 1: crawl status report + FTP listing -> manifest.json + CSV/XLSX."""
from __future__ import annotations

import csv
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bs4 import BeautifulSoup

from rel19.common import cached_text, http_get, jdump, load_json, log, parse_filename, parse_size
from rel19.paths import CACHE, DOCS, FTP_BASE, INDEX, MANIFEST, REL19_ROOT, SERIES, STATUS_URL

INDEX_COLS = [
    "series", "spec", "version", "nature", "title", "wg", "status",
    "filename", "size_bytes", "date", "url", "local_zip",
]


def parse_status_report(html: str) -> dict[str, dict]:
    soup = BeautifulSoup(html, "lxml")
    out: dict[str, dict] = {}
    for anchor, section in (("activeRel-19", "active"), ("deadRel-19", "dead")):
        table = soup.find("table", id=f"a3dyntab-{anchor}")
        if table is None:
            log.warning("table %s not found", anchor)
            continue
        for tr in table.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) < 5:
                continue
            a = tds[1].find("a")
            num = (a.get_text(strip=True) if a else tds[1].get_text(strip=True)).strip()
            if not re.match(r"^\d{2}\.\d{3}", num):
                continue
            out[num] = {
                "nature": tds[0].get_text(strip=True),
                "title": " ".join(tds[2].get_text(" ", strip=True).split()),
                "version": tds[3].get_text(strip=True),
                "wg": tds[4].get_text(strip=True),
                "section": section,
            }
    return out


def parse_listing(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    rows = []
    for tr in soup.find_all("tr"):
        a = tr.find("a", href=re.compile(r"\.zip$", re.I))
        if a is None:
            continue
        href = a["href"]
        if href.startswith("/"):
            href = "https://www.3gpp.org" + href
        tds = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
        rows.append(
            {
                "url": href,
                "filename": href.rsplit("/", 1)[-1],
                "date": tds[3] if len(tds) > 3 else "",
                "size_bytes": parse_size(tds[4]) if len(tds) > 4 else 0,
            }
        )
    return rows


def fetch_spec_title(spec: str) -> str:
    tokens = spec.split("-")
    base = tokens[0].replace(".", "")
    candidates = ["-".join([base] + tokens[1:i]) for i in range(len(tokens), 0, -1)]
    for candidate in candidates:
        url = f"https://www.3gpp.org/DynaReport/{candidate}.htm"
        try:
            html = http_get(url, timeout=30, retries=1).text
        except Exception:  # noqa: BLE001
            continue
        soup = BeautifulSoup(html, "lxml")
        node = soup.find(id="titleVal")
        if node:
            return " ".join(node.get_text(" ", strip=True).split())
    return ""


def write_index(files: list[dict]) -> None:
    csv_path = INDEX / "Rel-19-index.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=INDEX_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(files)
    log.info("wrote %s", csv_path)

    try:
        from openpyxl import Workbook
    except ImportError:
        log.warning("openpyxl not installed -> skip xlsx")
        return
    wb = Workbook(write_only=True)
    ws = wb.create_sheet("Rel-19")
    ws.append(INDEX_COLS)
    for f in files:
        ws.append([f.get(c, "") for c in INDEX_COLS])
    xlsx = INDEX / "Rel-19-index.xlsx"
    wb.save(xlsx)
    log.info("wrote %s", xlsx)


def cmd_manifest(args) -> None:
    INDEX.mkdir(parents=True, exist_ok=True)

    log.info("crawling status report ...")
    html = cached_text(STATUS_URL, CACHE / "status-report.htm", force=args.force)
    specs = parse_status_report(html)
    log.info("status report: %d specs (active+dead)", len(specs))
    if len(specs) < 100:
        (CACHE / "status-report.htm").unlink(missing_ok=True)
        raise SystemExit(
            f"status report parse failed ({len(specs)} specs). "
            "Cache da bi xoa, chay lai de tai lai trang."
        )

    files: list[dict] = []
    for s in SERIES:
        url = f"{FTP_BASE}/{s}_series/"
        try:
            page = cached_text(url, CACHE / "ftp-listing" / f"{s}_series.htm", force=args.force)
        except Exception as exc:  # noqa: BLE001
            log.warning("series %s: %s", s, exc)
            continue
        rows = parse_listing(page)
        selected = []
        for row in rows:
            meta = parse_filename(row["filename"])
            if meta is None:
                continue
            row.update(meta)
            row["series"] = s
            row["local_zip"] = str(REL19_ROOT / f"{s}_series" / row["filename"])
            row["local_docx"] = str(DOCS / f"{s}_series" / (Path(row["filename"]).stem + ".docx"))
            selected.append(row)
        files.extend(selected)
        log.info("series %s: %d zip files", s, len(selected))

    for f in files:
        info = specs.get(f["spec"], {})
        f["nature"] = info.get("nature", "")
        f["title"] = info.get("title", "")
        f["status"] = info.get("section", "not-in-status-report")
        f["wg"] = info.get("wg", "")
        f["sr_version"] = info.get("version", "")

    # bổ sung title cho spec multipart không có trong status report
    missing = sorted({f["spec"] for f in files if not f["title"]})
    if missing:
        titles = load_json(CACHE / "titles.json", {})
        todo = [s for s in missing if s not in titles]
        if todo:
            log.info("fetching %d missing titles from DynaReport ...", len(todo))
            with ThreadPoolExecutor(max_workers=6) as pool:
                for spec, title in zip(todo, pool.map(fetch_spec_title, todo)):
                    titles[spec] = title
            jdump(CACHE / "titles.json", titles)
        for f in files:
            if not f["title"]:
                f["title"] = titles.get(f["spec"], "")

    files.sort(key=lambda x: (x["series"], x["spec"], x["filename"]))
    jdump(MANIFEST, files)
    write_index(files)
    total = sum(f["size_bytes"] for f in files)
    log.info("manifest: %d files, %.2f GB", len(files), total / 1024 ** 3)
