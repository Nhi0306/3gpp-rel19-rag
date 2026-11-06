"""Giữ máy luôn thức và tự tắt máy khi pipeline chạy xong.

Cách dùng:
    python scripts/watch_and_shutdown.py --log logs/embed_f16.log

- Gọi SetThreadExecutionState(...) để Windows không sleep/hibernate trong lúc chờ.
- Theo dõi file log; khi thấy dòng "embed done:" thì hẹn giờ tắt máy.
- Nếu job lỗi/dừng giữa chừng (log không đổi quá lâu) thì KHÔNG tắt máy.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002


def log(msg: str, path: Path) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def keep_awake() -> None:
    ctypes.windll.kernel32.SetThreadExecutionState(
        ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
    )


def allow_sleep() -> None:
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)


def read_tail(path: Path, size: int = 200_000) -> str:
    try:
        with path.open("rb") as fh:
            fh.seek(0, 2)
            end = fh.tell()
            fh.seek(max(0, end - size))
            return fh.read().decode("utf-8", errors="replace")
    except (FileNotFoundError, PermissionError):
        return ""


def total_chunks() -> int:
    total = 0
    parts_dir = Path("Rel-19/chunks/embed-parts")
    if parts_dir.exists():
        for p in sorted(parts_dir.glob("part*.json")):
            try:
                total += int(json.loads(p.read_text(encoding="utf-8")).get("chunks", 0))
            except (json.JSONDecodeError, OSError, ValueError):
                pass
    return total


def report(args, state_path: Path, started: float) -> None:
    """Ghi file tong ket de nguoi dung xem lai sau khi may tat."""
    total = total_chunks()
    elapsed = time.time() - started
    text = (
        "3GPP Rel-19 - embed hoan tat\n"
        f"thoi diem   : {datetime.now():%Y-%m-%d %H:%M:%S}\n"
        f"chunks      : {total}\n"
        f"thoi gian   : {elapsed/3600:.2f} gio\n"
        f"collection  : 3gpp_rel19\n"
        f"log         : {args.log}\n"
    )
    out = Path("logs/DONE-3gpp-rel19.txt")
    out.write_text(text, encoding="utf-8")
    log(f"da ghi bao cao: {out} ({total} chunks)", state_path)


def notify(args) -> None:
    """Hien thi thong bao desktop (best-effort, khong chan viec tat may)."""
    total = total_chunks()
    ps = (
        "Add-Type -AssemblyName System.Windows.Forms,System.Drawing;"
        "$n=New-Object System.Windows.Forms.NotifyIcon;"
        "$n.Icon=[System.Drawing.SystemIcons]::Information;"
        "$n.Visible=$true;"
        f"$n.ShowBalloonTip(10000,'3GPP Rel-19',"
        f"'Embed xong: {total} chunks. May se tat sau {args.delay}s.','Info');"
        "Start-Sleep -Seconds 12;$n.Dispose()"
    )
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            check=False, timeout=40,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:  # noqa: BLE001
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="logs/embed_f16.log")
    ap.add_argument("--pattern", default="embed done:")
    ap.add_argument("--delay", type=int, default=120, help="giây chờ trước khi tắt máy")
    ap.add_argument("--stale-min", type=int, default=40,
                    help="nếu log không đổi quá N phút thì coi như job chết, không tắt máy")
    ap.add_argument("--state", default="logs/shutdown-watch.log")
    args = ap.parse_args()

    log_path = Path(args.log)
    state_path = Path(args.state)
    state_path.parent.mkdir(parents=True, exist_ok=True)

    keep_awake()
    start_time = time.time()
    log(f"watcher started, log={log_path}, pattern='{args.pattern}'", state_path)

    last_size = -1
    last_change = time.time()
    while True:
        keep_awake()  # refresh mỗi vòng để chắc chắn không sleep
        tail = read_tail(log_path)
        if args.pattern in tail:
            log("pipeline finished -> shutting down", state_path)
            report(args, state_path, start_time)
            notify(args)
            allow_sleep()
            subprocess.run(
                ["shutdown", "/s", "/t", str(args.delay), "/c",
                 "3GPP Rel-19: embed finished, may se tat."],
                check=False,
            )
            return

        if log_path.exists():
            size = log_path.stat().st_size
            if size != last_size:
                last_size = size
                last_change = time.time()
        elif last_size != -2:
            last_size = -2
            last_change = time.time()

        if time.time() - last_change > args.stale_min * 60:
            log(f"log khong doi qua {args.stale_min} phut -> job coi nhu dung, KHONG tat may",
                state_path)
            allow_sleep()
            return
        time.sleep(10)


if __name__ == "__main__":
    main()
