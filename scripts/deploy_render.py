"""Dat bien moi truong tren Render roi deploy lai service, qua Render API.

Chuẩn bị: tạo API key tại https://dashboard.render.com/account/api-keys
rồi thêm vào .env:  RENDER_API_KEY=rnd_...

Chạy:
    python scripts\\deploy_render.py                 # đặt biến + deploy + chờ live
    python scripts\\deploy_render.py --no-deploy     # chỉ đặt biến
    python scripts\\deploy_render.py --service ten   # chọn service khác
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://api.render.com/v1"
ROOT = Path(__file__).resolve().parents[1]
SERVICE = "threegpp-rel19-rag"

WANTED = {
    "RAG_PROFILE": "cheap",
    "SYNTH_MODEL": "google/gemini-2.5-flash",
    "RAG_MAX_REFINE": "1",
    "RAG_DAILY_LIMIT": "100",
    "RAG_RATE_PER_MIN": "10",
    "RAG_MAX_QUESTION_CHARS": "2000",
}
SECRETS = ("QDRANT_URL", "QDRANT_API_KEY", "OPENROUTER_API_KEY")


def load_env() -> dict[str, str]:
    out: dict[str, str] = {}
    path = ROOT / ".env"
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        out[key.strip()] = val.strip().strip('"').strip("'")
    return out


def api(key: str, method: str, path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        BASE + path,
        method=method,
        data=data,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:600]
        raise SystemExit(f"Render API lỗi {exc.code} ({method} {path}): {detail}") from exc


def find_service(key: str, name: str) -> dict:
    items = api(key, "GET", "/services?limit=100")
    services = [it.get("service", it) for it in items]
    for svc in services:
        if svc.get("name") == name:
            return svc
    names = ", ".join(s.get("name", "?") for s in services) or "(trống)"
    raise SystemExit(f"Không thấy service '{name}'. Đang có: {names}")


def set_env(key: str, service_id: str, values: dict[str, str]) -> None:
    for name, value in values.items():
        api(key, "PUT", f"/services/{service_id}/env-vars/{name}", {"value": value})
        if name in SECRETS:
            shown = f"***{value[-4:]}" if len(value) > 4 else "***"
        else:
            shown = value
        print(f"  {name} = {shown}")


def trigger(key: str, service_id: str) -> str:
    deploy = api(key, "POST", f"/services/{service_id}/deploys", {"clearCache": "do_not_clear"})
    return deploy.get("id") or deploy.get("deploy", {}).get("id", "")


def wait_live(key: str, service_id: str, deploy_id: str, timeout: int = 900) -> str:
    done = {"live", "build_failed", "canceled", "deactivated", "pre_deploy_failed"}
    start = time.time()
    status = "?"
    while time.time() - start < timeout:
        time.sleep(15)
        dep = api(key, "GET", f"/services/{service_id}/deploys/{deploy_id}")
        status = dep.get("status", "?")
        print(f"  [{int(time.time() - start):4d}s] {status}")
        if status in done:
            break
    return status


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--service", default=SERVICE)
    ap.add_argument("--no-deploy", action="store_true")
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    env = load_env()
    key = env.get("RENDER_API_KEY", "").strip()
    if not key:
        print("Thiếu RENDER_API_KEY trong .env. Tạo tại "
              "https://dashboard.render.com/account/api-keys rồi thêm dòng:\n"
              "  RENDER_API_KEY=rnd_...")
        return 1

    missing = [k for k in SECRETS if not env.get(k)]
    if missing:
        print("Thiếu trong .env: " + ", ".join(missing))
        return 1

    svc = find_service(key, args.service)
    sid = svc.get("id", "")
    print(f"Service: {svc.get('name')} ({sid})")

    values = {k: env[k] for k in SECRETS}
    values.update(WANTED)
    print("Đặt biến môi trường:")
    set_env(key, sid, values)

    if args.no_deploy:
        print("Xong (chưa deploy).")
        return 0

    deploy_id = trigger(key, sid)
    print(f"Đã trigger deploy {deploy_id}, chờ build...")
    status = wait_live(key, sid, deploy_id, args.timeout)
    print(f"Trạng thái deploy: {status}")

    svc = api(key, "GET", f"/services/{sid}")
    url = ((svc.get("serviceDetails") or {}).get("url") or "").rstrip("/")
    if not url:
        print("Không lấy được URL service.")
        return 0 if status == "live" else 1

    print(f"URL: {url}")
    if status != "live":
        return 1

    for path in ("/api/health", "/api/diag"):
        try:
            with urllib.request.urlopen(url + path, timeout=120) as resp:
                print(f"{path}: {resp.read().decode('utf-8', 'replace')}")
        except Exception as exc:  # noqa: BLE001
            print(f"{path} lỗi: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
