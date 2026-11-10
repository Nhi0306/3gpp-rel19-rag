#!/usr/bin/env python3
"""Deploy web RAG 3GPP Rel-19 len Hugging Face Spaces (Docker).

Chuan bi: them HF_TOKEN (quyen write) vao .env hoac bien moi truong.

    python scripts\\deploy_hf.py --space 3gpp-rel19-rag
    python scripts\\deploy_hf.py --space 3gpp-rel19-rag --wait
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from huggingface_hub import HfApi  # noqa: E402

from config import CFG, ROOT  # noqa: E402

# khong dua len Space: du lieu goc, cache, log, file cau hinh Space
IGNORE = [
    ".git",
    ".git/**",
    ".env",
    ".env.*",
    "Rel-19/**",
    "cache/**",
    "logs/**",
    "deploy/**",
    "**/__pycache__/**",
    "*.pyc",
    "**/*.pyc",
    "*.zip",
    "*.docx",
    "*.sqlite",
    "**/*.sqlite",
    "data/*.json",
    "README.md",
]

SECRETS = ["QDRANT_URL", "QDRANT_API_KEY", "OPENROUTER_API_KEY"]


def _fail(msg: str) -> None:
    print(f"LOI: {msg}", file=sys.stderr)
    raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy RAG len Hugging Face Spaces")
    parser.add_argument("--space", default="3gpp-rel19-rag", help="ten Space (mac dinh 3gpp-rel19-rag)")
    parser.add_argument("--token", default=os.getenv("HF_TOKEN"), help="HF token (mac dinh lay tu HF_TOKEN)")
    parser.add_argument("--profile", default="cheap", choices=["cheap", "balanced", "quality"])
    parser.add_argument("--daily-limit", type=int, default=100)
    parser.add_argument("--rate", type=int, default=10)
    parser.add_argument("--max-chars", type=int, default=2000)
    parser.add_argument("--private", action="store_true", help="tao Space private thay vi public")
    parser.add_argument("--wait", action="store_true", help="cho den khi Space chay xong")
    parser.add_argument("--timeout", type=int, default=900, help="thoi gian cho toi da (giay)")
    args = parser.parse_args()

    token = (args.token or os.getenv("HF_TOKEN") or "").strip()
    for key in SECRETS:
        if not CFG.get(key):
            _fail(f"thieu {key} trong .env")

    # khong co HF_TOKEN -> dung phien da dang nhap bang `hf auth login`
    api = HfApi(token=token or None)
    try:
        me = api.whoami()
    except Exception as exc:  # noqa: BLE001
        _fail("chua dang nhap Hugging Face. Them HF_TOKEN (quyen Write) vao .env "
              "hoac chay `hf auth login`. Chi tiet: " + str(exc))
    user = me.get("name") or me.get("user") or ""
    repo_id = f"{user}/{args.space}"

    print(f"[1/4] Tao Space {repo_id} ...")
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="docker",
        private=args.private,
        exist_ok=True,
    )

    print("[2/4] Upload ma nguon ...")
    api.upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=str(ROOT),
        ignore_patterns=IGNORE,
        commit_message="Deploy 3GPP Rel-19 RAG",
    )
    api.upload_file(
        path_or_fileobj=str(ROOT / "deploy" / "hf" / "README.md"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="space",
        commit_message="Space config",
    )

    print("[3/4] Dat secrets ...")
    for key in SECRETS:
        api.add_space_secret(repo_id=repo_id, key=key, value=CFG[key])
        print(f"  - secret {key}")

    print("[4/4] Dat variables ...")
    variables = {
        "RAG_PROFILE": args.profile,
        "RAG_MAX_REFINE": "1",
        "RAG_DAILY_LIMIT": str(args.daily_limit),
        "RAG_RATE_PER_MIN": str(args.rate),
        "RAG_MAX_QUESTION_CHARS": str(args.max_chars),
        "QDRANT_COLLECTION": CFG["QDRANT_COLLECTION"],
        "SPARSE_COLLECTION": CFG["SPARSE_COLLECTION"],
        "EMBED_PROVIDER": CFG["EMBED_PROVIDER"],
        "EMBED_MODEL": CFG["EMBED_MODEL"],
        "EMBED_DIM": str(CFG["EMBED_DIM"]),
    }
    for key, value in variables.items():
        api.add_space_variable(repo_id=repo_id, key=key, value=value)
        print(f"  - variable {key}={value}")

    page = f"https://huggingface.co/spaces/{repo_id}"
    direct = f"https://{repo_id.replace('/', '-').lower()}.hf.space"
    logs = f"{page}/logs"
    print(f"\nSpace:  {page}")
    print(f"Direct: {direct}")
    print(f"Build:  {logs}")

    if not args.wait:
        print("\nDang build. Mo trang Space de theo doi, hoac chay lai voi --wait.")
        return

    print("\nCho build (toi da %ds) ..." % args.timeout)
    time.sleep(10)  # de HF kip chuyen sang BUILDING sau khi set secrets/variables
    deadline = time.time() + args.timeout
    last = None
    while time.time() < deadline:
        try:
            runtime = api.get_space_runtime(repo_id=repo_id)
            stage = str(getattr(runtime.stage, "value", runtime.stage))
        except Exception as exc:  # noqa: BLE001
            stage = f"khong doc duoc ({exc})"
        if stage != last:
            print(f"  [{time.strftime('%H:%M:%S')}] {stage}")
            last = stage
        if stage == "RUNNING":
            print(f"\nXong: {direct}")
            return
        if stage in ("BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR", "NO_APP_FILE"):
            _fail(f"build loi ({stage}). Xem log: {logs}")
        time.sleep(15)
    _fail(f"het thoi gian cho. Xem log: {logs}")


if __name__ == "__main__":
    main()
