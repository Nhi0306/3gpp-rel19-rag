"""Đường dẫn và hằng số dùng chung cho toàn pipeline."""
from __future__ import annotations

from config import REL19_ROOT, ROOT

# --- thư mục ---------------------------------------------------------------
CACHE = ROOT / "cache"
INDEX = REL19_ROOT / "index"
DOCS = REL19_ROOT / "docs"
PARSED = REL19_ROOT / "parsed"
TEXT = REL19_ROOT / "text"
CHUNKS = REL19_ROOT / "chunks"

# --- file ------------------------------------------------------------------
MANIFEST = INDEX / "manifest.json"
CHUNKS_FILE = CHUNKS / "Rel-19-chunks.jsonl"
CHUNKED_DOCS = CHUNKS / "chunked-docs.txt"
EMBED_PARTS_DIR = CHUNKS / "embed-parts"
EMBED_CURSOR = CHUNKS / "embed-cursor.json"

# --- nguồn dữ liệu ---------------------------------------------------------
STATUS_URL = (
    "https://www.3gpp.org/index.php"
    "?option=com_a3dynpage&view=a3dynpage&code=status-report.htm"
)
FTP_BASE = "https://www.3gpp.org/ftp/Specs/latest/Rel-19"
SERIES = list(range(21, 39))
