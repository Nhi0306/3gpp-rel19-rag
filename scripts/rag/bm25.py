"""BM25 cho corpus 3GPP: tokenize + sparse vector (dung chung index va query).

Cong thuc:
    idf(t)   = ln(1 + (N - df + 0.5) / (df + 0.5))
    doc(t,d) = tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
    query(t) = idf(t)
Dot product giua 2 sparse vector = BM25 score.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

TOKEN_RE = re.compile(r"[^\W_]+(?:[._\-][^\W_]+)*", re.UNICODE)

STOP = frozenset("""
a an the and or of to in for on with by as at is are was were be been being
this that these those from into than then there their its it we you they he
she not no if but so such can could may might will would shall should do does
did done have has had also any all each other more most some only own same
which who whom what when where why how about over under between during before
after above below up down out off again further once here both few very
""".split())


def tokenize(text: str) -> list[str]:
    """Tach token, giu cum nhu 5gs / ng-ran / 23.501 / 4.2.3.

    Token co dau phan cach (., -, _) duoc tach them thanh cac phan de tang recall.
    """
    out: list[str] = []
    for m in TOKEN_RE.finditer(text.lower()):
        tok = m.group(0).strip("._-")
        if not tok or tok in STOP:
            continue
        if len(tok) == 1 and not tok.isdigit():
            continue
        out.append(tok)
        if any(ch in tok for ch in "-._"):
            parts = [p for p in re.split(r"[._\-]", tok) if len(p) > 1 and p not in STOP]
            if parts and parts != [tok]:
                out.extend(parts)
    return out


class BM25Stats:
    """Thong ke corpus: vocab, df, N, avgdl."""

    def __init__(self, data: dict) -> None:
        self.terms: list[str] = data["terms"]
        self.df: list[int] = data["df"]
        self.n_docs: int = int(data["N"])
        self.avgdl: float = float(data["avgdl"]) or 1.0
        self.k1: float = float(data.get("k1", 1.2))
        self.b: float = float(data.get("b", 0.75))
        self.index: dict[str, int] = {t: i for i, t in enumerate(self.terms)}

    @classmethod
    def load(cls, path: Path) -> "BM25Stats":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {"N": self.n_docs, "avgdl": self.avgdl, "k1": self.k1, "b": self.b,
                 "terms": self.terms, "df": self.df},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    # ---- vector hoa ----
    def doc_vector(self, tokens: list[str]) -> tuple[list[int], list[float]]:
        tf = Counter(t for t in tokens if t in self.index)
        if not tf:
            return [], []
        dl = max(1, len(tokens))
        norm = self.k1 * (1 - self.b + self.b * dl / self.avgdl)
        inds: list[int] = []
        vals: list[float] = []
        for term, f in tf.items():
            inds.append(self.index[term])
            vals.append(f * (self.k1 + 1) / (f + norm))
        return inds, vals

    def query_vector(self, text: str) -> tuple[list[int], list[float]]:
        inds: list[int] = []
        vals: list[float] = []
        for term in {t for t in tokenize(text) if t in self.index}:
            i = self.index[term]
            df = self.df[i]
            inds.append(i)
            vals.append(math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5)))
        return inds, vals


def build_stats(chunks_file: Path, log_every: int = 10000, progress=None) -> BM25Stats:
    """Quet JSONL mot lan de tinh df + avgdl."""
    df: dict[str, int] = {}
    n = 0
    total_tokens = 0
    with chunks_file.open("rb") as fh:
        for line in fh:
            if not line.strip():
                continue
            text = json.loads(line.decode("utf-8"))["text"]
            tokens = tokenize(text)
            total_tokens += len(tokens)
            for term in set(tokens):
                df[term] = df.get(term, 0) + 1
            n += 1
            if progress and n % log_every == 0:
                progress(n, len(df))
    terms = sorted(df)
    return BM25Stats({
        "N": n,
        "avgdl": total_tokens / n if n else 1.0,
        "terms": terms,
        "df": [df[t] for t in terms],
    })
