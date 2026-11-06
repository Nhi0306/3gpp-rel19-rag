"""Schema cho structured output cua cac agent."""
from __future__ import annotations

from pydantic import BaseModel, Field


class Plan(BaseModel):
    """Ket qua phan tich cau hoi cua Supervisor."""

    language: str = Field(default="vi", description="vi | en")
    intent: str = Field(
        default="lookup",
        description="lookup | compare | procedure | definition | other",
    )
    specs: list[str] = Field(default_factory=list, description="vd ['23.501']")
    section: str = Field(default="", description="so dieu khoan neu co, vd '5.7.1'")
    keywords: list[str] = Field(default_factory=list, description="thuat ngu ky thuat")
    sub_queries: list[str] = Field(default_factory=list, description="1-3 truy van doc lap")
    needs_history: bool = Field(default=False, description="cau hoi noi tiep can ngu canh")


class Grade(BaseModel):
    """Ket qua cham do lien quan cua evidence."""

    relevant: list[int] = Field(default_factory=list, description="index 1-based cua chunk lien quan")
    sufficient: bool = Field(default=False, description="du thong tin de tra loi chua")
    missing: str = Field(default="", description="thong tin con thieu (neu co)")


class Verdict(BaseModel):
    """Ket qua kiem chung cau tra loi."""

    ok: bool = True
    issues: list[str] = Field(default_factory=list)
    unsupported: list[int] = Field(default_factory=list, description="so hieu citation khong co can cu")


class Citation(BaseModel):
    rank: int
    chunk_id: str
    spec: str
    version: str
    section: str
    section_path: str
    title: str
    score: float
    source: str
    snippet: str
