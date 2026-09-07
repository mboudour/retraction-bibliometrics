#!/usr/bin/env python3
"""Shared helpers for Step 1 revision scripts."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd


def project_root() -> Path:
    """Script package must be installed at <project_root>/revision/scripts/."""
    return Path(__file__).resolve().parents[2]


def revision_dirs(root: Path) -> dict[str, Path]:
    base = root / "revision"
    result = {
        "base": base,
        "data": base / "data",
        "output": base / "output",
        "logs": base / "logs",
        "config": base / "config",
    }
    for path in result.values():
        path.mkdir(parents=True, exist_ok=True)
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def find_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    lookup = {str(c).strip().lower(): str(c) for c in columns}
    for candidate in candidates:
        found = lookup.get(candidate.lower())
        if found:
            return found
    return None


def normalise_doi(series: pd.Series) -> pd.Series:
    value = series.fillna("").astype(str).str.strip().str.lower()
    value = value.str.replace(r"^https?://(dx\.)?doi\.org/", "", regex=True)
    value = value.str.replace(r"^doi:\s*", "", regex=True)
    return value.replace("", pd.NA)


def parse_dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=False)


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def latex_escape(value: object) -> str:
    text = str(value)
    for old, new in {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#"}.items():
        text = text.replace(old, new)
    return text


def split_reason_field(value: object) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [item.strip() for item in re.split(r"\s*[;|]\s*", text) if item.strip()]


def save_dual_figure(fig, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=350, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")


def read_master(root: Path) -> pd.DataFrame:
    path = root / "revision" / "data" / "revision_master.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"Run 01_freeze_and_audit.py first: {path}")
    return pd.read_csv(path, low_memory=False)


def source_paths(root: Path) -> dict[str, Path]:
    return {
        "merged": root / "data_collection" / "raw_data" / "merged_dataset.csv",
        "node_metrics": root / "wp4_structural" / "output" / "wp4_node_metrics.csv",
        "citation_long": root / "wp2_citation_decay" / "output" / "wp2_citation_long.csv",
    }
