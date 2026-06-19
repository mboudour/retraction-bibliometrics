"""
shared/utils.py
Common utilities shared across all work package scripts.
"""

import json
import os
from pathlib import Path
import pandas as pd

# ---------------------------------------------------------------------------
# Canonical paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RAW_DATA = ROOT / "data_collection" / "raw_data"

RW_CLEAN   = RAW_DATA / "retraction_watch_clean.csv"
OA_CSV     = RAW_DATA / "openalex_metadata.csv"
OA_JSONL   = RAW_DATA / "openalex_metadata.jsonl"
MERGED_CSV = RAW_DATA / "merged_dataset.csv"


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_retraction_watch(path: Path = RW_CLEAN) -> pd.DataFrame:
    """Load the cleaned Retraction Watch dataset."""
    df = pd.read_csv(path, low_memory=False)
    df["RetractionDate"] = pd.to_datetime(df["RetractionDate"], errors="coerce")
    df["OriginalPaperDate"] = pd.to_datetime(df["OriginalPaperDate"], errors="coerce")
    df["retraction_year"] = df["RetractionDate"].dt.year
    df["pub_year"] = df["OriginalPaperDate"].dt.year
    df["time_to_retraction"] = df["retraction_year"] - df["pub_year"]
    return df


def load_openalex(path: Path = OA_CSV) -> pd.DataFrame:
    """Load the OpenAlex metadata summary CSV."""
    df = pd.read_csv(path, low_memory=False)
    # Normalise DOI to bare form (no https://doi.org/ prefix)
    df["doi_normalised"] = (
        df["doi"]
        .fillna("")
        .str.replace("https://doi.org/", "", regex=False)
        .str.lower()
        .str.strip()
    )
    return df


def load_openalex_jsonl(path: Path = OA_JSONL) -> list[dict]:
    """Load the full OpenAlex JSONL records into a list of dicts."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_merged(path: Path = MERGED_CSV) -> pd.DataFrame:
    """Load the merged (RW + OpenAlex) dataset produced by WP1."""
    return pd.read_csv(path, low_memory=False)


def merge_datasets(rw: pd.DataFrame, oa: pd.DataFrame) -> pd.DataFrame:
    """
    Merge Retraction Watch and OpenAlex on normalised DOI.
    Returns the merged DataFrame.
    """
    merged = rw.merge(
        oa,
        on="doi_normalised",
        how="left",
        suffixes=("_rw", "_oa"),
    )
    return merged


# ---------------------------------------------------------------------------
# Citation-by-year helpers
# ---------------------------------------------------------------------------

def parse_counts_by_year(json_str: str) -> dict[int, int]:
    """Parse the counts_by_year_json column into {year: count}."""
    try:
        records = json.loads(json_str) if isinstance(json_str, str) else []
        return {r["year"]: r["cited_by_count"] for r in records}
    except Exception:
        return {}


def expand_counts_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand the counts_by_year_json column into a long-format DataFrame
    with columns: doi_normalised, year, cited_by_count.
    """
    rows = []
    for _, row in df.iterrows():
        doi = row.get("doi_normalised", "")
        cby = parse_counts_by_year(row.get("counts_by_year_json", "[]"))
        for yr, cnt in cby.items():
            rows.append({"doi_normalised": doi, "year": yr, "cited_by_count": cnt})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reason parsing
# ---------------------------------------------------------------------------

def parse_reasons(reason_str: str) -> list[str]:
    """Split a semicolon-separated reason string into a list of clean tokens."""
    if not isinstance(reason_str, str):
        return []
    return [r.strip() for r in reason_str.split(";") if r.strip()]


def is_misconduct(reason_str: str) -> bool:
    """Return True if the retraction reason includes fraud/misconduct indicators."""
    misconduct_keywords = {
        "Fabrication", "Falsification", "Plagiarism",
        "Misconduct", "Fraud", "Manipulation",
        "Fake Peer Review", "Paper Mill",
    }
    reasons = parse_reasons(reason_str)
    return any(any(kw.lower() in r.lower() for kw in misconduct_keywords) for r in reasons)
