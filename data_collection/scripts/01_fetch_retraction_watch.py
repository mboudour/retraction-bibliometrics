#!/usr/bin/env python3
"""
Script 1: Fetch and clean the Retraction Watch dataset from Crossref GitLab.

This script downloads the full Retraction Watch CSV (updated daily by Crossref),
filters it to true retractions only, keeps records with a valid OriginalPaperDOI,
and saves a cleaned CSV ready for OpenAlex enrichment.

No API key required.

Usage:
    python3 01_fetch_retraction_watch.py [--year-min YYYY] [--year-max YYYY]

Output:
    retraction_watch_raw.csv       -- full download (all update types)
    retraction_watch_clean.csv     -- filtered: Retractions only, with valid DOI
"""

import argparse
import os
import sys
import urllib.request
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RW_URL = (
    "https://gitlab.com/crossref/retraction-watch-data/-/raw/main/retraction_watch.csv"
)
RAW_FILE = "retraction_watch_raw.csv"
CLEAN_FILE = "retraction_watch_clean.csv"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Download and clean Retraction Watch data.")
parser.add_argument(
    "--year-min",
    type=int,
    default=None,
    help="Keep only retractions with OriginalPaperDate >= this year (e.g. 2000).",
)
parser.add_argument(
    "--year-max",
    type=int,
    default=None,
    help="Keep only retractions with OriginalPaperDate <= this year (e.g. 2022).",
)
args = parser.parse_args()


# ---------------------------------------------------------------------------
# Step 1: Download the raw CSV (skip if already present)
# ---------------------------------------------------------------------------
def download_with_progress(url: str, dest: str) -> None:
    """Download a URL to dest, printing a simple progress indicator."""
    print(f"Downloading: {url}")
    print(f"Saving to  : {dest}")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100.0, downloaded / total_size * 100)
            mb = downloaded / 1_048_576
            sys.stdout.write(f"\r  {pct:5.1f}%  ({mb:.1f} MB downloaded)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()  # newline after progress bar


if os.path.exists(RAW_FILE):
    print(f"[INFO] Raw file already exists: {RAW_FILE}  (delete to re-download)")
else:
    download_with_progress(RW_URL, RAW_FILE)

print(f"[OK] Raw file size: {os.path.getsize(RAW_FILE) / 1_048_576:.1f} MB")

# ---------------------------------------------------------------------------
# Step 2: Load and inspect
# ---------------------------------------------------------------------------
print("\nLoading CSV …")
df = pd.read_csv(RAW_FILE, low_memory=False)
print(f"  Total rows (all update types): {len(df):,}")
print(f"  Columns: {list(df.columns)}")

# ---------------------------------------------------------------------------
# Step 3: Filter to true Retractions only
# ---------------------------------------------------------------------------
df_ret = df[df["RetractionNature"] == "Retraction"].copy()
print(f"\n  Rows after filtering to 'Retraction': {len(df_ret):,}")

# ---------------------------------------------------------------------------
# Step 4: Keep only records with a usable OriginalPaperDOI
# ---------------------------------------------------------------------------
INVALID_DOI_VALUES = {"unavailable", "Unavailable", "", "nan"}

df_ret["OriginalPaperDOI"] = df_ret["OriginalPaperDOI"].fillna("").str.strip()
has_doi = ~df_ret["OriginalPaperDOI"].isin(INVALID_DOI_VALUES)
df_ret = df_ret[has_doi].copy()
print(f"  Rows after requiring a valid OriginalPaperDOI: {len(df_ret):,}")

# ---------------------------------------------------------------------------
# Step 5: Normalise DOI to lowercase (OpenAlex expects lowercase DOIs)
# ---------------------------------------------------------------------------
df_ret["doi_normalised"] = df_ret["OriginalPaperDOI"].str.lower().str.strip()

# ---------------------------------------------------------------------------
# Step 6: Parse OriginalPaperDate and apply optional year filters
# ---------------------------------------------------------------------------
df_ret["OriginalPaperDate"] = pd.to_datetime(
    df_ret["OriginalPaperDate"], errors="coerce"
)
df_ret["pub_year"] = df_ret["OriginalPaperDate"].dt.year

if args.year_min is not None:
    df_ret = df_ret[df_ret["pub_year"] >= args.year_min]
    print(f"  Rows after year_min={args.year_min}: {len(df_ret):,}")

if args.year_max is not None:
    df_ret = df_ret[df_ret["pub_year"] <= args.year_max]
    print(f"  Rows after year_max={args.year_max}: {len(df_ret):,}")

# ---------------------------------------------------------------------------
# Step 7: Save cleaned file
# ---------------------------------------------------------------------------
df_ret.to_csv(CLEAN_FILE, index=False)
print(f"\n[OK] Saved cleaned retraction list: {CLEAN_FILE}")
print(f"     {len(df_ret):,} records ready for OpenAlex enrichment.")

# Quick summary
print("\n--- Summary of cleaned dataset ---")
print(f"  Total retractions:           {len(df_ret):,}")
print(f"  Unique normalised DOIs:      {df_ret['doi_normalised'].nunique():,}")
print(f"  Publication year range:      {int(df_ret['pub_year'].min())} – {int(df_ret['pub_year'].max())}")
print(f"  Top 5 retraction reasons:")
reasons = (
    df_ret["Reason"]
    .str.split(";")
    .explode()
    .str.strip()
    .value_counts()
    .head(5)
)
for reason, count in reasons.items():
    print(f"    {count:6,}  {reason}")
