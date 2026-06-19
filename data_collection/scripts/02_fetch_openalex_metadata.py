#!/usr/bin/env python3
"""
Script 2: Fetch OpenAlex metadata for retracted papers identified in Script 1.

Reads retraction_watch_clean.csv (produced by Script 1), queries the OpenAlex
API in batches of up to 50 DOIs per request, and saves the enriched metadata
to openalex_metadata.jsonl (one JSON record per line) and a summary CSV.

API key:
    Get a free key at https://openalex.org/settings/api
    Set it via the OPENALEX_API_KEY environment variable, or pass --api-key.
    Without a key the script still works but is rate-limited to ~100k req/day
    (polite pool via mailto= parameter).

Usage:
    python3 02_fetch_openalex_metadata.py [--api-key KEY] [--email you@example.com]
                                          [--batch-size 50] [--max-records N]
                                          [--resume]

Output:
    openalex_metadata.jsonl   -- one JSON object per matched work (full record)
    openalex_metadata.csv     -- flattened summary (key fields only)
    openalex_unmatched.txt    -- DOIs not found in OpenAlex
"""

import argparse
import json
import os
import time
import urllib.parse
import urllib.request
import urllib.error
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CLEAN_FILE = "retraction_watch_clean.csv"
JSONL_FILE = "openalex_metadata.jsonl"
CSV_FILE = "openalex_metadata.csv"
UNMATCHED_FILE = "openalex_unmatched.txt"
PROGRESS_FILE = "openalex_progress.txt"   # tracks last completed batch index

OPENALEX_BASE = "https://api.openalex.org/works"

# Fields to extract into the summary CSV
SUMMARY_FIELDS = [
    "id", "doi", "title", "publication_year", "type",
    "cited_by_count", "is_retracted",
    "primary_location_source_display_name",
    "primary_location_source_issn_l",
    "open_access_is_oa",
    "authorships_count",
    "concepts_top",
    "topics_top",
    "referenced_works_count",
    "counts_by_year",
]

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Fetch OpenAlex metadata for retracted papers.")
parser.add_argument("--api-key", default=os.environ.get("OPENALEX_API_KEY", ""),
                    help="OpenAlex API key (or set OPENALEX_API_KEY env var).")
parser.add_argument("--email", default="researcher@example.com",
                    help="Your email for the polite pool (used as mailto= param).")
parser.add_argument("--batch-size", type=int, default=50,
                    help="DOIs per API request (max 50 for OR filter).")
parser.add_argument("--max-records", type=int, default=None,
                    help="Stop after fetching N records (useful for testing).")
parser.add_argument("--resume", action="store_true",
                    help="Resume from last completed batch (reads PROGRESS_FILE).")
parser.add_argument("--sleep", type=float, default=0.12,
                    help="Seconds to sleep between requests (default 0.12 ≈ 8 req/s).")
args = parser.parse_args()

BATCH_SIZE = min(args.batch_size, 50)   # OpenAlex OR filter hard limit is 100 values

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_url(dois: list[str], api_key: str, email: str) -> str:
    """Build an OpenAlex API URL for a batch of DOIs."""
    doi_filter = "|".join(urllib.parse.quote(d, safe=":/") for d in dois)
    params = {
        "filter": f"doi:{doi_filter}",
        "per_page": str(len(dois)),
        "select": (
            "id,doi,title,publication_year,type,cited_by_count,is_retracted,"
            "primary_location,open_access,authorships,concepts,topics,"
            "referenced_works,counts_by_year,abstract_inverted_index"
        ),
    }
    if api_key:
        params["api_key"] = api_key
    else:
        params["mailto"] = email
    return OPENALEX_BASE + "?" + urllib.parse.urlencode(params)


def fetch_json(url: str, retries: int = 5) -> dict:
    """GET a URL and return parsed JSON, with exponential backoff on errors."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "RetractionsStudy/1.0 (research project)"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 2 ** attempt * 5
                print(f"  [429 rate limit] sleeping {wait}s …")
                time.sleep(wait)
            elif e.code in (500, 502, 503, 504):
                wait = 2 ** attempt * 2
                print(f"  [HTTP {e.code}] sleeping {wait}s …")
                time.sleep(wait)
            else:
                raise
        except Exception as exc:
            wait = 2 ** attempt * 2
            print(f"  [Error: {exc}] sleeping {wait}s …")
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch after {retries} attempts: {url[:120]}")


def flatten_work(w: dict) -> dict:
    """Flatten a single OpenAlex work record into a flat dict for the CSV."""
    loc = w.get("primary_location") or {}
    src = loc.get("source") or {}
    oa = w.get("open_access") or {}
    auths = w.get("authorships") or []
    concepts = w.get("concepts") or []
    topics = w.get("topics") or []
    refs = w.get("referenced_works") or []
    cby = w.get("counts_by_year") or []

    top_concept = concepts[0].get("display_name", "") if concepts else ""
    top_topic = topics[0].get("display_name", "") if topics else ""

    return {
        "openalex_id": w.get("id", ""),
        "doi": w.get("doi", ""),
        "title": w.get("title", ""),
        "publication_year": w.get("publication_year"),
        "type": w.get("type", ""),
        "cited_by_count": w.get("cited_by_count"),
        "is_retracted": w.get("is_retracted"),
        "journal_name": src.get("display_name", ""),
        "journal_issn_l": src.get("issn_l", ""),
        "is_oa": oa.get("is_oa"),
        "authorships_count": len(auths),
        "top_concept": top_concept,
        "top_topic": top_topic,
        "referenced_works_count": len(refs),
        "has_abstract": bool(w.get("abstract_inverted_index")),
        "counts_by_year_json": json.dumps(cby),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Load cleaned retraction list
print(f"Loading {CLEAN_FILE} …")
df = pd.read_csv(CLEAN_FILE, low_memory=False)
dois_all = df["doi_normalised"].dropna().unique().tolist()
print(f"  {len(dois_all):,} unique DOIs to look up in OpenAlex")

if args.max_records:
    dois_all = dois_all[: args.max_records]
    print(f"  [--max-records] Limiting to first {len(dois_all):,} DOIs")

# Resume support
start_batch = 0
if args.resume and Path(PROGRESS_FILE).exists():
    start_batch = int(Path(PROGRESS_FILE).read_text().strip()) + 1
    print(f"  [--resume] Resuming from batch index {start_batch}")

# Prepare output files
jsonl_mode = "a" if args.resume else "w"
csv_rows: list[dict] = []
unmatched: list[str] = []

# Load existing CSV rows if resuming
if args.resume and Path(CSV_FILE).exists():
    csv_rows = pd.read_csv(CSV_FILE).to_dict("records")
    print(f"  [--resume] Loaded {len(csv_rows):,} existing CSV rows")

# Batch the DOIs
batches = [dois_all[i: i + BATCH_SIZE] for i in range(0, len(dois_all), BATCH_SIZE)]
total_batches = len(batches)
total_fetched = 0

print(f"\nStarting fetch: {total_batches:,} batches of up to {BATCH_SIZE} DOIs each")
print(f"  API key present: {'yes' if args.api_key else 'no (using polite pool)'}")
print(f"  Sleep between requests: {args.sleep}s\n")

with open(JSONL_FILE, jsonl_mode, encoding="utf-8") as jsonl_out:
    for batch_idx, batch_dois in enumerate(batches):
        if batch_idx < start_batch:
            continue

        url = build_url(batch_dois, args.api_key, args.email)

        try:
            data = fetch_json(url)
        except RuntimeError as exc:
            print(f"  [SKIP batch {batch_idx}] {exc}")
            unmatched.extend(batch_dois)
            continue

        results = data.get("results", [])
        found_dois = {r.get("doi", "").lower().replace("https://doi.org/", "")
                      for r in results}

        for work in results:
            jsonl_out.write(json.dumps(work, ensure_ascii=False) + "\n")
            csv_rows.append(flatten_work(work))

        # Track unmatched DOIs in this batch
        for d in batch_dois:
            clean_d = d.lower().replace("https://doi.org/", "")
            if clean_d not in found_dois:
                unmatched.append(d)

        total_fetched += len(results)

        # Progress report every 100 batches
        if batch_idx % 100 == 0 or batch_idx == total_batches - 1:
            pct = (batch_idx + 1) / total_batches * 100
            print(
                f"  Batch {batch_idx + 1:,}/{total_batches:,}  ({pct:.1f}%)  "
                f"fetched so far: {total_fetched:,}  unmatched: {len(unmatched):,}"
            )

        # Save progress checkpoint
        Path(PROGRESS_FILE).write_text(str(batch_idx))

        time.sleep(args.sleep)

# ---------------------------------------------------------------------------
# Save summary CSV and unmatched list
# ---------------------------------------------------------------------------
if csv_rows:
    df_out = pd.DataFrame(csv_rows)
    df_out.to_csv(CSV_FILE, index=False)
    print(f"\n[OK] Saved summary CSV: {CSV_FILE}  ({len(df_out):,} records)")

if unmatched:
    Path(UNMATCHED_FILE).write_text("\n".join(unmatched))
    print(f"[OK] Saved unmatched DOIs: {UNMATCHED_FILE}  ({len(unmatched):,} DOIs)")

print(f"\nDone. Total works fetched from OpenAlex: {total_fetched:,}")
print(f"      Unmatched DOIs (not in OpenAlex):   {len(unmatched):,}")
