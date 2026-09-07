#!/usr/bin/env python3
"""Step 1.2–1.3: freeze merged data and create a transparent quality audit.

Run from the project root after copying this package into revision/:
    python revision/scripts/01_freeze_and_audit.py
"""
from __future__ import annotations

import platform
import sys

import pandas as pd

from common import (
    find_column,
    normalise_doi,
    now_utc,
    parse_dates,
    project_root,
    revision_dirs,
    sha256,
    source_paths,
    write_json,
)


def main() -> None:
    root = project_root()
    dirs = revision_dirs(root)
    source = source_paths(root)["merged"]
    if not source.exists():
        raise FileNotFoundError(f"Expected merged data at: {source}")

    df = pd.read_csv(source, low_memory=False)
    original_n = len(df)
    doi_col = find_column(df.columns, ["doi_normalised", "doi", "OriginalPaperDOI"])
    id_col = find_column(df.columns, ["openalex_id", "OpenAlexID"])
    pub_col = find_column(df.columns, ["OriginalPaperDate", "publication_date", "pub_date"])
    ret_col = find_column(df.columns, ["RetractionDate", "retraction_date"])

    df["revision_doi"] = normalise_doi(df[doi_col]) if doi_col else pd.NA
    df["revision_openalex_id"] = (
        df[id_col].astype("string").str.strip().replace("", pd.NA) if id_col else pd.NA
    )
    pub_dt = parse_dates(df[pub_col]) if pub_col else pd.Series(pd.NaT, index=df.index)
    ret_dt = parse_dates(df[ret_col]) if ret_col else pd.Series(pd.NaT, index=df.index)
    df["revision_publication_date"] = pub_dt.dt.strftime("%Y-%m-%d")
    df["revision_retraction_date"] = ret_dt.dt.strftime("%Y-%m-%d")

    flags = pd.DataFrame(
        {
            "missing_doi": df["revision_doi"].isna(),
            "missing_openalex_id": df["revision_openalex_id"].isna(),
            "missing_publication_date": pub_dt.isna(),
            "missing_retraction_date": ret_dt.isna(),
            "retraction_before_publication": (ret_dt < pub_dt).fillna(False),
            "duplicate_doi": df["revision_doi"].notna() & df["revision_doi"].duplicated(keep=False),
            "duplicate_openalex_id": (
                df["revision_openalex_id"].notna()
                & df["revision_openalex_id"].duplicated(keep=False)
            ),
        }
    )
    flags["any_flag"] = flags.any(axis=1)

    audit = pd.DataFrame(
        {
            "check": flags.columns,
            "n_records": [int(flags[c].sum()) for c in flags.columns],
            "share_of_rows": [float(flags[c].mean()) for c in flags.columns],
        }
    )
    audit.to_csv(dirs["output"] / "data_quality_audit.csv", index=False)

    keep_cols = [
        c
        for c in [
            id_col,
            doi_col,
            pub_col,
            ret_col,
            "revision_openalex_id",
            "revision_doi",
            "revision_publication_date",
            "revision_retraction_date",
        ]
        if c
    ]
    pd.concat([df[keep_cols], flags], axis=1).loc[flags["any_flag"]].to_csv(
        dirs["output"] / "invalid_or_flagged_records.csv", index=False
    )

    master = dirs["data"] / "revision_master.csv.gz"
    df.to_csv(master, index=False, compression="gzip")
    write_json(
        dirs["output"] / "revision_manifest.json",
        {
            "created_utc": now_utc(),
            "source_file": str(source),
            "source_sha256": sha256(source),
            "frozen_file": str(master),
            "frozen_sha256": sha256(master),
            "rows": int(original_n),
            "columns": list(df.columns),
            "identifier_columns": {"doi": doi_col, "openalex_id": id_col},
            "date_columns": {"publication": pub_col, "retraction": ret_col},
            "python": sys.version,
            "platform": platform.platform(),
            "rule": "No source rows were dropped; invalid or ambiguous records were retained and flagged separately.",
        },
    )
    print(f"Frozen {original_n:,} rows -> {master}")
    print(f"Audit written -> {dirs['output']}")


if __name__ == "__main__":
    main()
