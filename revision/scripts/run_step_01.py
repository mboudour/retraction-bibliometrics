#!/usr/bin/env python3
"""Run Step 1 in two passes.

Pass A freezes/audits data and creates reason_mapping_review.csv.
Pass B runs automatically only after every mapping row is marked APPROVED.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


def run(script: Path) -> None:
    print(f"\n{'=' * 78}\nRUNNING: {script.name}\n{'=' * 78}")
    subprocess.run([sys.executable, str(script)], check=True)


def main() -> None:
    scripts = Path(__file__).resolve().parent
    revision_dir = scripts.parent
    output = revision_dir / "output"
    for name in ["01_freeze_and_audit.py", "02_build_sample_flow.py", "03_prepare_reason_mapping_review.py"]:
        run(scripts / name)

    review = pd.read_csv(output / "reason_mapping_review.csv")
    approved = review["review_status"].astype(str).str.upper().eq("APPROVED").all()
    if not approved:
        print("\nPASS A COMPLETE.")
        print(f"Review and approve {len(review):,} rows in: {output / 'reason_mapping_review.csv'}")
        print("Set review_status to APPROVED for every row, then rerun this same command for PASS B.")
        return

    for name in ["04_make_primary_reason_figure.py", "05_consistency_audit.py"]:
        run(scripts / name)
    print("\nSTEP 1 COMPLETE. Review revision/output/ before editing the manuscript.")


if __name__ == "__main__":
    main()
