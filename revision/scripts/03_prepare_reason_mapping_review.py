#!/usr/bin/env python3
"""Prepare an auditable review sheet for RWDB reason-code classification.

This script deliberately does NOT make Figure 4. It produces one row per raw
RWDB reason code. Review and approve the suggested category before running
04_make_primary_reason_figure.py.
"""
from __future__ import annotations

import re

import pandas as pd

from common import find_column, project_root, read_master, revision_dirs, split_reason_field


def suggest_category(reason: str) -> str:
    """Conservative first-pass suggestion; every row must be reviewed by a human."""
    r = reason.lower()
    tests = [
        ("Paper Mill", r"paper mill"),
        ("Misconduct/Fraud", r"falsification|fabrication|misconduct|fraud|forgery|forged|manipulation of images|image manipulation"),
        ("Plagiarism/Duplication", r"plagiari|duplication|duplicate publication|duplicate article|overlap"),
        ("Error/Unreliable Results", r"unreliable|not reproducible|error in|erroneous|data.*not (provided|available)|original data.*not|concerns/issues about (data|results|conclusions)|error.*(data|methods|analyses|results|image)"),
        ("Publication Process/Editorial", r"investigation|peer review|referenc|attribution|journal|publisher|editor|notice|author unresponsive|authorship|affiliation|date of article|withdrawn|withdrawal|removed|rogue editor|breach of policy|irb|iacuc|consent"),
    ]
    for category, pattern in tests:
        if re.search(pattern, r, flags=re.I):
            return category
    return "Other/Unclear"


def main() -> None:
    root = project_root()
    dirs = revision_dirs(root)
    df = read_master(root)
    reason_col = find_column(df.columns, ["Reason", "reason", "RetractionReason"])
    if not reason_col:
        raise KeyError("No reason field found. Expected Reason, reason, or RetractionReason.")

    exploded = []
    for value in df[reason_col]:
        reasons = split_reason_field(value)
        if not reasons:
            reasons = ["[No stated reason]"]
        exploded.extend(reasons)

    review = (
        pd.Series(exploded, name="raw_reason")
        .value_counts()
        .rename_axis("raw_reason")
        .reset_index(name="n_papers")
    )
    review["suggested_primary_category"] = review["raw_reason"].map(suggest_category)
    review["approved_primary_category"] = review["suggested_primary_category"]
    review["review_status"] = "REVIEW"
    review["notes"] = "Verify or amend suggested category; then set review_status to APPROVED."

    output = dirs["output"] / "reason_mapping_review.csv"
    # Preserve prior human approvals when the underlying list of raw codes is unchanged.
    if output.exists():
        prior = pd.read_csv(output).fillna("")
        required = {"raw_reason", "approved_primary_category", "review_status", "notes"}
        if required.issubset(prior.columns):
            prior = prior[list(required)].drop_duplicates("raw_reason")
            review = review.drop(columns=["approved_primary_category", "review_status", "notes"]).merge(
                prior, on="raw_reason", how="left"
            )
            review["approved_primary_category"] = review["approved_primary_category"].replace("", pd.NA).fillna(review["suggested_primary_category"])
            review["review_status"] = review["review_status"].replace("", pd.NA).fillna("REVIEW")
            review["notes"] = review["notes"].replace("", pd.NA).fillna("Verify or amend suggested category; then set review_status to APPROVED.")
    review = review.sort_values(["n_papers", "raw_reason"], ascending=[False, True])
    review.to_csv(output, index=False)
    (
        review.groupby("suggested_primary_category", as_index=False)["n_papers"]
        .sum()
        .sort_values("n_papers", ascending=False)
        .to_csv(dirs["output"] / "reason_mapping_suggestion_summary.csv", index=False)
    )
    print(f"Wrote {output}")
    print("Review each row, amend approved_primary_category if needed, and set review_status=APPROVED.")


if __name__ == "__main__":
    main()
