#!/usr/bin/env python3
"""Generate a mutually exclusive, audited primary-reason Figure 4.

Prerequisite: reason_mapping_review.csv must have every review_status set to
APPROVED and an approved_primary_category selected from the priority file.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from common import find_column, project_root, read_master, revision_dirs, save_dual_figure, split_reason_field


def main() -> None:
    root = project_root()
    dirs = revision_dirs(root)
    out = dirs["output"]
    review_path = out / "reason_mapping_review.csv"
    priority_path = dirs["config"] / "reason_category_priority.csv"
    if not review_path.exists():
        raise FileNotFoundError("Run 03_prepare_reason_mapping_review.py first.")
    if not priority_path.exists():
        raise FileNotFoundError(f"Missing priority configuration: {priority_path}")

    review = pd.read_csv(review_path).fillna("")
    priority = pd.read_csv(priority_path).sort_values("priority")
    valid = set(priority["primary_category"])
    unresolved = review[~review["review_status"].astype(str).str.upper().eq("APPROVED")]
    invalid = review[~review["approved_primary_category"].isin(valid)]
    if not unresolved.empty or not invalid.empty:
        raise ValueError(
            "Reason mapping is not approved. Set review_status=APPROVED for every row and "
            "use a valid approved_primary_category. "
            f"Unapproved rows={len(unresolved):,}; invalid rows={len(invalid):,}."
        )

    mapper = dict(zip(review["raw_reason"], review["approved_primary_category"]))
    rank = dict(zip(priority["primary_category"], priority["priority"]))
    fallback = "Other/Unclear"
    df = read_master(root)
    reason_col = find_column(df.columns, ["Reason", "reason", "RetractionReason"])
    if not reason_col:
        raise KeyError("No reason field found.")

    record_rows = []
    for index, value in df[reason_col].items():
        raw = split_reason_field(value) or ["[No stated reason]"]
        missing = [r for r in raw if r not in mapper]
        if missing:
            raise ValueError(f"Raw reason(s) missing from approved mapping, e.g. {missing[:3]}")
        categories = sorted({mapper[r] for r in raw}, key=lambda x: rank[x])
        primary = categories[0] if categories else fallback
        record_rows.append(
            {
                "record_index": index,
                "primary_reason_category": primary,
                "all_reason_categories": "; ".join(categories),
                "n_reason_categories": len(categories),
            }
        )
    records = pd.DataFrame(record_rows)
    records.to_csv(out / "reason_primary_record_assignments.csv.gz", index=False, compression="gzip")

    counts = records["primary_reason_category"].value_counts().reindex(priority["primary_category"], fill_value=0)
    primary = counts.rename_axis("primary_reason_category").reset_index(name="n_papers")
    primary["percent_of_all_retracted_papers"] = 100 * primary["n_papers"] / len(records)
    primary["denominator"] = len(records)
    primary.to_csv(out / "reason_primary_category_counts.csv", index=False)

    multi = records.assign(_one=1).copy()
    expanded = multi.assign(all_reason_categories=multi["all_reason_categories"].str.split("; ")).explode("all_reason_categories")
    prevalence = expanded.groupby("all_reason_categories")["record_index"].nunique().reindex(priority["primary_category"], fill_value=0)
    prevalence = prevalence.rename_axis("broad_category").reset_index(name="n_papers")
    prevalence["percent_of_all_retracted_papers"] = 100 * prevalence["n_papers"] / len(records)
    prevalence["denominator"] = len(records)
    prevalence.to_csv(out / "reason_multilabel_prevalence_supplement.csv", index=False)

    plot = primary.sort_values("percent_of_all_retracted_papers", ascending=True)
    colors = {
        "Paper Mill": "#A23B72",
        "Misconduct/Fraud": "#E76F51",
        "Plagiarism/Duplication": "#F4A261",
        "Error/Unreliable Results": "#2A9D8F",
        "Publication Process/Editorial": "#457B9D",
        "Other/Unclear": "#6C757D",
    }
    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    bars = ax.barh(plot["primary_reason_category"], plot["percent_of_all_retracted_papers"], color=[colors[x] for x in plot["primary_reason_category"]], edgecolor="white")
    max_value = max(plot["percent_of_all_retracted_papers"].max(), 1)
    for bar, n, pct in zip(bars, plot["n_papers"], plot["percent_of_all_retracted_papers"]):
        ax.text(bar.get_width() + max_value * 0.012, bar.get_y() + bar.get_height()/2, f"{pct:.1f}%  (n={n:,})", va="center", fontsize=9)
    ax.set_xlim(0, max_value * 1.25)
    ax.set_xlabel("Share of retracted papers (%)")
    ax.set_ylabel("")
    ax.set_title("Primary retraction-reason category in the Retraction Watch–OpenAlex corpus")
    ax.grid(axis="x", linestyle=":", alpha=0.45)
    ax.set_axisbelow(True)
    fig.text(0.5, 0.01, "Each paper is assigned one primary category using the documented priority rule; full multi-label prevalence is reported in the supplement.", ha="center", fontsize=8.3)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save_dual_figure(fig, out / "fig_retraction_reasons_primary")
    plt.close(fig)

    print(primary.sort_values("n_papers", ascending=False).to_string(index=False))
    print(f"Wrote primary-category Figure 4 to {out}")


if __name__ == "__main__":
    main()
