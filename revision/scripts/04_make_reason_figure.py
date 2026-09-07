#!/usr/bin/env python3
"""Step 1.6: generate the revised Figure 4 from audited multi-label prevalence."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from common import project_root, revision_dirs, save_dual_figure


def main() -> None:
    root = project_root()
    dirs = revision_dirs(root)
    source = dirs["output"] / "reason_category_multilabel_prevalence.csv"
    if not source.exists():
        raise FileNotFoundError("Run 03_audit_retraction_reasons.py before creating Figure 4.")
    data = pd.read_csv(source).sort_values("percent_of_all_retracted_papers", ascending=True)

    colors = {
        "Paper Mill": "#A23B72",
        "Misconduct/Fraud": "#E76F51",
        "Plagiarism/Duplication": "#F4A261",
        "Error/Unreliable Results": "#2A9D8F",
        "Publisher/Editorial": "#457B9D",
        "Other/Unclear": "#6C757D",
    }
    fig, ax = plt.subplots(figsize=(9.3, 5.4))
    bars = ax.barh(
        data["broad_category"],
        data["percent_of_all_retracted_papers"],
        color=[colors.get(x, "#6C757D") for x in data["broad_category"]],
        edgecolor="white",
        linewidth=0.8,
    )
    for bar, n, pct in zip(bars, data["n_papers"], data["percent_of_all_retracted_papers"]):
        ax.text(bar.get_width() + max(data["percent_of_all_retracted_papers"]) * 0.012,
                bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%  (n={n:,})", va="center", fontsize=9)
    ax.set_xlabel("Share of all retracted papers (%)")
    ax.set_ylabel("")
    ax.set_title("Retraction reasons in the Retraction Watch–OpenAlex corpus")
    ax.grid(axis="x", linestyle=":", alpha=0.45)
    ax.set_axisbelow(True)
    max_x = max(data["percent_of_all_retracted_papers"].max() * 1.25, 5)
    ax.set_xlim(0, max_x)
    fig.text(0.5, 0.01,
             "Multi-label coding: a paper may belong to more than one category; percentages do not sum to 100%.",
             ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))

    stem = dirs["output"] / "fig_retraction_reasons_multilabel"
    save_dual_figure(fig, stem)
    plt.close(fig)
    print(f"Wrote {stem.with_suffix('.pdf')} and {stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
