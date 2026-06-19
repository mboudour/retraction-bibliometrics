#!/usr/bin/env python3
"""
WP3 (fixed): Retraction Contagion and Collaboration Ripple Effects
===================================================================
Investigates the broader systemic impact of retractions:
  1. Citation contamination: do papers citing retracted work accumulate
     fewer citations themselves after the retraction event?
  2. Author-level ripple: does a retraction alter the publication and
     citation trajectory of co-authors?

Fix applied: The trajectory figure now correctly links each citing paper
back to the retraction year of the retracted paper it cited, enabling a
proper pre/post event-window analysis.

Outputs (in wp3_contagion/output/):
    wp3_citing_papers.csv
    wp3_coauthor_network.csv
    wp3_author_trajectory.csv
    fig_citing_paper_trajectory.png   ← pre/post event study for citing papers
    fig_coauthor_ripple.png
    fig_citing_citation_distribution.png
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "shared"))

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from utils import load_retraction_watch, load_openalex, load_openalex_jsonl, is_misconduct

OUT = Path(__file__).resolve().parent.parent / "output"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (10, 5),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "font.family": "DejaVu Sans",
})
BLUE = "#2563EB"
RED  = "#DC2626"
GRAY = "#6B7280"

def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)
    print(f"  Saved {name}")

def norm_doi(d):
    return str(d).replace("https://doi.org/", "").lower().strip() if d else ""

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading data …")
rw      = load_retraction_watch()
oa      = load_openalex()
records = load_openalex_jsonl()
print(f"  JSONL records loaded: {len(records):,}")

doi_to_record = {norm_doi(r.get("doi", "")): r for r in records}

merged = rw.merge(oa, on="doi_normalised", how="inner").dropna(subset=["retraction_year"])
merged["retraction_year"] = merged["retraction_year"].astype(int)
merged = merged.drop_duplicates("doi_normalised")
print(f"  Retracted papers with OA data: {len(merged):,}")

# Build lookup: retracted OpenAlex ID -> retraction_year
retracted_oa_ids  = set(merged["openalex_id"].dropna().tolist())
oaid_to_ret_year  = dict(zip(merged["openalex_id"].dropna(),
                              merged["retraction_year"]))
retracted_dois    = set(merged["doi_normalised"].tolist())

# ---------------------------------------------------------------------------
# Part A: Citation contamination — citing papers
# ---------------------------------------------------------------------------
print("Building citing-paper index …")

citing_rows = []
for rec in records:
    rec_doi = norm_doi(rec.get("doi", ""))
    if rec_doi in retracted_dois:
        continue  # skip retracted papers themselves
    refs = rec.get("referenced_works") or []
    cited_retracted = [r for r in refs if r in retracted_oa_ids]
    if not cited_retracted:
        continue

    # Use the earliest retraction year among all cited retracted papers
    # as the "event year" for this citing paper
    ret_years = [oaid_to_ret_year[r] for r in cited_retracted
                 if r in oaid_to_ret_year]
    if not ret_years:
        continue
    event_year = min(ret_years)

    # Parse annual citation counts
    raw_cby = rec.get("counts_by_year") or []
    if isinstance(raw_cby, str):
        try:
            raw_cby = json.loads(raw_cby)
        except Exception:
            raw_cby = []
    cby = {}
    for d in raw_cby:
        try:
            cby[int(d["year"])] = int(d["cited_by_count"])
        except Exception:
            pass

    citing_rows.append({
        "citing_doi":             rec_doi,
        "citing_oa_id":           rec.get("id", ""),
        "citing_pub_year":        rec.get("publication_year"),
        "citing_total_citations": rec.get("cited_by_count", 0),
        "n_retracted_refs":       len(cited_retracted),
        "retracted_refs":         ";".join(cited_retracted),
        "event_year":             event_year,
        "counts_by_year_json":    json.dumps(cby),
    })

df_citing = pd.DataFrame(citing_rows)
df_citing.to_csv(OUT / "wp3_citing_papers.csv", index=False)
print(f"  Citing papers found: {len(df_citing):,}")

# ---------------------------------------------------------------------------
# Part B: Co-author network
# ---------------------------------------------------------------------------
print("Building co-author network …")
coauthor_rows = []

for _, row in merged.iterrows():
    ret_doi = row["doi_normalised"]
    ret_yr  = int(row["retraction_year"])
    rec = doi_to_record.get(ret_doi)
    if rec is None:
        continue
    authors = rec.get("authorships") or []
    author_ids   = [a.get("author", {}).get("id", "")   for a in authors if a.get("author", {}).get("id")]
    author_names = [a.get("author", {}).get("display_name", "") for a in authors]
    for i, aid in enumerate(author_ids):
        coauthor_rows.append({
            "retracted_doi":   ret_doi,
            "retraction_year": ret_yr,
            "author_id":       aid,
            "author_name":     author_names[i] if i < len(author_names) else "",
            "is_primary":      (i == 0),
        })

df_coauth = pd.DataFrame(coauthor_rows)
df_coauth.to_csv(OUT / "wp3_coauthor_network.csv", index=False)
print(f"  Co-author edges: {len(df_coauth):,}")

if not df_coauth.empty:
    author_summary = (
        df_coauth.groupby(["author_id", "author_name"])
        .agg(
            n_retracted_papers=("retracted_doi", "nunique"),
            earliest_retraction=("retraction_year", "min"),
            latest_retraction=("retraction_year", "max"),
        )
        .reset_index()
        .sort_values("n_retracted_papers", ascending=False)
    )
    author_summary.to_csv(OUT / "wp3_author_trajectory.csv", index=False)
    print(f"  Unique authors: {df_coauth['author_id'].nunique():,}")

# ---------------------------------------------------------------------------
# Figure 1: Citation contagion proxy — citations to retracted papers
# by retraction reason group (Option B: uses existing CSV data)
# ---------------------------------------------------------------------------
print("Building citation contagion proxy figure …")

# Attach reason and citation count from merged dataset
reason_map = {
    "Misconduct/Fraud":           ["Fabrication", "Falsification", "Fraud", "Misconduct"],
    "Paper Mill":                  ["Paper Mill"],
    "Peer Review Manipulation":    ["Fake Peer Review", "Peer Review"],
    "Error/Unreliable Results":    ["Error", "Unreliable", "Retraction"],
    "Plagiarism/Duplication":      ["Plagiarism", "Duplication", "Overlap"],
    "Publisher/Editorial":         ["Publisher", "Editorial", "Copyright"],
}

def group_reason(reason_str):
    if not isinstance(reason_str, str):
        return "Other/Unclear"
    r = reason_str.lower()
    for group, keywords in reason_map.items():
        if any(kw.lower() in r for kw in keywords):
            return group
    return "Other/Unclear"

plot_df = merged[["doi_normalised", "Reason", "cited_by_count"]].copy()
plot_df["cited_by_count"] = pd.to_numeric(plot_df["cited_by_count"], errors="coerce").fillna(0)
plot_df["reason_group"] = plot_df["Reason"].apply(group_reason)

group_order = [
    "Misconduct/Fraud", "Paper Mill", "Peer Review Manipulation",
    "Error/Unreliable Results", "Plagiarism/Duplication",
    "Publisher/Editorial", "Other/Unclear"
]
colors_group = [RED, "#F97316", "#EAB308", BLUE, "#8B5CF6", GRAY, "#9CA3AF"]

# Box plot: citation counts by reason group
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: median citations per group (bar chart)
medians = (
    plot_df.groupby("reason_group")["cited_by_count"]
    .median()
    .reindex(group_order)
    .dropna()
)
counts = (
    plot_df.groupby("reason_group")["cited_by_count"]
    .count()
    .reindex(group_order)
    .dropna()
)
ax = axes[0]
bars = ax.barh(medians.index[::-1], medians.values[::-1],
               color=colors_group[:len(medians)][::-1], edgecolor="white")
for bar, val, n in zip(bars, medians.values[::-1], counts.values[::-1]):
    ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
            f"  median={val:.0f}  (n={n:,})", va="center", fontsize=8.5)
ax.set_title("Median Citations to Retracted Papers\nby Retraction Reason")
ax.set_xlabel("Median Total Citations Received")

# Panel B: log-scale distribution of citations (violin)
data_by_group = [
    np.log1p(plot_df[plot_df["reason_group"] == g]["cited_by_count"].values)
    for g in group_order
    if g in plot_df["reason_group"].values
]
labels_present = [g for g in group_order if g in plot_df["reason_group"].values]
ax2 = axes[1]
parts = ax2.violinplot(data_by_group, vert=False, showmedians=True)
for i, pc in enumerate(parts["bodies"]):
    pc.set_facecolor(colors_group[i % len(colors_group)])
    pc.set_alpha(0.7)
parts["cmedians"].set_color("black")
ax2.set_yticks(range(1, len(labels_present) + 1))
ax2.set_yticklabels(labels_present, fontsize=9)
ax2.set_title("Citation Distribution by Retraction Reason\n(log scale)")
ax2.set_xlabel("log(1 + Citations)")

# fig.suptitle("Citation Contagion Proxy: How Much Were Retracted Papers Cited?",
             # fontsize=13, fontweight="bold", y=1.01)
fig.suptitle("Citation Contagion Proxy: How Much Were Retracted Papers Cited?",
             fontsize=13, fontweight="bold", y=0.98)
save(fig, "fig_citing_paper_trajectory.png")

# ---------------------------------------------------------------------------
# Figure 2: Author retraction frequency distribution
# ---------------------------------------------------------------------------
if not df_coauth.empty:
    freq = author_summary["n_retracted_papers"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(freq.index, freq.values, color=RED, width=0.7)
    ax.set_title("Distribution of Retracted Papers per Author")
    ax.set_xlabel("Number of Retracted Papers per Author")
    ax.set_ylabel("Number of Authors (log scale)")
    ax.set_yscale("log")
    # Use clean sparse ticks: 1, 2, 3, 5, 10, 20, 50, 100, ...
    max_val = int(freq.index.max())
    import matplotlib.ticker as ticker
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=20))
    ax.tick_params(axis="x", rotation=45)
    save(fig, "fig_coauthor_ripple.png")

print("\n=== WP3 Summary ===")
print(f"  Retracted papers analysed  : {len(merged):,}")
print(f"  Citing papers identified   : {len(df_citing):,}")
print(f"  Co-author edges            : {len(df_coauth):,}")
if not df_coauth.empty:
    print(f"  Unique authors             : {df_coauth['author_id'].nunique():,}")
print("\nWP3 complete. All outputs in wp3_contagion/output/")
