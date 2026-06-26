#!/usr/bin/env python3
"""
WP1 (v2): Descriptive Mapping — Fixed & Enhanced
=================================================
Fixes:
  1. Adds retraction-notice-year figure (separate from publication year)
  2. Fixes pct_still_cited_after_retraction computation
  3. Adds grouped retraction-reason taxonomy figure
  4. Adds open-access trend over time figure

Outputs (in wp1_descriptive/output/):
  merged_dataset.csv
  wp1_summary_stats.csv
  fig_retractions_per_year.png          (by publication year — unchanged)
  fig_retractions_by_notice_year.png    (NEW: by retraction-notice year)
  fig_top_disciplines.png
  fig_top_countries.png
  fig_top_journals.png
  fig_retraction_reasons.png            (granular, unchanged)
  fig_retraction_reasons_grouped.png    (NEW: grouped taxonomy)
  fig_citation_distribution.png
  fig_time_to_retraction.png
  fig_oa_trend.png                      (NEW: open-access rate over time)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "shared"))

import json
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from utils import load_retraction_watch, load_openalex, RAW_DATA

OUT = Path(__file__).resolve().parent.parent / "output"
OUT.mkdir(parents=True, exist_ok=True)

STYLE = {
    "figure.figsize": (15, 7.5),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "font.family": "DejaVu Sans",
}
plt.rcParams.update(STYLE)
BLUE = "#2563EB"
RED  = "#DC2626"
GREEN = "#16A34A"

# ---------------------------------------------------------------------------
# Load & merge
# ---------------------------------------------------------------------------
print("Loading data …")
rw = load_retraction_watch()
oa = load_openalex()

# Parse retraction date → year
rw["retraction_year"] = pd.to_datetime(
    rw["RetractionDate"], errors="coerce", dayfirst=False
).dt.year

# Parse original paper date → year (fallback to pub_year)
rw["orig_year"] = pd.to_datetime(
    rw["OriginalPaperDate"], errors="coerce", dayfirst=False
).dt.year.fillna(rw.get("pub_year", np.nan))

merged = oa.merge(rw, on="doi_normalised", how="inner")
merged.to_csv(OUT / "merged_dataset.csv", index=False)
merged.to_csv(RAW_DATA / "merged_dataset.csv", index=False)
print(f"  Merged dataset: {len(merged):,} rows")

# ---------------------------------------------------------------------------
# Grouped retraction-reason taxonomy
# ---------------------------------------------------------------------------
REASON_GROUPS = {
    "Misconduct/Fraud":        ["fabricat", "falsif", "fraud", "plagiar", "manipulat",
                                 "self-plagiar", "duplicate", "duplication of/in image",
                                 "duplication of/in article"],
    "Paper Mill":              ["paper mill", "computer-aided content", "computer-generated"],
    "Peer Review Manipulation":["compromised peer review", "concerns/issues about peer review",
                                "fake peer review"],
    "Data/Results Issues":     ["concerns/issues about data", "unreliable results",
                                "concerns/issues about results", "error in data",
                                "error in results"],
    "Publisher/Editorial":     ["investigation by journal", "investigation by third party",
                                "investigation by company", "breach of policy",
                                "author unresponsive", "objections by author",
                                "removed", "date of article"],
    "Attribution/Referencing": ["concerns/issues about referencing", "concerns/issues about image",
                                "concerns/issues about authorship"],
    "Unclear/Other":           ["notice - limited or no information", "unknown"],
}

def assign_group(reason_str):
    if not isinstance(reason_str, str):
        return "Unclear/Other"
    r = reason_str.lower()
    for group, keywords in REASON_GROUPS.items():
        if any(kw in r for kw in keywords):
            return group
    return "Unclear/Other"

merged["reason_group"] = merged["Reason"].apply(assign_group)

# ---------------------------------------------------------------------------
# pct_still_cited_after_retraction
# ---------------------------------------------------------------------------
def parse_cby(json_str):
    try:
        return {d["year"]: d["cited_by_count"] for d in json.loads(json_str)}
    except Exception:
        return {}

def still_cited_after(row):
    cby = parse_cby(row.get("counts_by_year_json", "[]"))
    ret_year = row.get("retraction_year")
    if not cby or pd.isna(ret_year):
        return np.nan
    post = {y: c for y, c in cby.items() if y > ret_year}
    return int(sum(post.values()) > 0)

print("  Computing pct_still_cited_after_retraction …")
merged["still_cited_post"] = merged.apply(still_cited_after, axis=1)
pct_still_cited = merged["still_cited_post"].mean() * 100

# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
summary = {
    "total_retractions": len(merged),
    "matched_to_openalex": len(merged),
    "match_rate_pct": round(len(merged) / len(rw) * 100, 1),
    "pub_year_min": merged["publication_year"].min(),
    "pub_year_max": merged["publication_year"].max(),
    "retraction_year_min": merged["retraction_year"].min(),
    "retraction_year_max": merged["retraction_year"].max(),
    "median_time_to_retraction_years": (merged["retraction_year"] - merged["publication_year"]).median(),
    "mean_citations": round(merged["cited_by_count"].mean(), 1),
    "median_citations": merged["cited_by_count"].median(),
    "pct_still_cited_after_retraction": round(pct_still_cited, 1),
    "pct_misconduct": round((merged["reason_group"] == "Misconduct/Fraud").mean() * 100, 1),
    "pct_paper_mill": round((merged["reason_group"] == "Paper Mill").mean() * 100, 1),
    "unique_journals": merged["journal_name"].nunique(),
    "unique_countries": merged["Country"].nunique(),
}
pd.DataFrame(list(summary.items()), columns=["Metric", "Value"]).to_csv(
    OUT / "wp1_summary_stats.csv", index=False)
print("  Saved wp1_summary_stats.csv")
for k, v in summary.items():
    print(f"    {k}: {v}")

# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------
def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)
    print(f"  Saved {name}")

# ---------------------------------------------------------------------------
# Fig 1: Retractions by publication year
# ---------------------------------------------------------------------------
yr = merged["publication_year"].dropna().astype(int)
yr = yr[(yr >= 1985) & (yr <= 2025)]
fig, ax = plt.subplots()
ax.bar(yr.value_counts().sort_index().index,
       yr.value_counts().sort_index().values, color=BLUE, width=0.8)
ax.set_title("Retracted Papers by Publication Year")
ax.set_xlabel("Publication Year"); ax.set_ylabel("Number of Papers")
save(fig, "fig_retractions_per_year.png")

# ---------------------------------------------------------------------------
# Fig 2 (NEW): Retractions by retraction-notice year
# ---------------------------------------------------------------------------
ryr = merged["retraction_year"].dropna().astype(int)
ryr = ryr[(ryr >= 1990) & (ryr <= 2025)]
fig, ax = plt.subplots()
ax.bar(ryr.value_counts().sort_index().index,
       ryr.value_counts().sort_index().values, color=RED, width=0.8)
ax.set_title("Retractions by Retraction-Notice Year")
ax.set_xlabel("Year of Retraction Notice"); ax.set_ylabel("Number of Retractions")
save(fig, "fig_retractions_by_notice_year.png")

# ---------------------------------------------------------------------------
# Fig 3: Top 20 disciplines
# ---------------------------------------------------------------------------
disc = merged["top_concept"].dropna().value_counts().head(20)
fig, ax = plt.subplots(figsize=(15, 8))
ax.barh(disc.index[::-1], disc.values[::-1], color=BLUE)
ax.set_title("Top 20 Disciplines (OpenAlex Top Concept)")
ax.set_xlabel("Number of Retracted Papers")
save(fig, "fig_top_disciplines.png")

# ---------------------------------------------------------------------------
# Fig 4: Top 20 countries
# ---------------------------------------------------------------------------
countries = merged["Country"].dropna()
countries = countries.str.split("+").explode().str.strip()
ctry = countries.value_counts().head(20)
fig, ax = plt.subplots()
ax.barh(ctry.index[::-1], ctry.values[::-1], color=GREEN)
ax.set_title("Top 20 Countries of Retracted Papers")
ax.set_xlabel("Number of Retracted Papers")
save(fig, "fig_top_countries.png")

# ---------------------------------------------------------------------------
# Fig 5: Top 20 journals
# ---------------------------------------------------------------------------
jrnl = merged["journal_name"].dropna().value_counts().head(20)
fig, ax = plt.subplots(figsize=(15, 8))
ax.barh(jrnl.index[::-1], jrnl.values[::-1], color=RED)
ax.set_title("Top 20 Journals by Number of Retractions")
ax.set_xlabel("Number of Retracted Papers")
save(fig, "fig_top_journals.png")

# ---------------------------------------------------------------------------
# Fig 6: Top 20 granular retraction reasons
# ---------------------------------------------------------------------------
reasons = merged["Reason"].dropna()
reasons = reasons.str.split("+").explode().str.strip()
top_reasons = reasons.value_counts().head(20)
fig, ax = plt.subplots(figsize=(15, 10.5))
ax.barh(top_reasons.index[::-1], top_reasons.values[::-1], color=RED)
ax.set_title("Top 20 Retraction Reasons (Granular)")
ax.set_xlabel("Number of Retractions")
save(fig, "fig_retraction_reasons.png")

# ---------------------------------------------------------------------------
# Fig 7 (NEW): Grouped retraction-reason taxonomy
# ---------------------------------------------------------------------------
grp = merged["reason_group"].value_counts()
colors_grp = [BLUE, RED, GREEN, "#7C3AED", "#D97706", "#0891B2", "#6B7280"]
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(grp.index[::-1], grp.values[::-1],
               color=colors_grp[:len(grp)], edgecolor="white")
for bar, val in zip(bars, grp.values[::-1]):
    ax.text(bar.get_width() + 200, bar.get_y() + bar.get_height()/2,
            f"{val:,}  ({val/len(merged)*100:.1f}%)",
            va="center", fontsize=10)
ax.set_title("Retraction Reasons — Grouped Taxonomy")
ax.set_xlabel("Number of Retractions")
ax.set_xlim(0, grp.max() * 1.25)
save(fig, "fig_retraction_reasons_grouped.png")

# ---------------------------------------------------------------------------
# Fig 8: Citation count distribution (log scale)
# ---------------------------------------------------------------------------
cites = merged["cited_by_count"].dropna()
cites = cites[cites > 0]
fig, ax = plt.subplots()
ax.hist(cites, bins=50, color=BLUE, edgecolor="white", linewidth=0.3)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_title("Citation Count Distribution of Retracted Papers")
ax.set_xlabel("Total Citations (log scale)"); ax.set_ylabel("Number of Papers (log scale)")
save(fig, "fig_citation_distribution.png")

# ---------------------------------------------------------------------------
# Fig 9: Time from publication to retraction
# ---------------------------------------------------------------------------
ttp = (merged["retraction_year"] - merged["publication_year"]).dropna()
ttp = ttp[(ttp >= 0) & (ttp <= 30)]
fig, ax = plt.subplots()
ax.hist(ttp, bins=31, color=GREEN, edgecolor="white", linewidth=0.3, range=(0, 30))
ax.axvline(ttp.median(), color=RED, linestyle="--", linewidth=1.5,
           label=f"Median = {ttp.median():.1f} yrs")
ax.set_title("Time from Publication to Retraction")
ax.set_xlabel("Years"); ax.set_ylabel("Number of Papers")
ax.legend()
save(fig, "fig_time_to_retraction.png")

# ---------------------------------------------------------------------------
# Fig 10 (NEW): Open-access rate of retracted papers over time
# ---------------------------------------------------------------------------
oa_trend = merged.groupby("retraction_year")["is_oa"].mean().reset_index()
oa_trend = oa_trend[(oa_trend["retraction_year"] >= 2000) &
                    (oa_trend["retraction_year"] <= 2024)]
fig, ax = plt.subplots()
ax.plot(oa_trend["retraction_year"], oa_trend["is_oa"] * 100,
        color=BLUE, linewidth=2, marker="o", markersize=4)
ax.set_title("Open-Access Rate of Retracted Papers by Retraction Year")
ax.set_xlabel("Year of Retraction Notice")
ax.set_ylabel("% Open Access")
ax.set_ylim(0, 100)
save(fig, "fig_oa_trend.png")

print("\n=== WP1 Summary ===")
print(f"  Total retracted papers in merged dataset : {len(merged):,}")
print(f"  Still cited after retraction             : {pct_still_cited:.1f}%")
print(f"  Misconduct/Fraud share                   : {summary['pct_misconduct']}%")
print(f"  Paper Mill share                         : {summary['pct_paper_mill']}%")
print("\nWP1 complete. All outputs in wp1_descriptive/output/")
