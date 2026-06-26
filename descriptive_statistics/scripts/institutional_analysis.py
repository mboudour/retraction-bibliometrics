#!/usr/bin/env python3
"""
WP1 Extension — Institutional Analysis of Retracted Papers
===========================================================
Produces statistics and figures on the institutions and countries
associated with retracted papers, using the merged Retraction Watch +
OpenAlex dataset produced by wp1_descriptive_mapping.py.

Outputs (written to wp1_descriptive/output/):
  - wp1_country_stats.csv          : retractions per country (unique + multi-country)
  - wp1_institution_stats.csv      : top institutions by retraction count
  - fig_top_countries.png          : bar chart of top 25 countries
  - fig_country_trend.png          : retraction trends for top 8 countries over time
  - fig_top_institutions.png       : bar chart of top 30 institutions
  - fig_country_reason_heatmap.png : heatmap of retraction reason by country (top 15)
  - fig_country_citations.png      : median citations by country (top 20)
"""

import os, sys, re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────────────────
HERE   = os.path.dirname(os.path.abspath(__file__))

def _find_merged():
    candidates = [
        os.path.join(HERE, "..", "wp1_descriptive", "output", "merged_dataset.csv"),
        os.path.join(HERE, "wp1_descriptive", "output", "merged_dataset.csv"),
        "/home/ubuntu/final_review/wp1_descriptive/output/merged_dataset.csv",
        "/home/ubuntu/final_review/descriptive_statistics/output/merged_dataset.csv",
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    raise FileNotFoundError("Cannot find merged_dataset.csv. Run wp1_descriptive_mapping.py first.")

MERGED = _find_merged()
OUT    = os.path.dirname(MERGED)
os.makedirs(OUT, exist_ok=True)

PALETTE = sns.color_palette("tab20", 20)
sns.set_theme(style="whitegrid", font_scale=1.05)

# ── Load data ───────────────────────────────────────────────────────────────
print("Loading merged dataset …")
df = pd.read_csv(MERGED, low_memory=False)
print(f"  {len(df):,} records loaded")

# Normalise retraction year
df["retraction_year"] = pd.to_numeric(df.get("retraction_year", df.get("pub_year")), errors="coerce")
df["publication_year"] = pd.to_numeric(df.get("publication_year", df.get("pub_year")), errors="coerce")
df["cited_by_count"]   = pd.to_numeric(df.get("cited_by_count", 0), errors="coerce").fillna(0)

# ── Grouped reason taxonomy (reuse from WP1) ───────────────────────────────
REASON_MAP = {
    "Falsification/Fabrication of Data": "Misconduct/Fraud",
    "Manipulation of Images":            "Misconduct/Fraud",
    "Fraud":                             "Misconduct/Fraud",
    "Misconduct":                        "Misconduct/Fraud",
    "Paper Mill":                        "Paper Mill",
    "Fake Peer Review":                  "Peer Review Manipulation",
    "Manipulation of Results":           "Misconduct/Fraud",
    "Plagiarism of Data":                "Plagiarism/Duplication",
    "Plagiarism of Text":                "Plagiarism/Duplication",
    "Duplication of Article":            "Plagiarism/Duplication",
    "Duplication of Image":              "Plagiarism/Duplication",
    "Results Not Reproducible":          "Error/Unreliable Results",
    "Error in Data":                     "Error/Unreliable Results",
    "Error in Methods":                  "Error/Unreliable Results",
    "Error in Analyses":                 "Error/Unreliable Results",
    "Unreliable Results":                "Error/Unreliable Results",
    "Concerns/Issues with Data":         "Error/Unreliable Results",
    "Publisher/Editor Concerns":         "Publisher/Editorial",
    "Editorial Concerns":                "Publisher/Editorial",
    "Withdrawn":                         "Publisher/Editorial",
}

def map_reason(raw):
    if pd.isna(raw):
        return "Other/Unclear"
    for key, grp in REASON_MAP.items():
        if key.lower() in str(raw).lower():
            return grp
    return "Other/Unclear"

df["reason_group"] = df["Reason"].apply(map_reason)

# ── 1. Country-level analysis ───────────────────────────────────────────────
print("Computing country statistics …")

# Explode multi-country papers
country_rows = []
for _, row in df.iterrows():
    raw = str(row.get("Country", "")) if pd.notna(row.get("Country")) else ""
    countries = [c.strip() for c in raw.split(";") if c.strip()]
    if not countries:
        countries = ["Unknown"]
    for c in countries:
        country_rows.append({
            "country":          c,
            "retraction_year":  row["retraction_year"],
            "publication_year": row["publication_year"],
            "cited_by_count":   row["cited_by_count"],
            "reason_group":     row["reason_group"],
            "openalex_id":      row.get("openalex_id", ""),
        })

cdf = pd.DataFrame(country_rows)

# Aggregate
country_stats = (
    cdf.groupby("country")
       .agg(
           retraction_count  = ("openalex_id", "count"),
           median_citations  = ("cited_by_count", "median"),
           mean_citations    = ("cited_by_count", "mean"),
       )
       .reset_index()
       .sort_values("retraction_count", ascending=False)
)
country_stats.to_csv(os.path.join(OUT, "wp1_country_stats.csv"), index=False)
print(f"  Country stats saved: {len(country_stats)} countries")

# ── 2. Institution-level analysis ───────────────────────────────────────────
print("Computing institution statistics …")

# Keywords that identify a segment part as a university-level institution
# (used to extract the parent institution name for grouping)
UNI_KEYWORDS = [
    "university", "universit\u00e9", "universitat", "universidade", "universidad",
    "institute of technology", "institute of science", "institute of medicine",
    "institute of engineering", "institute of management",
    "medical college", "engineering college", "dental college",
    "school of medicine", "school of public health",
    "academy", "klinikum",
]
# Broader fallback keywords (used only when no UNI_KEYWORD match is found)
INST_FALLBACK = [
    "institute", "hospital", "college", "school",
    "centre", "center", "clinic", "foundation",
]

def extract_institution_name(seg):
    """
    Extract the university/institution name from a comma-delimited affiliation segment.
    Priority: university-level keywords first, then broader institution keywords.
    Returns None if no institution-level part is found (segment will be skipped).
    """
    parts = [p.strip() for p in seg.split(",")]
    # First pass: look for university-level keywords
    for part in parts:
        pl = part.lower()
        if any(kw in pl for kw in UNI_KEYWORDS):
            return part
    # Second pass: look for broader institution keywords
    for part in parts:
        pl = part.lower()
        if any(kw in pl for kw in INST_FALLBACK):
            return part
    # Fallback: return None to skip this segment
    return None

inst_rows = []
for _, row in df.iterrows():
    raw = str(row.get("Institution", "")) if pd.notna(row.get("Institution")) else ""
    segments = [s.strip() for s in raw.split(";") if s.strip()]
    if not segments:
        continue
    seen_in_row = set()  # avoid double-counting same university within one paper
    for seg in segments:
        if seg.lower() in ("unavailable", "unknown", ""):
            continue
        name = extract_institution_name(seg)
        if name and name not in seen_in_row:
            seen_in_row.add(name)
            inst_rows.append({
                "institution":     name,
                "country":         str(row.get("Country", "")).split(";")[0].strip(),
                "retraction_year": row["retraction_year"],
                "cited_by_count":  row["cited_by_count"],
                "reason_group":    row["reason_group"],
                "openalex_id":     row.get("openalex_id", ""),
            })

idf = pd.DataFrame(inst_rows)

inst_stats = (
    idf.groupby("institution")
       .agg(
           retraction_count = ("openalex_id", "count"),
           median_citations = ("cited_by_count", "median"),
           primary_country  = ("country", lambda x: x.mode().iloc[0] if len(x) > 0 else ""),
       )
       .reset_index()
       .sort_values("retraction_count", ascending=False)
)
# Filter out very generic / noisy segments (standalone labels without a proper name)
noise = {
    "Department", "School", "College", "Faculty", "Institute", "Laboratory",
    "Center", "Centre", "Division", "Section", "Unit", "Unknown", "",
    "School of Medicine", "School of Public Health", "School of Pharmacy",
    "School of Nursing", "School of Dentistry", "School of Engineering",
    "School of Science", "School of Technology", "School of Business",
    "Medical School", "Graduate School", "School of Basic Medicine",
    "School of Clinical Medicine", "School of Life Sciences",
    "School of Biological Sciences", "School of Chemistry",
    "School of Mathematics", "School of Physics", "School of Computer Science",
    "School of Information Technology", "School of Management",
    "School of Economics", "School of Education",
    "College of Medicine", "College of Science", "College of Engineering",
    "College of Pharmacy", "College of Nursing", "College of Dentistry",
    "Faculty of Medicine", "Faculty of Science", "Faculty of Engineering",
    "Faculty of Pharmacy", "Faculty of Dentistry",
    "Hospital", "Medical Center", "Medical Centre",
}
inst_stats = inst_stats[~inst_stats["institution"].isin(noise)]
inst_stats.to_csv(os.path.join(OUT, "wp1_institution_stats.csv"), index=False)
print(f"  Institution stats saved: {len(inst_stats)} institutions")

# ── 3. Figure: Top 25 countries ─────────────────────────────────────────────
print("Plotting top 25 countries …")
top25 = country_stats.head(25).copy()
fig, ax = plt.subplots(figsize=(12, 7))
bars = ax.barh(top25["country"][::-1], top25["retraction_count"][::-1],
               color=sns.color_palette("Reds_r", 25))
ax.set_xlabel("Number of Retracted Papers", fontsize=12)
ax.set_title("Top 25 Countries by Number of Retracted Papers\n(2000–2022, Retraction Watch + OpenAlex)", fontsize=13, fontweight="bold")
for bar, val in zip(bars, top25["retraction_count"][::-1]):
    ax.text(bar.get_width() + 80, bar.get_y() + bar.get_height()/2,
            f"{val:,}", va="center", fontsize=8.5)
ax.set_xlim(0, top25["retraction_count"].max() * 1.15)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_top_countries.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig_top_countries.png")

# ── 4. Figure: Country retraction trend over time (top 8) ───────────────────
print("Plotting country trends …")
top8 = country_stats.head(8)["country"].tolist()
trend = (
    cdf[cdf["country"].isin(top8) & cdf["retraction_year"].between(2000, 2022)]
       .groupby(["country", "retraction_year"])
       .size()
       .reset_index(name="count")
)
fig, ax = plt.subplots(figsize=(13, 6))
for i, country in enumerate(top8):
    sub = trend[trend["country"] == country].sort_values("retraction_year")
    ax.plot(sub["retraction_year"], sub["count"], marker="o", markersize=4,
            linewidth=2, label=country, color=PALETTE[i])
ax.set_xlabel("Retraction Year", fontsize=12)
ax.set_ylabel("Number of Retractions", fontsize=12)
ax.set_title("Retraction Trends Over Time — Top 8 Countries (2000–2022)", fontsize=13, fontweight="bold")
ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
ax.legend(loc="upper left", fontsize=9, ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_country_trend.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig_country_trend.png")

# ── 5. Figure: Top 30 institutions ──────────────────────────────────────────
print("Plotting top 30 institutions …")
top30 = inst_stats.head(30).copy()
fig, ax = plt.subplots(figsize=(12, 10))
colors = [sns.color_palette("Blues_r", 30)[i] for i in range(30)]
bars = ax.barh(top30["institution"][::-1], top30["retraction_count"][::-1], color=colors[::-1])
ax.set_xlabel("Number of Retracted Papers", fontsize=12)
ax.set_title("Top 30 Institutions by Number of Retracted Papers", fontsize=13, fontweight="bold")
for bar, val in zip(bars, top30["retraction_count"][::-1]):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
            f"{val:,}", va="center", fontsize=8)
ax.set_xlim(0, top30["retraction_count"].max() * 1.18)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_top_institutions.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig_top_institutions.png")

# ── 6. Figure: Retraction reason × country heatmap (top 15 countries) ───────
print("Plotting reason × country heatmap …")
top15_c = country_stats.head(15)["country"].tolist()
heat_data = (
    cdf[cdf["country"].isin(top15_c)]
       .groupby(["country", "reason_group"])
       .size()
       .unstack(fill_value=0)
)
# Normalise to row percentages
heat_pct = heat_data.div(heat_data.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(13, 7))
sns.heatmap(heat_pct, annot=True, fmt=".0f", cmap="YlOrRd",
            linewidths=0.4, ax=ax, cbar_kws={"label": "% of retractions"})
ax.set_title("Retraction Reason Profile by Country — Top 15 Countries\n(row percentages)", fontsize=13, fontweight="bold")
ax.set_xlabel("Retraction Reason Group", fontsize=11)
ax.set_ylabel("Country", fontsize=11)
plt.xticks(rotation=30, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_country_reason_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig_country_reason_heatmap.png")

# ── 7. Figure: Median citations by country (top 20) ─────────────────────────
print("Plotting median citations by country …")
top20_c = country_stats.head(20).copy().sort_values("median_citations", ascending=False)
fig, ax = plt.subplots(figsize=(12, 7))
bars = ax.barh(top20_c["country"][::-1], top20_c["median_citations"][::-1],
               color=sns.color_palette("Greens_r", 20)[::-1])
ax.set_xlabel("Median Citations Received", fontsize=12)
ax.set_title("Median Citations per Retracted Paper — Top 20 Countries by Retraction Count",
             fontsize=12, fontweight="bold")
for bar, val in zip(bars, top20_c["median_citations"][::-1]):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
            f"{val:.1f}", va="center", fontsize=8.5)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_country_citations.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig_country_citations.png")

# ── Summary ──────────────────────────────────────────────────────────────────
print()
print("=== Institutional Analysis Summary ===")
print(f"  Countries with retractions: {len(country_stats):,}")
print(f"  Top country: {country_stats.iloc[0]['country']} ({country_stats.iloc[0]['retraction_count']:,} papers)")
print(f"  Institutions identified: {len(inst_stats):,}")
print(f"  Top institution: {inst_stats.iloc[0]['institution']} ({inst_stats.iloc[0]['retraction_count']:,} papers)")
print()
print("All outputs written to:", OUT)
