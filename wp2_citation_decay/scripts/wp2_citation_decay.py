#!/usr/bin/env python3
"""
WP2 (v2): Citation Decay Event Study — Fixed & Enhanced
========================================================
Fixes:
  1. Field-year normalisation: citation counts are normalised by the mean
     citation count of all papers in the same field (top_concept) and
     publication year, reducing confounding from discipline-level citation norms.
  2. Per-period significance: each event-time estimate is tested against
     zero with a t-test; stars are added to the event-study plot.
  3. Adds a table of DiD estimates with p-values and stars.

Outputs (in wp2_citation_decay/output/):
  wp2_citation_long.csv
  wp2_event_study_estimates.csv       (now includes p_value, sig_stars)
  wp2_normalised_estimates.csv        (NEW: field-year normalised estimates)
  fig_avg_citation_trajectory.png
  fig_event_study_plot.png            (updated with significance markers)
  fig_event_study_normalised.png      (NEW: normalised event study)
  fig_persistence_by_reason.png
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "shared"))

import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from utils import RAW_DATA, is_misconduct

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
EVENT_WINDOW = range(-10, 11)

def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)
    print(f"  Saved {name}")

# ---------------------------------------------------------------------------
# Load merged dataset
# ---------------------------------------------------------------------------
merged_path = RAW_DATA / "merged_dataset.csv"
if not merged_path.exists():
    merged_path = RAW_DATA.parent.parent / "wp1_descriptive" / "output" / "merged_dataset.csv"

print("Loading merged dataset …")
df = pd.read_csv(merged_path, low_memory=False)
print(f"  Rows: {len(df):,}")

df["retraction_year"] = pd.to_numeric(df.get("retraction_year",
    pd.to_datetime(df.get("RetractionDate",""), errors="coerce").dt.year), errors="coerce")
df["publication_year"] = pd.to_numeric(df["publication_year"], errors="coerce")
df["is_misconduct"] = df["Reason"].apply(is_misconduct)

def parse_cby(s):
    try:
        return {int(d["year"]): int(d["cited_by_count"]) for d in json.loads(s)}
    except Exception:
        return {}

# ---------------------------------------------------------------------------
# Build long-format citation panel
# ---------------------------------------------------------------------------
print("Building citation panel …")
rows = []
for _, row in df.iterrows():
    cby = parse_cby(row.get("counts_by_year_json", "[]"))
    ret_yr = row.get("retraction_year")
    pub_yr = row.get("publication_year")
    concept = row.get("top_concept", "Unknown")
    misconduct = row.get("is_misconduct", False)
    if pd.isna(ret_yr) or pd.isna(pub_yr):
        continue
    ret_yr = int(ret_yr); pub_yr = int(pub_yr)
    for t in EVENT_WINDOW:
        cal_yr = ret_yr + t
        citations = cby.get(cal_yr, 0)
        rows.append({
            "doi": row.get("doi_normalised", ""),
            "pub_year": pub_yr,
            "ret_year": ret_yr,
            "event_time": t,
            "cal_year": cal_yr,
            "citations": citations,
            "concept": concept,
            "is_misconduct": misconduct,
        })

long = pd.DataFrame(rows)
long.to_csv(OUT / "wp2_citation_long.csv", index=False)
print(f"  Long panel: {len(long):,} rows  ({long['doi'].nunique():,} papers)")

# ---------------------------------------------------------------------------
# Field-year normalisation
# ---------------------------------------------------------------------------
print("Computing field-year citation norms …")
field_year_means = (long.groupby(["concept", "cal_year"])["citations"]
                    .mean().rename("field_year_mean").reset_index())
long = long.merge(field_year_means, on=["concept", "cal_year"], how="left")
long["citations_norm"] = np.where(
    long["field_year_mean"] > 0,
    long["citations"] / long["field_year_mean"],
    np.nan
)

# ---------------------------------------------------------------------------
# Event-study estimates (raw)
# ---------------------------------------------------------------------------
print("Computing event-study estimates …")
baseline_mask = long["event_time"].isin([-3, -2, -1])
baseline = long[baseline_mask].groupby("doi")["citations"].mean().rename("baseline")
long = long.merge(baseline, on="doi", how="left")

pre_mean = long[long["event_time"] < 0]["citations"].mean()

est_rows = []
for t in EVENT_WINDOW:
    sub = long[long["event_time"] == t]["citations"].dropna()
    if len(sub) < 2:
        continue
    mean_c = sub.mean()
    se = sub.sem()
    did = mean_c - pre_mean
    tstat, pval = stats.ttest_1samp(sub - pre_mean, 0)
    stars = ("***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "")
    est_rows.append({
        "event_time": t, "mean_citations": mean_c, "se": se, "n": len(sub),
        "did_estimate": did,
        "ci_lower": did - 1.96 * se, "ci_upper": did + 1.96 * se,
        "p_value": round(pval, 4), "sig_stars": stars,
    })

est = pd.DataFrame(est_rows)
est.to_csv(OUT / "wp2_event_study_estimates.csv", index=False)
print("  Saved wp2_event_study_estimates.csv")

# ---------------------------------------------------------------------------
# Event-study estimates (normalised)
# ---------------------------------------------------------------------------
pre_mean_norm = long[long["event_time"] < 0]["citations_norm"].mean()
norm_rows = []
for t in EVENT_WINDOW:
    sub = long[long["event_time"] == t]["citations_norm"].dropna()
    if len(sub) < 2:
        continue
    mean_c = sub.mean()
    se = sub.sem()
    did = mean_c - pre_mean_norm
    tstat, pval = stats.ttest_1samp(sub.dropna() - pre_mean_norm, 0)
    stars = ("***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "")
    norm_rows.append({
        "event_time": t, "mean_norm_citations": mean_c, "se": se, "n": len(sub),
        "did_estimate": did,
        "ci_lower": did - 1.96 * se, "ci_upper": did + 1.96 * se,
        "p_value": round(pval, 4), "sig_stars": stars,
    })

norm_est = pd.DataFrame(norm_rows)
norm_est.to_csv(OUT / "wp2_normalised_estimates.csv", index=False)
print("  Saved wp2_normalised_estimates.csv")

# ---------------------------------------------------------------------------
# Figure 1: Average citation trajectory
# ---------------------------------------------------------------------------
traj = long.groupby("event_time")["citations"].mean()
fig, ax = plt.subplots()
ax.plot(traj.index, traj.values, color=BLUE, linewidth=2, marker="o", markersize=4)
ax.axvline(0, color=RED, linestyle="--", linewidth=1.5, label="Retraction year")
ax.set_title("Mean Annual Citations Around Retraction Year")
ax.set_xlabel("Years Relative to Retraction"); ax.set_ylabel("Mean Annual Citations")
ax.legend()
save(fig, "fig_avg_citation_trajectory.png")

# ---------------------------------------------------------------------------
# Figure 2: Event-study plot (raw, with significance stars)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots()
ax.fill_between(est["event_time"], est["ci_lower"], est["ci_upper"],
                alpha=0.2, color=BLUE)
ax.plot(est["event_time"], est["did_estimate"], color=BLUE,
        linewidth=2, marker="o", markersize=5,
        label="DiD estimate vs. pre-retraction baseline (±95% CI)")
ax.axhline(0, color="black", linewidth=0.8)
ax.axvline(0, color=RED, linestyle="--", linewidth=1.5, label="Retraction year")
# Add significance stars
for _, row in est.iterrows():
    if row["sig_stars"]:
        ax.text(row["event_time"], row["did_estimate"] + 0.12,
                row["sig_stars"], ha="center", fontsize=9, color=RED)
ax.set_title("Event-Study: Change in Citations Relative to Pre-Retraction Baseline")
ax.set_xlabel("Years Relative to Retraction")
ax.set_ylabel("Change in Mean Annual Citations")
ax.legend()
save(fig, "fig_event_study_plot.png")

# ---------------------------------------------------------------------------
# Figure 3: Normalised event-study plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots()
ax.fill_between(norm_est["event_time"], norm_est["ci_lower"], norm_est["ci_upper"],
                alpha=0.2, color=GREEN)
ax.plot(norm_est["event_time"], norm_est["did_estimate"], color=GREEN,
        linewidth=2, marker="o", markersize=5,
        label="Field-year normalised DiD (±95% CI)")
ax.axhline(0, color="black", linewidth=0.8)
ax.axvline(0, color=RED, linestyle="--", linewidth=1.5, label="Retraction year")
for _, row in norm_est.iterrows():
    if row["sig_stars"]:
        ax.text(row["event_time"], row["did_estimate"] + 0.005,
                row["sig_stars"], ha="center", fontsize=9, color=RED)
ax.set_title("Event-Study: Field-Year Normalised Citation Change")
ax.set_xlabel("Years Relative to Retraction")
ax.set_ylabel("Change in Normalised Citations (relative to field-year mean)")
ax.legend()
save(fig, "fig_event_study_normalised.png")

# ---------------------------------------------------------------------------
# Figure 4: Persistence by reason (misconduct vs error)
# ---------------------------------------------------------------------------
post = long[long["event_time"] > 0]
misc_traj = post[post["is_misconduct"]].groupby("event_time")["citations"].mean()
err_traj  = post[~post["is_misconduct"]].groupby("event_time")["citations"].mean()
fig, ax = plt.subplots()
ax.axvline(0, color=RED, linestyle="--", linewidth=1, alpha=0.5)
ax.plot(misc_traj.index, misc_traj.values, color=RED, linewidth=2,
        marker="o", markersize=4, label="Misconduct/Fraud")
ax.plot(err_traj.index,  err_traj.values,  color=BLUE, linewidth=2,
        marker="o", markersize=4, label="Error/Other")
ax.set_title("Post-Retraction Citation Persistence: Misconduct vs. Error")
ax.set_xlabel("Years After Retraction"); ax.set_ylabel("Mean Annual Citations")
ax.legend()
save(fig, "fig_persistence_by_reason.png")

print("\n=== WP2 Summary ===")
print(f"  Papers in panel: {long['doi'].nunique():,}")
print(f"  DiD at t=0 (retraction year): {est[est['event_time']==0]['did_estimate'].values[0]:.3f}")
print(f"  DiD at t=5: {est[est['event_time']==5]['did_estimate'].values[0]:.3f}")
print("\nWP2 complete. All outputs in wp2_citation_decay/output/")
