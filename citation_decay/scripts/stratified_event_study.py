"""
Stratified Event Study — WP2 Revision
======================================
Computes DiD citation trajectories stratified by:
  1. Retraction reason group
  2. Discipline (top_concept)
  3. Pre-retraction citation percentile
  4. Time-to-retraction group
Outputs figures and CSV tables to citation_decay/output/
"""

import json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = Path("/home/ubuntu/final_review")
DATA = BASE / "descriptive_statistics/output/merged_dataset.csv"
OUT  = BASE / "citation_decay/output"
OUT.mkdir(parents=True, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})
PALETTE = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
           "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading data …")
df = pd.read_csv(DATA, low_memory=False)
df["publication_year"] = pd.to_numeric(df["publication_year"], errors="coerce")
df["retraction_year"]  = pd.to_numeric(df["retraction_year"],  errors="coerce")
df["time_to_retraction"] = df["retraction_year"] - df["publication_year"]
df = df.dropna(subset=["publication_year","retraction_year","counts_by_year_json"])

# ── Parse citation time series ───────────────────────────────────────────────
def parse_cby(raw):
    try:
        d = json.loads(raw)
        return {int(e["year"]): int(e["cited_by_count"]) for e in d}
    except Exception:
        return {}

df["cby"] = df["counts_by_year_json"].apply(parse_cby)

# ── Reason groups ────────────────────────────────────────────────────────────
REASON_MAP = {
    "Paper Mill":            "Paper Mill",
    "Fake Peer Review":      "Peer Review Manip.",
    "Falsification":         "Misconduct/Fraud",
    "Fabrication":           "Misconduct/Fraud",
    "Manipulation of Image": "Misconduct/Fraud",
    "Fraud":                 "Misconduct/Fraud",
    "Plagiarism":            "Plagiarism/Duplication",
    "Duplication":           "Plagiarism/Duplication",
    "Results Not Reproducible": "Error/Unreliable",
    "Error in Data":         "Error/Unreliable",
    "Error in Methods":      "Error/Unreliable",
    "Error in Analyses":     "Error/Unreliable",
    "Unreliable Results":    "Error/Unreliable",
    "Concerns/Issues":       "Error/Unreliable",
}

def map_reason(raw):
    if pd.isna(raw): return "Other/Unclear"
    for key, grp in REASON_MAP.items():
        if key.lower() in str(raw).lower():
            return grp
    return "Other/Unclear"

df["reason_group"] = df["Reason"].apply(map_reason)

# ── Citation percentile (pre-retraction) ─────────────────────────────────────
df["cite_pct"] = pd.qcut(df["cited_by_count"], q=4,
                          labels=["Q1 (low)","Q2","Q3","Q4 (high)"])

# ── Time-to-retraction group ─────────────────────────────────────────────────
def ttr_group(t):
    if   t <= 1:  return "Fast (≤1 yr)"
    elif t <= 5:  return "Medium (2–5 yr)"
    else:         return "Slow (>5 yr)"

df["ttr_group"] = df["time_to_retraction"].apply(ttr_group)

# ── Discipline (top_concept, top 6) ─────────────────────────────────────────
top_disc = df["top_concept"].value_counts().head(6).index.tolist()
df["discipline"] = df["top_concept"].where(df["top_concept"].isin(top_disc), other="Other")

# ── Core event-study function ─────────────────────────────────────────────────
def event_study(sub, window=(-8, 8)):
    """Return DataFrame with event_time, mean_cites, ci_lo, ci_hi, n."""
    records = []
    lo, hi = window
    for _, row in sub.iterrows():
        ret_yr = int(row["retraction_year"])
        cby    = row["cby"]
        for t in range(lo, hi+1):
            yr = ret_yr + t
            c  = cby.get(yr, 0)
            records.append({"event_time": t, "citations": c})
    if not records:
        return pd.DataFrame()
    tmp = pd.DataFrame(records)
    agg = tmp.groupby("event_time")["citations"].agg(
        mean_cites="mean",
        std_cites="std",
        n="count"
    ).reset_index()
    agg["se"] = agg["std_cites"] / np.sqrt(agg["n"])
    agg["ci_lo"] = agg["mean_cites"] - 1.96 * agg["se"]
    agg["ci_hi"] = agg["mean_cites"] + 1.96 * agg["se"]
    return agg

# ── Plot helper ───────────────────────────────────────────────────────────────
def plot_strata(strata_dict, title, fname, ylabel="Mean citations per paper per year"):
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (label, agg) in enumerate(strata_dict.items()):
        if agg.empty: continue
        color = PALETTE[i % len(PALETTE)]
        ax.plot(agg["event_time"], agg["mean_cites"], marker="o", ms=4,
                color=color, label=label, linewidth=1.6)
        ax.fill_between(agg["event_time"], agg["ci_lo"], agg["ci_hi"],
                        alpha=0.15, color=color)
    ax.axvline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.7)
    ax.axvline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.7)
    ax.set_xlabel("Years relative to retraction (t = 0)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.legend(fontsize=8, framealpha=0.9, ncol=2)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    plt.tight_layout()
    fig.savefig(OUT / fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")

# ── 1. Stratify by reason group ───────────────────────────────────────────────
print("1. Stratified by retraction reason …")
reason_strata = {}
for grp in ["Paper Mill","Peer Review Manip.","Misconduct/Fraud",
            "Error/Unreliable","Plagiarism/Duplication","Other/Unclear"]:
    sub = df[df["reason_group"] == grp]
    if len(sub) >= 50:
        reason_strata[f"{grp} (n={len(sub):,})"] = event_study(sub)

plot_strata(reason_strata,
            "Citation Trajectories by Retraction Reason",
            "fig_event_study_by_reason.png")

# ── 2. Stratify by discipline ─────────────────────────────────────────────────
print("2. Stratified by discipline …")
disc_strata = {}
for disc in top_disc + ["Other"]:
    sub = df[df["discipline"] == disc]
    if len(sub) >= 50:
        disc_strata[f"{disc} (n={len(sub):,})"] = event_study(sub)

plot_strata(disc_strata,
            "Citation Trajectories by Discipline",
            "fig_event_study_by_discipline.png")

# ── 3. Stratify by pre-retraction citation percentile ────────────────────────
print("3. Stratified by citation percentile …")
pct_strata = {}
for pct in ["Q1 (low)","Q2","Q3","Q4 (high)"]:
    sub = df[df["cite_pct"] == pct]
    if len(sub) >= 50:
        pct_strata[f"{pct} (n={len(sub):,})"] = event_study(sub)

plot_strata(pct_strata,
            "Citation Trajectories by Pre-Retraction Citation Quartile",
            "fig_event_study_by_citation_quartile.png")

# ── 4. Stratify by time-to-retraction ────────────────────────────────────────
print("4. Stratified by time-to-retraction …")
ttr_strata = {}
for grp in ["Fast (≤1 yr)","Medium (2–5 yr)","Slow (>5 yr)"]:
    sub = df[df["ttr_group"] == grp]
    if len(sub) >= 50:
        ttr_strata[f"{grp} (n={len(sub):,})"] = event_study(sub)

plot_strata(ttr_strata,
            "Citation Trajectories by Time-to-Retraction",
            "fig_event_study_by_ttr.png")

# ── 5. Save summary CSV ───────────────────────────────────────────────────────
rows = []
for strat_name, strata_dict in [
    ("reason", reason_strata),
    ("discipline", disc_strata),
    ("citation_quartile", pct_strata),
    ("ttr_group", ttr_strata),
]:
    for label, agg in strata_dict.items():
        if agg.empty: continue
        agg = agg.copy()
        agg["stratum_type"]  = strat_name
        agg["stratum_label"] = label
        rows.append(agg)

if rows:
    pd.concat(rows, ignore_index=True).to_csv(
        OUT / "wp2_stratified_event_study.csv", index=False)
    print("  Saved wp2_stratified_event_study.csv")

print("Done — stratified event study complete.")
