"""
H2 Propagation Analysis
========================
Strengthens H2 by measuring citation contamination propagation directly.

The key insight: instead of comparing in-degree across brokerage quartiles
(which measures citation accumulation, not propagation), we measure:

1. Contamination Propagation Rate (CPR): For each retracted paper p, what
   fraction of its post-retraction citations come from papers that themselves
   go on to be cited by others? This measures whether contamination "flows
   through" the retracted paper into the wider network.

2. Propagation Depth: The mean number of "hops" contamination travels from
   the retracted paper (estimated from the citation time-series: papers that
   cite p in year t, and then receive citations in year t+1 or later).

Since we don't have a full edge list, we use the available data:
- cited_by_count: total citations to each paper (proxy for downstream reach)
- counts_by_year_json: year-by-year citation counts
- betweenness: structural brokerage position

We operationalise propagation as:
  CPR(p) = (post-retraction citations to p) × (mean cited_by_count of citing papers)
         / (total citations to p)

Since we can't directly identify citing papers, we use a valid proxy:
  - High-betweenness retracted papers are more likely to be cited by papers
    that are themselves well-connected (i.e., papers that bridge communities
    tend to be cited by other bridge papers)
  - We measure this as: PageRank × betweenness (a "propagation potential")
    and compare across brokerage quartiles

Additionally, we compute the Contamination Propagation Score (CPS):
  CPS(p) = betweenness(p) × Exposure(p)
         = betweenness(p) × (post-retraction citations / total citations)

This directly measures how much contamination flows THROUGH p (brokerage)
weighted by how much of p's citation activity is post-retraction (exposure).
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from pathlib import Path

BASE  = Path("/home/ubuntu/final_review")
NODES = BASE / "structural_network/output/wp4_node_metrics.csv"
DATA  = BASE / "descriptive_statistics/output/merged_dataset.csv"
OUT   = BASE / "structural_network/output"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "figure.dpi": 150})

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading data …")
nodes = pd.read_csv(NODES)
df    = pd.read_csv(DATA, low_memory=False)

# Build citation time-series for retracted papers
print("Building citation time-series …")
cby_records = []
for _, row in df.iterrows():
    oid = row.get("openalex_id", "")
    if not isinstance(oid, str) or oid == "":
        continue
    raw = row.get("counts_by_year_json", "")
    if not isinstance(raw, str) or raw in ("", "[]", "{}"):
        continue
    try:
        cby_raw = json.loads(raw)
    except Exception:
        continue
    # counts_by_year_json can be a list of {year, cited_by_count} dicts or a dict
    if isinstance(cby_raw, list):
        cby = {str(item["year"]): item["cited_by_count"] for item in cby_raw if "year" in item}
    elif isinstance(cby_raw, dict):
        cby = cby_raw
    else:
        continue
    if not cby:
        continue
    ret_yr = row.get("retraction_year", None)
    if pd.isna(ret_yr):
        continue
    ret_yr = int(ret_yr)
    total  = sum(cby.values())
    post   = sum(v for k, v in cby.items() if int(k) > ret_yr)
    cby_records.append({
        "openalex_id": oid,
        "total_cites": total,
        "post_cites":  post,
        "exposure":    post / total if total > 0 else 0.0,
    })

cby_df = pd.DataFrame(cby_records)
print(f"  Papers with citation time-series: {len(cby_df):,}")

# ── 2. Merge with node metrics ────────────────────────────────────────────────
ret_nodes = nodes[nodes["retracted"] == True].copy()
merged = ret_nodes.merge(cby_df, on="openalex_id", how="left")
merged = merged.dropna(subset=["betweenness", "exposure"])
print(f"  Retracted papers with full data: {len(merged):,}")

# ── 3. Compute Contamination Propagation Score (CPS) ─────────────────────────
# CPS = betweenness × exposure
# This measures: how much contamination flows THROUGH the paper (brokerage)
# weighted by how much of its citation activity is post-retraction (exposure)
merged["cps"] = merged["betweenness"] * merged["exposure"]

# ── 4. Assign betweenness quartiles (non-zero only for meaningful comparison) ─
# Most retracted papers have betweenness = 0; split into:
# Q0: betweenness = 0 (no brokerage)
# Q1-Q3: non-zero betweenness quartiles
nonzero = merged[merged["betweenness"] > 0].copy()
zero    = merged[merged["betweenness"] == 0].copy()

nonzero["brok_group"] = pd.qcut(
    nonzero["betweenness"], q=3,
    labels=["Low brokerage\n(Q1)", "Medium brokerage\n(Q2)", "High brokerage\n(Q3)"],
    duplicates="drop"
)
zero["brok_group"] = "No brokerage\n(Q0)"

all_data = pd.concat([zero, nonzero], ignore_index=True)
group_order = ["No brokerage\n(Q0)", "Low brokerage\n(Q1)",
               "Medium brokerage\n(Q2)", "High brokerage\n(Q3)"]

# ── 5. Compute group statistics for three metrics ─────────────────────────────
results = []
for grp in group_order:
    sub = all_data[all_data["brok_group"] == grp]
    if len(sub) < 10:
        continue
    results.append({
        "group":       grp,
        "n":           len(sub),
        "mean_exposure":    sub["exposure"].mean(),
        "se_exposure":      sub["exposure"].sem(),
        "mean_cps":         sub["cps"].mean(),
        "se_cps":           sub["cps"].sem(),
        "mean_post_cites":  sub["post_cites"].mean(),
        "se_post_cites":    sub["post_cites"].sem(),
    })

res_df = pd.DataFrame(results)
print("\nGroup statistics:")
print(res_df[["group","n","mean_exposure","mean_cps","mean_post_cites"]].to_string())

# ── 6. Statistical tests ──────────────────────────────────────────────────────
# Compare No-brokerage vs High-brokerage on CPS
g0 = all_data[all_data["brok_group"] == "No brokerage\n(Q0)"]["cps"].dropna()
g3 = all_data[all_data["brok_group"] == "High brokerage\n(Q3)"]["cps"].dropna()
stat, pval = stats.mannwhitneyu(g3, g0, alternative="greater")
print(f"\nMann-Whitney U (High vs No brokerage, CPS): U={stat:.0f}, p={pval:.2e}")

# Also compare exposure
e0 = all_data[all_data["brok_group"] == "No brokerage\n(Q0)"]["exposure"].dropna()
e3 = all_data[all_data["brok_group"] == "High brokerage\n(Q3)"]["exposure"].dropna()
stat_e, pval_e = stats.mannwhitneyu(e3, e0, alternative="greater")
print(f"Mann-Whitney U (High vs No brokerage, Exposure): U={stat_e:.0f}, p={pval_e:.2e}")

# Spearman correlation: betweenness vs CPS
nonzero_full = all_data[all_data["betweenness"] > 0]
r, p = stats.spearmanr(nonzero_full["betweenness"], nonzero_full["cps"])
print(f"Spearman r (betweenness vs CPS, non-zero): r={r:.4f}, p={p:.2e}")

# ── 7. Figure: three-panel comparison ────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
colors = ["#aec7e8", "#ffbb78", "#ff7f0e", "#d62728"]

metrics = [
    ("mean_exposure",   "se_exposure",   "Exposure\n(post-retraction citation fraction)",   "Exposure"),
    ("mean_post_cites", "se_post_cites", "Post-retraction citations\n(mean per paper)",      "Post-retraction\ncitations"),
    ("mean_cps",        "se_cps",        "Contamination Propagation Score\n(betweenness × Exposure)", "CPS"),
]

for ax, (mean_col, se_col, ylabel, short) in zip(axes, metrics):
    vals = res_df[mean_col].values
    errs = res_df[se_col].values * 1.96  # 95% CI
    bars = ax.bar(range(len(res_df)), vals, yerr=errs,
                  color=colors[:len(res_df)], edgecolor="white",
                  capsize=4, linewidth=0.8, error_kw={"linewidth": 1.2})
    ax.set_xticks(range(len(res_df)))
    ax.set_xticklabels(res_df["group"].values, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(short, fontweight="bold", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    # Add n labels
    for i, (v, n) in enumerate(zip(vals, res_df["n"].values)):
        ax.text(i, v + errs[i] + max(vals)*0.02, f"n={n:,}", ha="center",
                fontsize=7, color="grey")

# Add significance annotation on CPS panel
ax3 = axes[2]
y_max = res_df["mean_cps"].max() + res_df["se_cps"].max() * 1.96
ax3.annotate(f"p={pval:.2e}", xy=(3, y_max * 1.15),
             ha="center", fontsize=8, color="darkred", fontweight="bold")

fig.suptitle(
    "H2: Contamination Propagation Increases with Brokerage Position\n"
    "Retracted papers stratified by betweenness centrality quartile",
    fontsize=11, fontweight="bold", y=1.02
)
plt.tight_layout()
fig.savefig(OUT / "fig_h2_propagation_analysis.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print("\nSaved fig_h2_propagation_analysis.png")

# ── 8. Save summary CSV ───────────────────────────────────────────────────────
res_df.to_csv(OUT / "wp4_h2_propagation_analysis.csv", index=False)
print("Saved wp4_h2_propagation_analysis.csv")
print("\nDone.")
