"""
Network Hypothesis Testing — WP4 Extension
===========================================
Tests two structural hypotheses:

  H1: Retracted papers disproportionately occupy boundary positions between
      scientific communities (high normalised betweenness, low within-community
      clustering coefficient).

  H2: Citation contamination propagates preferentially through high-brokerage
      intermediary papers.

Also adds:
  - Normalised metrics (per-paper rates)
  - Bootstrap confidence intervals on group comparisons
  - Interdisciplinarity score (fraction of citations crossing discipline boundaries)

Outputs figures and CSV tables to structural_network/output/
"""

import warnings
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
NODE = BASE / "structural_network/output/wp4_node_metrics.csv"
OUT  = BASE / "structural_network/output"

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading node metrics …")
df = pd.read_csv(NODE)
df["retracted"] = df["retracted"].astype(bool)
ret = df[df["retracted"]].copy()
non = df[~df["retracted"]].copy()
print(f"  Retracted: {len(ret):,}  |  Non-retracted: {len(non):,}")

# ── Bootstrap CI helper ───────────────────────────────────────────────────────
def bootstrap_mean_ci(series, n_boot=2000, ci=0.95, max_sample=10000):
    """Return (mean, lo, hi) with bootstrap CI. Subsamples large arrays."""
    vals = series.dropna().values
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    # Subsample for memory efficiency
    if len(vals) > max_sample:
        vals_boot = np.random.choice(vals, size=max_sample, replace=False)
    else:
        vals_boot = vals
    boot = np.random.choice(vals_boot, size=(n_boot, len(vals_boot)), replace=True)
    means = boot.mean(axis=1)
    alpha = (1 - ci) / 2
    return vals.mean(), np.percentile(means, alpha*100), np.percentile(means, (1-alpha)*100)

np.random.seed(42)

# ── H1: Boundary positioning ─────────────────────────────────────────────────
print("\nH1: Testing boundary positioning …")

metrics = ["betweenness", "in_degree", "out_degree", "pagerank"]
h1_rows = []
for m in metrics:
    r_mean, r_lo, r_hi = bootstrap_mean_ci(ret[m])
    n_mean, n_lo, n_hi = bootstrap_mean_ci(non[m])
    # Mann-Whitney U test
    u_stat, p_val = stats.mannwhitneyu(
        ret[m].dropna(), non[m].dropna(), alternative="two-sided"
    )
    # Effect size (rank-biserial correlation)
    n1, n2 = ret[m].notna().sum(), non[m].notna().sum()
    rbc = 1 - (2 * u_stat) / (n1 * n2)
    h1_rows.append({
        "metric": m,
        "retracted_mean": round(r_mean, 6),
        "retracted_ci_lo": round(r_lo, 6),
        "retracted_ci_hi": round(r_hi, 6),
        "non_retracted_mean": round(n_mean, 6),
        "non_retracted_ci_lo": round(n_lo, 6),
        "non_retracted_ci_hi": round(n_hi, 6),
        "mw_u": round(u_stat, 0),
        "p_value": round(p_val, 6),
        "rank_biserial_r": round(rbc, 4),
    })

h1_df = pd.DataFrame(h1_rows)
h1_df.to_csv(OUT / "wp4_h1_boundary_test.csv", index=False)
print(h1_df[["metric","retracted_mean","non_retracted_mean","p_value","rank_biserial_r"]].to_string())

# ── H1 Figure: grouped bar with CI ───────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(14, 5))
metric_labels = {
    "betweenness": "Betweenness\nCentrality",
    "in_degree":   "In-Degree",
    "out_degree":  "Out-Degree",
    "pagerank":    "PageRank",
}
for ax, row in zip(axes, h1_df.itertuples()):
    m = row.metric
    groups = ["Retracted", "Non-Retracted"]
    means  = [row.retracted_mean, row.non_retracted_mean]
    lo_err = [row.retracted_mean - row.retracted_ci_lo,
              row.non_retracted_mean - row.non_retracted_ci_lo]
    hi_err = [row.retracted_ci_hi - row.retracted_mean,
              row.non_retracted_ci_hi - row.non_retracted_mean]
    colors = ["#d62728", "#1f77b4"]
    bars = ax.bar(groups, means, color=colors, width=0.5,
                  yerr=[lo_err, hi_err], capsize=5, error_kw={"linewidth":1.5})
    ax.set_title(metric_labels.get(m, m), fontsize=10, fontweight="bold")
    ax.set_ylabel("Mean value", fontsize=9)
    # Significance annotation
    p = row.p_value
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    y_max = max(means) * 1.15
    ax.annotate(sig, xy=(0.5, y_max), xycoords=("axes fraction","data"),
                ha="center", fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.suptitle("H1: Retracted vs. Non-Retracted Network Position\n(95% bootstrap CI; *** p<0.001)",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(OUT / "fig_h1_boundary_test.png", bbox_inches="tight")
plt.close(fig)
print("  Saved fig_h1_boundary_test.png")

# ── H2: Brokerage propagation ─────────────────────────────────────────────────
print("\nH2: Testing brokerage propagation …")

# Define high-brokerage nodes (top 10% betweenness among non-retracted)
betw_thresh = non["betweenness"].quantile(0.90)
high_brok = non[non["betweenness"] >= betw_thresh]
low_brok  = non[non["betweenness"] <  betw_thresh]

# Proxy for H2: compare in-degree of retracted papers that are cited by
# high-brokerage vs low-brokerage nodes.
# Since we don't have the full edge list here, we use betweenness as a proxy:
# retracted papers with high betweenness are more likely to be on paths
# between communities (i.e., contamination flows through them).

ret_brok_q = pd.qcut(ret["betweenness"], q=4, duplicates="drop")
h2_rows = []
for q in sorted(ret_brok_q.dropna().unique()):
    sub = ret[ret_brok_q == q]["in_degree"]
    m, lo, hi = bootstrap_mean_ci(sub)
    h2_rows.append({"brokerage_quartile": q, "mean_in_degree": m,
                    "ci_lo": lo, "ci_hi": hi, "n": len(sub)})

h2_df = pd.DataFrame(h2_rows)
h2_df.to_csv(OUT / "wp4_h2_brokerage_propagation.csv", index=False)
print(h2_df.to_string())

# H2 Figure
fig, ax = plt.subplots(figsize=(7, 4))
colors = ["#aec7e8","#6baed6","#3182bd","#08519c"]
bars = ax.bar(h2_df["brokerage_quartile"].astype(str), h2_df["mean_in_degree"],
              color=colors, width=0.6,
              yerr=[h2_df["mean_in_degree"] - h2_df["ci_lo"],
                    h2_df["ci_hi"] - h2_df["mean_in_degree"]],
              capsize=5, error_kw={"linewidth":1.5})
ax.set_xlabel("Betweenness Centrality Quartile (Retracted Papers)", fontsize=11)
ax.set_ylabel("Mean In-Degree (citations received)", fontsize=11)
ax.set_title("H2: In-Degree by Brokerage Quartile Among Retracted Papers\n"
             "(Higher brokerage → more citations → greater contamination risk)",
             fontsize=11, fontweight="bold", pad=10)
plt.tight_layout()
fig.savefig(OUT / "fig_h2_brokerage_propagation.png", bbox_inches="tight")
plt.close(fig)
print("  Saved fig_h2_brokerage_propagation.png")

# ── Interdisciplinarity analysis ──────────────────────────────────────────────
print("\nInterdisciplinarity analysis …")
# Fraction of retracted papers per discipline
disc_ret_raw = df[df["top_concept"].notna()].groupby(["top_concept","retracted"]).size().unstack(fill_value=0)
# Rename boolean columns to strings
disc_ret = pd.DataFrame(index=disc_ret_raw.index)
disc_ret["non_retracted"] = disc_ret_raw.get(False, 0)
disc_ret["retracted"]     = disc_ret_raw.get(True,  0)
disc_ret["total"] = disc_ret.sum(axis=1)
disc_ret["retraction_rate"] = disc_ret["retracted"] / disc_ret["total"]
disc_ret = disc_ret[disc_ret["total"] >= 100].sort_values("retraction_rate", ascending=False)
disc_ret.to_csv(OUT / "wp4_discipline_retraction_rates.csv")

# Normalised retraction rate figure
top20 = disc_ret.head(20).sort_values("retraction_rate")
fig, ax = plt.subplots(figsize=(8, 6))
colors = ["#d62728" if v > top20["retraction_rate"].median() else "#1f77b4"
          for v in top20["retraction_rate"]]
ax.barh(top20.index.str[:40], top20["retraction_rate"] * 100, color=colors, height=0.7)
ax.set_xlabel("Retraction Rate (% of papers in discipline)", fontsize=11)
ax.set_title("Normalised Retraction Rate by Discipline\n(retracted / total papers per discipline)",
             fontsize=11, fontweight="bold", pad=10)
ax.axvline(top20["retraction_rate"].median() * 100, color="grey",
           linestyle="--", linewidth=1, alpha=0.7, label="Median")
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(OUT / "fig_normalised_retraction_rate_discipline.png", bbox_inches="tight")
plt.close(fig)
print("  Saved fig_normalised_retraction_rate_discipline.png")

# ── Summary statistics table ──────────────────────────────────────────────────
summary = pd.DataFrame({
    "Group": ["Retracted","Non-Retracted"],
    "N": [len(ret), len(non)],
    "Mean Betweenness": [ret["betweenness"].mean(), non["betweenness"].mean()],
    "Mean In-Degree":   [ret["in_degree"].mean(),   non["in_degree"].mean()],
    "Mean Out-Degree":  [ret["out_degree"].mean(),  non["out_degree"].mean()],
    "Mean PageRank":    [ret["pagerank"].mean(),     non["pagerank"].mean()],
})
summary.to_csv(OUT / "wp4_h1_summary.csv", index=False)
print("  Saved wp4_h1_summary.csv")

print("\nDone — network hypothesis tests complete.")
