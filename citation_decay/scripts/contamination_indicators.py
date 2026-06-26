"""
Contamination Indicator Family — WP2 Extension
================================================
Defines and computes four bibliometric indicators for scholarly units U:

  Exposure(U)     = fraction of U's outgoing citations pointing to
                    eventually-retracted papers
  Contamination(U,t) = post-retraction citations received by papers in U /
                    all citations received by papers in U
  Persistence(U)  = mean years post-retraction that citations from U continue
  Recovery(U)     = rate of decline of Contamination after t=0 (slope of
                    linear fit on post-retraction contamination series)

Units: journal, publisher, country, discipline (top_concept)
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

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading data …")
df = pd.read_csv(DATA, low_memory=False)
df["publication_year"] = pd.to_numeric(df["publication_year"], errors="coerce")
df["retraction_year"]  = pd.to_numeric(df["retraction_year"],  errors="coerce")
df = df.dropna(subset=["publication_year","retraction_year","counts_by_year_json"])

def parse_cby(raw):
    try:
        d = json.loads(raw)
        return {int(e["year"]): int(e["cited_by_count"]) for e in d}
    except Exception:
        return {}

df["cby"] = df["counts_by_year_json"].apply(parse_cby)

# ── Contamination(U, t) per unit ─────────────────────────────────────────────
def contamination_series(sub, window=(-5, 10)):
    """
    For a group of retracted papers, compute:
      - total citations per event-time year
      - post-retraction citations per event-time year
      - Contamination(t) = post / total
    """
    lo, hi = window
    records = []
    for _, row in sub.iterrows():
        ret_yr = int(row["retraction_year"])
        cby    = row["cby"]
        for t in range(lo, hi+1):
            yr = ret_yr + t
            c  = cby.get(yr, 0)
            records.append({"event_time": t, "citations": c,
                             "is_post": int(t > 0)})
    if not records:
        return pd.DataFrame()
    tmp = pd.DataFrame(records)
    agg = tmp.groupby("event_time").agg(
        total_cites=("citations","sum"),
        post_cites=("citations", lambda x: x[tmp.loc[x.index,"is_post"]==1].sum()),
    ).reset_index()
    agg["contamination"] = np.where(
        agg["total_cites"] > 0,
        agg["post_cites"] / agg["total_cites"],
        0.0
    )
    return agg

# ── Compute indicators per unit ───────────────────────────────────────────────
def compute_indicators(df_sub, unit_col, top_n=20, min_papers=30):
    """Return DataFrame of indicators per unit value."""
    rows = []
    top_vals = df_sub[unit_col].value_counts().head(top_n).index.tolist()
    for val in top_vals:
        sub = df_sub[df_sub[unit_col] == val]
        if len(sub) < min_papers:
            continue
        n = len(sub)
        # Contamination series
        cs = contamination_series(sub)
        if cs.empty:
            continue
        post = cs[cs["event_time"] > 0]
        pre  = cs[cs["event_time"] <= 0]
        # Contamination = mean post-retraction contamination ratio
        contamination = post["contamination"].mean() if not post.empty else 0.0
        # Persistence = mean years post-retraction with >0 citations
        persist_years = post[post["total_cites"] > 0]["event_time"].max() if not post.empty else 0
        # Recovery = slope of contamination decline post-retraction
        if len(post) >= 3:
            slope, _, _, _, _ = np.polyfit(post["event_time"], post["contamination"], 1, full=False) if False else \
                                (np.polyfit(post["event_time"].values, post["contamination"].values, 1)[0], 0, 0, 0, 0)
        else:
            slope = 0.0
        # Exposure = fraction of total citations that are post-retraction
        total_all = cs["total_cites"].sum()
        total_post = cs[cs["event_time"] > 0]["total_cites"].sum()
        exposure = total_post / total_all if total_all > 0 else 0.0
        rows.append({
            "unit": val,
            "n_papers": n,
            "exposure": round(exposure, 4),
            "contamination": round(contamination, 4),
            "persistence_yrs": round(persist_years, 1),
            "recovery_slope": round(slope, 5),
        })
    return pd.DataFrame(rows).sort_values("contamination", ascending=False)

# ── Run for each unit type ────────────────────────────────────────────────────
units = {
    "journal":    ("journal_name",  "Journal"),
    "publisher":  ("Publisher",     "Publisher"),
    "country":    ("Country",       "Country"),
    "discipline": ("top_concept",   "Discipline"),
}

all_indicators = {}
for key, (col, label) in units.items():
    if col not in df.columns:
        print(f"  Skipping {key} — column {col} not found")
        continue
    print(f"Computing indicators for {label} …")
    ind = compute_indicators(df, col, top_n=30, min_papers=30)
    if ind.empty:
        print(f"  No results for {label}")
        continue
    ind.to_csv(OUT / f"wp2_contamination_{key}.csv", index=False)
    all_indicators[key] = (ind, label, col)
    print(f"  {len(ind)} units — saved wp2_contamination_{key}.csv")

# ── Plot: Contamination bar charts ───────────────────────────────────────────
def bar_chart(ind, label, fname, top_n=20, metric="contamination",
              xlabel="Contamination Index"):
    sub = ind.head(top_n).sort_values(metric)
    fig, ax = plt.subplots(figsize=(8, max(4, len(sub)*0.35)))
    colors = ["#d62728" if v > sub[metric].median() else "#1f77b4"
              for v in sub[metric]]
    bars = ax.barh(sub["unit"].str[:45], sub[metric], color=colors, height=0.7)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_title(f"Top {top_n} {label}s by {xlabel}", fontsize=12,
                 fontweight="bold", pad=10)
    ax.axvline(sub[metric].median(), color="grey", linestyle="--",
               linewidth=1, alpha=0.7, label="Median")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(OUT / fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")

for key, (ind, label, col) in all_indicators.items():
    bar_chart(ind, label, f"fig_contamination_{key}.png")

# ── Combined summary figure: 4-panel ─────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
for ax, (key, (ind, label, col)) in zip(axes, all_indicators.items()):
    sub = ind.head(15).sort_values("contamination")
    colors = ["#d62728" if v > sub["contamination"].median() else "#1f77b4"
              for v in sub["contamination"]]
    ax.barh(sub["unit"].str[:35], sub["contamination"], color=colors, height=0.7)
    ax.set_xlabel("Contamination Index", fontsize=9)
    ax.set_title(f"By {label}", fontsize=10, fontweight="bold")
    ax.tick_params(axis="y", labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.suptitle("Contamination Index Across Scholarly Units\n"
             "(Post-retraction citations / all citations)", 
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
fig.savefig(OUT / "fig_contamination_4panel.png", bbox_inches="tight")
plt.close(fig)
print("  Saved fig_contamination_4panel.png")

# ── Indicator definitions table (for paper) ───────────────────────────────────
defs = pd.DataFrame([
    ["Exposure(U)",       "Fraction of U's total citations that are post-retraction",
     r"$\sum_{t>0} C(U,t) / \sum_t C(U,t)$"],
    ["Contamination(U,t)","Post-retraction citations received by U / all citations received by U at time t",
     r"$C_{\mathrm{post}}(U,t) / C(U,t)$"],
    ["Persistence(U)",    "Latest year post-retraction with at least one citation from U",
     r"$\max\{t > 0 : C(U,t) > 0\}$"],
    ["Recovery(U)",       "Slope of linear fit on Contamination(U,t) for t > 0 (negative = recovering)",
     r"$\hat{\beta}_1$ from OLS on $\mathrm{CI}(U,t) = \beta_0 + \beta_1 t$"],
], columns=["Indicator","Definition","Formula"])
defs.to_csv(OUT / "wp2_indicator_definitions.csv", index=False)
print("  Saved wp2_indicator_definitions.csv")

print("\nDone — contamination indicators complete.")
