#!/usr/bin/env python3
"""Step 8: direct, time-respecting test of H2 (brokerage and citation persistence).

Narrowed H2: Among retracted papers, higher pre-retraction local focal-path
brokerage is associated with greater post-retraction citation persistence.

The script does NOT claim to observe multi-step epistemic propagation. It uses
only the Step 6 cached citation neighbourhoods and annual citation histories.
For each retracted focal paper p, its brokerage score is constructed from works
published before p's retraction year that cite p (sources) and references cited
by p (targets). A source-target pair is counted as brokered by p when the source
cites p and p cites the target but the selected source-reference list does not
contain a direct source-to-target citation. This is a local, sampled, directed,
pre-retraction path count.

The primary persistence outcome is log(1 + citations in years r+1 ... r+H), with
H=4 by default. The adjusted association controls for log(1 + pre-retraction
citations in years r-H ... r-1) and log paper age at retraction. Associations are
descriptive and cannot establish that citations transmitted flawed knowledge.

Run from the project root:
    python revision/scripts/10_test_h2_brokerage_persistence.py
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import latex_escape, now_utc, project_root, revision_dirs, save_dual_figure, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 8: time-respecting brokerage-persistence association for H2.")
    parser.add_argument("--window", type=int, default=4, choices=(1, 2, 3, 4), help="Post/pre half-window H (default: 4).")
    parser.add_argument("--observation-end", type=int, default=2025, help="Last fully observed citation calendar year (default: 2025).")
    parser.add_argument("--bootstrap", type=int, default=5000, help="Bootstrap replications for Spearman CI (default: 5000).")
    parser.add_argument("--permutations", type=int, default=10000, help="Permutation replications for primary Spearman test (default: 10000).")
    parser.add_argument("--seed", type=int, default=20260910, help="Random seed (default: 20260910).")
    return parser.parse_args()


def oa_id(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().rstrip("/")
    return text.rsplit("/", 1)[-1].upper() if text else ""


def parse_counts(value: Any) -> dict[int, int]:
    if isinstance(value, dict):
        raw = value
    elif value is None or (isinstance(value, float) and np.isnan(value)):
        raw = {}
    else:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "{}"}:
            raw = {}
        else:
            try:
                raw = json.loads(text)
            except json.JSONDecodeError:
                try:
                    raw = ast.literal_eval(text)
                except (ValueError, SyntaxError):
                    raw = {}
    result: dict[int, int] = {}
    if isinstance(raw, dict):
        for key, item in raw.items():
            try:
                result[int(key)] = int(item)
            except (ValueError, TypeError):
                continue
    return result


def load_neighbourhoods(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        raise FileNotFoundError(f"Step 6 cache is required: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            focal = oa_id(item.get("focal_id"))
            if focal and item.get("focal_group") == "retracted" and "error" not in item:
                records[focal] = item
    return records


def temporal_focal_path_brokerage(record: dict[str, Any], retraction_year: int) -> tuple[int, int, int, int]:
    """Return brokerage count, eligible sources, targets, and direct source-target ties."""
    targets = {oa_id(value) for value in record.get("focal_outbound_references", []) if oa_id(value)}
    sources = []
    for inbound in record.get("inbound_records", []) or []:
        year = inbound.get("publication_year")
        try:
            year = int(year)
        except (ValueError, TypeError):
            continue
        source = oa_id(inbound.get("id"))
        if source and year < retraction_year:
            refs = {oa_id(value) for value in inbound.get("referenced_works", []) if oa_id(value)}
            sources.append((source, refs))
    if not targets or not sources:
        return 0, len(sources), len(targets), 0
    direct_ties = 0
    brokered_paths = 0
    for _, source_refs in sources:
        overlaps = len(source_refs & targets)
        direct_ties += overlaps
        brokered_paths += max(0, len(targets) - overlaps)
    return int(brokered_paths), int(len(sources)), int(len(targets)), int(direct_ties)


def hc3_ols(y: np.ndarray, x: np.ndarray, terms: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    residuals = y - x @ beta
    leverage = np.clip(np.einsum("ij,jk,ik->i", x, xtx_inv, x), 0.0, 0.999999)
    scaled = residuals / (1.0 - leverage)
    meat = x.T @ (x * (scaled ** 2)[:, None])
    covariance = xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    df = max(1, len(y) - x.shape[1])
    t_values = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    p_values = 2 * stats.t.sf(np.abs(t_values), df=df)
    crit = stats.t.ppf(0.975, df=df)
    return pd.DataFrame({
        "term": terms, "coefficient": beta, "hc3_standard_error": se,
        "ci_95_lower": beta - crit * se, "ci_95_upper": beta + crit * se,
        "t_statistic": t_values, "p_value": p_values,
        "n_observations": len(y), "residual_df": df,
    }), covariance


def spearman_bootstrap(x: np.ndarray, y: np.ndarray, iterations: int, seed: int) -> tuple[float, float, float]:
    rho, _ = stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    n = len(x)
    values = np.empty(iterations, dtype=float)
    for i in range(iterations):
        idx = rng.integers(0, n, n)
        values[i] = stats.spearmanr(x[idx], y[idx]).statistic
    values = values[np.isfinite(values)]
    return float(rho), float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def spearman_permutation(x: np.ndarray, y: np.ndarray, iterations: int, seed: int) -> float:
    observed = abs(float(stats.spearmanr(x, y).statistic))
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(iterations):
        statistic = abs(float(stats.spearmanr(rng.permutation(x), y).statistic))
        exceed += statistic >= observed
    return float((exceed + 1) / (iterations + 1))


def holm(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    adjusted = [np.nan] * len(values)
    running = 0.0
    n = len(values)
    for rank, (index, value) in enumerate(indexed):
        running = max(running, min(1.0, (n - rank) * value))
        adjusted[index] = running
    return adjusted


def make_dataset(root: Path, window: int, observation_end: int) -> tuple[pd.DataFrame, dict[str, int]]:
    dirs = revision_dirs(root)
    pairs_path = dirs["output"] / "step4_matched_pairs.csv"
    cache_path = dirs["data"] / "step6_focal_neighborhoods.jsonl"
    if not pairs_path.exists():
        raise FileNotFoundError(f"Step 4 matched pairs are required: {pairs_path}")
    pairs = pd.read_csv(pairs_path, low_memory=False)
    required = {"treated_id", "pseudo_event_year", "paper_age_at_event", "treated_counts_json", "treated_baseline_log_cites", "concept"}
    missing = required - set(pairs.columns)
    if missing:
        raise ValueError("step4_matched_pairs.csv lacks: " + ", ".join(sorted(missing)))
    records = load_neighbourhoods(cache_path)
    rows: list[dict[str, Any]] = []
    counts = {"matched_pairs": int(len(pairs)), "eligible_window": 0, "cache_found": 0, "retained": 0}
    for _, pair in pairs.iterrows():
        focal = oa_id(pair["treated_id"])
        try:
            year = int(pair["pseudo_event_year"])
            age = float(pair["paper_age_at_event"])
        except (ValueError, TypeError):
            continue
        if year + window > observation_end:
            continue
        counts["eligible_window"] += 1
        record = records.get(focal)
        if not record:
            continue
        counts["cache_found"] += 1
        citation_counts = parse_counts(pair["treated_counts_json"])
        pre_cites = sum(citation_counts.get(year - offset, 0) for offset in range(1, window + 1))
        post_cites = sum(citation_counts.get(year + offset, 0) for offset in range(1, window + 1))
        brokerage, sources, targets, direct_ties = temporal_focal_path_brokerage(record, year)
        if targets == 0:
            continue
        rows.append({
            "treated_id": focal,
            "pair_id": pair.get("pair_id", ""),
            "retraction_year": year,
            "paper_age_at_retraction": age,
            "concept": str(pair.get("concept", "Unknown") or "Unknown"),
            "pre_citations_window": pre_cites,
            "post_citations_window": post_cites,
            "log_pre_citations": math.log1p(pre_cites),
            "log_post_citations": math.log1p(post_cites),
            "pre_retraction_focal_path_brokerage": brokerage,
            "log_pre_retraction_brokerage": math.log1p(brokerage),
            "eligible_pre_retraction_citing_sources": sources,
            "retained_focal_references": targets,
            "observed_direct_source_target_ties": direct_ties,
        })
    data = pd.DataFrame(rows)
    counts["retained"] = int(len(data))
    return data, counts


def analyze(data: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    y = data["log_post_citations"].to_numpy(float)
    brokerage = data["log_pre_retraction_brokerage"].to_numpy(float)
    rho, rho_low, rho_high = spearman_bootstrap(brokerage, y, args.bootstrap, args.seed)
    rho_p = spearman_permutation(brokerage, y, args.permutations, args.seed)
    x = np.column_stack([
        np.ones(len(data)),
        brokerage,
        data["log_pre_citations"].to_numpy(float),
        np.log1p(data["paper_age_at_retraction"].clip(lower=0).to_numpy(float)),
    ])
    regression, _ = hc3_ols(y, x, ["Intercept", "Pre-retraction focal-path brokerage", "Pre-retraction citations", "log(1 + paper age at retraction)"])
    brokerage_row = regression.loc[regression["term"].eq("Pre-retraction focal-path brokerage")].iloc[0]
    raw_p = [rho_p, float(brokerage_row["p_value"])]
    adjusted_p = holm(raw_p)
    inference = pd.DataFrame([
        {
            "test": "Primary: Spearman association", "outcome": "log(1 + post-retraction citations)",
            "estimate": rho, "ci_95_lower": rho_low, "ci_95_upper": rho_high,
            "statistic": np.nan, "p_value": rho_p, "holm_adjusted_p_value": adjusted_p[0],
            "n_papers": len(data), "note": f"{args.permutations:,} brokerage-label permutations; {args.bootstrap:,} bootstrap replications",
        },
        {
            "test": "Adjusted: HC3 OLS brokerage coefficient", "outcome": "log(1 + post-retraction citations)",
            "estimate": float(brokerage_row["coefficient"]), "ci_95_lower": float(brokerage_row["ci_95_lower"]),
            "ci_95_upper": float(brokerage_row["ci_95_upper"]), "statistic": float(brokerage_row["t_statistic"]),
            "p_value": float(brokerage_row["p_value"]), "holm_adjusted_p_value": adjusted_p[1],
            "n_papers": len(data), "note": "Adjusted for pre-retraction citations and paper age; HC3 standard errors",
        },
    ])
    summary = {
        "hypothesis": "Among retracted papers, higher pre-retraction local focal-path brokerage is associated with greater post-retraction citation persistence.",
        "interpretation_limit": "The analysis measures an association between sampled pre-retraction local citation-path brokerage and later citation counts. It does not identify multi-step propagation or the citation stance of later citing papers.",
        "primary_outcome": f"log(1 + citations in years r+1 through r+{args.window})",
        "brokerage_measure": "Unnormalized count of sampled shortest source-to-target paths through a focal paper, using only selected citing works published before its retraction year.",
        "window": args.window,
        "observation_end_year": args.observation_end,
        "n_papers": int(len(data)),
        "spearman_rho": rho,
        "spearman_ci_95": [rho_low, rho_high],
        "spearman_permutation_p": rho_p,
        "adjusted_brokerage_coefficient": float(brokerage_row["coefficient"]),
        "adjusted_brokerage_ci_95": [float(brokerage_row["ci_95_lower"]), float(brokerage_row["ci_95_upper"])],
        "adjusted_brokerage_p": float(brokerage_row["p_value"]),
        "holm_adjusted_p_values": {"spearman": adjusted_p[0], "adjusted_ols": adjusted_p[1]},
        "created_at": now_utc(),
    }
    return inference, regression, summary


def sensitivity(root: Path, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for window in (1, 2, 3, 4):
        data, cohort = make_dataset(root, window, args.observation_end)
        if len(data) < 30:
            rows.append({"window": window, "n_papers": len(data), "spearman_rho": np.nan, "spearman_p": np.nan,
                         "adjusted_brokerage_coefficient": np.nan, "adjusted_brokerage_p": np.nan})
            continue
        inference, _, _ = analyze(data, argparse.Namespace(**{**vars(args), "window": window, "bootstrap": 1000, "permutations": 2000}))
        rows.append({
            "window": window, "n_papers": len(data), "spearman_rho": inference.iloc[0]["estimate"],
            "spearman_p": inference.iloc[0]["p_value"],
            "adjusted_brokerage_coefficient": inference.iloc[1]["estimate"], "adjusted_brokerage_p": inference.iloc[1]["p_value"],
        })
    return pd.DataFrame(rows)


def write_latex_table(inference: pd.DataFrame, output_dir: Path, window: int) -> None:
    newline = r"\\"
    lines = [
        r"\begin{table}[!htbp]", r"\centering", r"\small",
        rf"\caption{{Time-respecting test of H2: association between pre-retraction focal-path brokerage and post-retraction citation persistence over $H={window}$ years. Holm-adjusted $p$-values are reported across the two tests.}}",
        r"\label{tab:h2_brokerage_persistence}", r"\begin{tabular}{p{0.38\linewidth}rrrr}", r"\toprule",
        r"Test & Estimate & 95\% CI & $p$ & Holm $p$ " + newline, r"\midrule",
    ]
    for _, row in inference.iterrows():
        lines.append(
            f"{latex_escape(row['test'])} & {row['estimate']:.3f} & "
            f"[{row['ci_95_lower']:.3f}, {row['ci_95_upper']:.3f}] & "
            f"{row['p_value']:.3g} & {row['holm_adjusted_p_value']:.3g} " + newline
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (output_dir / "table_step8_h2_brokerage_persistence.tex").write_text("\n".join(lines), encoding="utf-8")


def make_figure(data: pd.DataFrame, summary: dict[str, Any], output_dir: Path) -> None:
    x = data["log_pre_retraction_brokerage"].to_numpy(float)
    y = data["log_post_citations"].to_numpy(float)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    ax.scatter(x, y, alpha=0.38, s=19, color="#2166AC", edgecolors="none")
    if len(np.unique(x)) > 1:
        slope, intercept, _, _, _ = stats.linregress(x, y)
        xline = np.linspace(x.min(), x.max(), 100)
        ax.plot(xline, intercept + slope * xline, color="#B2182B", linewidth=2,
                label=f"OLS descriptive fit; Spearman $\\rho$ = {summary['spearman_rho']:.2f}")
    ax.set_xlabel("log(1 + pre-retraction focal-path brokerage)")
    ax.set_ylabel(f"log(1 + citations in years r+1 to r+{summary['window']})")
    ax.set_title("A. Paper-level association")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(linestyle=":", alpha=0.45)

    ax = axes[1]
    bins = pd.qcut(data["pre_retraction_focal_path_brokerage"].rank(method="first"), q=4, labels=["Q1", "Q2", "Q3", "Q4"])
    plotting = data.assign(brokerage_quartile=bins).groupby("brokerage_quartile", observed=False)["post_citations_window"].agg(["median", "mean", "count"]).reset_index()
    ax.bar(plotting["brokerage_quartile"].astype(str), plotting["median"], color=["#D1E5F0", "#92C5DE", "#4393C3", "#2166AC"], edgecolor="white")
    for i, row in plotting.iterrows():
        ax.text(i, row["median"] + max(0.15, 0.02 * plotting["median"].max()), f"n={int(row['count'])}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Pre-retraction brokerage quartile")
    ax.set_ylabel(f"Median citations in years r+1 to r+{summary['window']}")
    ax.set_title("B. Unadjusted persistence by brokerage quartile")
    ax.grid(axis="y", linestyle=":", alpha=0.45)
    fig.suptitle("H2: Pre-Retraction Brokerage and Post-Retraction Citation Persistence", y=1.02, fontsize=13)
    fig.tight_layout()
    save_dual_figure(fig, output_dir / "fig_step8_h2_brokerage_persistence")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = project_root()
    output_dir = revision_dirs(root)["output"]
    data, cohort = make_dataset(root, args.window, args.observation_end)
    if len(data) < 30:
        raise RuntimeError(f"Only {len(data)} papers are eligible for H2 at H={args.window}; at least 30 are required.")
    inference, regression, summary = analyze(data, args)
    sensitivity_table = sensitivity(root, args)
    summary["cohort_flow"] = cohort
    summary["outputs"] = [
        "step8_h2_summary.json", "step8_h2_paper_level.csv", "step8_h2_inference.csv",
        "step8_h2_adjusted_regression.csv", "step8_h2_window_sensitivity.csv",
        "table_step8_h2_brokerage_persistence.tex", "fig_step8_h2_brokerage_persistence.png/.pdf",
    ]
    data.to_csv(output_dir / "step8_h2_paper_level.csv", index=False)
    inference.to_csv(output_dir / "step8_h2_inference.csv", index=False)
    regression.to_csv(output_dir / "step8_h2_adjusted_regression.csv", index=False)
    sensitivity_table.to_csv(output_dir / "step8_h2_window_sensitivity.csv", index=False)
    write_latex_table(inference, output_dir, args.window)
    make_figure(data, summary, output_dir)
    write_json(output_dir / "step8_h2_summary.json", summary)
    print(f"Step 8 complete: {len(data):,} retracted papers tested at H={args.window}. Inspect the summary and inference table before interpreting H2.")


if __name__ == "__main__":
    main()
