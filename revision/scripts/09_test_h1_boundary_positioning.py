#!/usr/bin/env python3
"""Step 7: adjusted matched-pair test of H1 (boundary positioning).

H1: Retracted papers are more likely to occupy boundary-spanning positions in
the sampled citation network than comparable non-retracted papers.

This script consumes the balanced 462-pair design from Step 4 and focal-path
brokerage measures computed in Step 6. Its primary outcome is log(1 + the
unnormalized focal-path brokerage count). The primary inferential analysis is
paired: retracted minus matched-control within each pair. The adjusted analysis
uses individual-paper regression with matched-pair fixed effects and the
individual pre-event citation baseline. Journal, publication year, article type,
paper age at the pseudo-event, and the journal's disciplinary context are
absorbed by the exact pair matching and pair fixed effects.

No API calls are made. Run from the project root:
    python revision/scripts/09_test_h1_boundary_positioning.py
"""
from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description="Step 7: adjusted matched-pair H1 boundary-positioning analysis.")
    parser.add_argument("--bootstrap", type=int, default=5000, help="Matched-pair bootstrap replications (default: 5000).")
    parser.add_argument("--permutations", type=int, default=10000, help="Paired sign-flip permutations for mean contrast (default: 10000).")
    parser.add_argument("--seed", type=int, default=20260909, help="Random seed (default: 20260909).")
    return parser.parse_args()


def canonical_id(value: Any) -> str:
    """Normalize either a full OpenAlex URL or a bare work ID to uppercase W-ID."""
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().rstrip("/")
    return text.rsplit("/", 1)[-1].upper() if text else ""


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    finite = [(name, p) for name, p in p_values.items() if np.isfinite(p)]
    finite.sort(key=lambda item: item[1])
    m = len(finite)
    result: dict[str, float] = {name: np.nan for name in p_values}
    running = 0.0
    for rank, (name, p) in enumerate(finite):
        adjusted = min(1.0, (m - rank) * p)
        running = max(running, adjusted)
        result[name] = running
    return result


def bootstrap_mean_ci(values: np.ndarray, iterations: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    if n < 2:
        return np.nan, np.nan
    sample_indices = rng.integers(0, n, size=(iterations, n))
    means = values[sample_indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def sign_flip_permutation(values: np.ndarray, iterations: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    observed = abs(float(values.mean()))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(iterations, len(values)))
    null = np.abs((signs * values).mean(axis=1))
    return float((np.sum(null >= observed) + 1) / (iterations + 1))


def mcnemar_exact(b: int, c: int) -> tuple[float, float]:
    """Return odds ratio (b/c) and exact two-sided binomial p-value."""
    discordant = b + c
    if discordant == 0:
        return np.nan, 1.0
    odds_ratio = (b + 0.5) / (c + 0.5)  # Haldane--Anscombe correction
    p_value = float(2 * stats.binom.cdf(min(b, c), discordant, 0.5))
    return odds_ratio, min(1.0, p_value)


def cluster_ols(y: np.ndarray, x: np.ndarray, clusters: np.ndarray, names: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    """OLS with matched-pair cluster-robust standard errors."""
    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    residuals = y - x @ beta
    unique_clusters = pd.unique(clusters)
    meat = np.zeros((x.shape[1], x.shape[1]), dtype=float)
    for cluster in unique_clusters:
        mask = clusters == cluster
        score = x[mask].T @ residuals[mask]
        meat += np.outer(score, score)
    g = len(unique_clusters)
    n, k = x.shape
    correction = (g / max(g - 1, 1)) * ((n - 1) / max(n - k, 1))
    covariance = correction * xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    df = max(1, g - 1)
    t_values = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    p_values = 2 * stats.t.sf(np.abs(t_values), df=df)
    crit = stats.t.ppf(0.975, df=df)
    result = pd.DataFrame({
        "term": names,
        "coefficient": beta,
        "clustered_standard_error": se,
        "ci_95_lower": beta - crit * se,
        "ci_95_upper": beta + crit * se,
        "t_statistic": t_values,
        "p_value": p_values,
        "n_observations": n,
        "n_pair_clusters": g,
        "residual_df": df,
    })
    return result, covariance


def load_control_topics(cache_path: Path) -> dict[str, str]:
    topics: dict[str, str] = {}
    if not cache_path.exists():
        return topics
    with cache_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            work_id = canonical_id(record.get("id"))
            if not work_id:
                continue
            topic = ((record.get("primary_topic") or {}).get("display_name") or "Unknown").strip()
            topics[work_id] = topic or "Unknown"
    return topics


def make_pair_data(output_dir: Path, data_dir: Path) -> pd.DataFrame:
    pairs_path = output_dir / "step4_matched_pairs.csv"
    metrics_path = output_dir / "step6_focal_node_metrics.csv"
    cache_path = data_dir / "step4_nonretracted_control_cache_v2.jsonl"
    if not pairs_path.exists() or not metrics_path.exists():
        raise FileNotFoundError("Step 7 requires Step 4 matched pairs and Step 6 focal-node metrics in revision/output/.")
    pairs = pd.read_csv(pairs_path)
    pairs["treated_id"] = pairs["treated_id"].map(canonical_id)
    pairs["control_id"] = pairs["control_id"].map(canonical_id)
    metrics = pd.read_csv(metrics_path)
    needed_pairs = {"pair_id", "treated_id", "control_id", "publication_year", "paper_age_at_event", "concept", "treated_baseline_log_cites", "control_baseline_log_cites"}
    missing = needed_pairs - set(pairs.columns)
    if missing:
        raise ValueError("step4_matched_pairs.csv lacks: " + ", ".join(sorted(missing)))
    if "directed_betweenness" not in metrics.columns:
        raise ValueError("step6_focal_node_metrics.csv lacks directed_betweenness.")
    topics = load_control_topics(cache_path)
    metric_cols = ["openalex_id", "pair_id", "node_group", "directed_betweenness", "in_degree", "out_degree", "pagerank"]
    focal = metrics.loc[metrics["is_focal"].astype(bool), metric_cols].copy()
    focal["openalex_id"] = focal["openalex_id"].map(canonical_id)
    t = focal.loc[focal["node_group"].eq("retracted")].copy()
    c = focal.loc[focal["node_group"].eq("matched_control")].copy()
    t = t.rename(columns={
        "openalex_id": "treated_id", "directed_betweenness": "treated_brokerage",
        "in_degree": "treated_in_degree", "out_degree": "treated_out_degree", "pagerank": "treated_pagerank",
    }).drop(columns=["node_group"])
    c = c.rename(columns={
        "openalex_id": "control_id", "directed_betweenness": "control_brokerage",
        "in_degree": "control_in_degree", "out_degree": "control_out_degree", "pagerank": "control_pagerank",
    }).drop(columns=["node_group"])
    data = pairs.merge(t, on=["pair_id", "treated_id"], how="inner").merge(c, on=["pair_id", "control_id"], how="inner")
    data["treated_concept"] = data["concept"].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
    data["control_concept"] = data["control_id"].map(topics).fillna("Unknown")
    data["discipline_mismatch"] = (data["treated_concept"] != data["control_concept"]).astype(int)
    for group in ("treated", "control"):
        data[f"log_{group}_brokerage"] = np.log1p(pd.to_numeric(data[f"{group}_brokerage"], errors="coerce").clip(lower=0))
    data["log_brokerage_difference"] = data["log_treated_brokerage"] - data["log_control_brokerage"]
    data["baseline_citation_difference"] = (
        pd.to_numeric(data["treated_baseline_log_cites"], errors="coerce") -
        pd.to_numeric(data["control_baseline_log_cites"], errors="coerce")
    )
    data["treated_any_brokerage"] = (pd.to_numeric(data["treated_brokerage"], errors="coerce").fillna(0) > 0).astype(int)
    data["control_any_brokerage"] = (pd.to_numeric(data["control_brokerage"], errors="coerce").fillna(0) > 0).astype(int)
    return data.replace([np.inf, -np.inf], np.nan).dropna(subset=["log_brokerage_difference", "baseline_citation_difference"]).copy()


def adjusted_model(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], dict[str, float]]:
    """Estimate an individual-paper model with matched-pair fixed effects.

    Pair effects absorb article type, journal, publication year, pseudo-event
    age, and journal-level disciplinary context. The individual pre-event
    citation baseline remains estimable within pair.
    """
    rows: list[dict[str, Any]] = []
    for _, row in data.iterrows():
        rows.extend([
            {
                "pair_id": row["pair_id"], "treatment": 1.0,
                "log_brokerage": row["log_treated_brokerage"],
                "baseline_cites": row["treated_baseline_log_cites"],
            },
            {
                "pair_id": row["pair_id"], "treatment": 0.0,
                "log_brokerage": row["log_control_brokerage"],
                "baseline_cites": row["control_baseline_log_cites"],
            },
        ])
    long = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).dropna()
    pair_dummies = pd.get_dummies(long["pair_id"], prefix="pair", drop_first=True, dtype=float)
    x = pd.concat([
        pd.Series(1.0, index=long.index, name="Intercept"),
        long["treatment"].rename("Retracted paper"),
        long["baseline_cites"].rename("Individual pre-event baseline citations"),
        pair_dummies,
    ], axis=1)
    result, covariance = cluster_ols(
        long["log_brokerage"].to_numpy(float), x.to_numpy(float), long["pair_id"].to_numpy(), x.columns.tolist()
    )
    treatment_index = x.columns.get_loc("Retracted paper")
    weight = np.zeros(x.shape[1], dtype=float)
    weight[treatment_index] = 1.0
    average_effect = float(weight @ result["coefficient"].to_numpy(float))
    average_se = float(np.sqrt(max(0.0, weight @ covariance @ weight)))
    df = max(1, long["pair_id"].nunique() - 1)
    t_value = average_effect / average_se if average_se > 0 else np.nan
    p_value = float(2 * stats.t.sf(abs(t_value), df=df)) if np.isfinite(t_value) else np.nan
    crit = stats.t.ppf(0.975, df=df)
    effect = {
        "average_adjusted_treatment_effect": average_effect,
        "clustered_standard_error": average_se,
        "ci_95_lower": average_effect - crit * average_se,
        "ci_95_upper": average_effect + crit * average_se,
        "t_statistic": t_value,
        "p_value": p_value,
    }
    meta = {
        "n_pairs": int(long["pair_id"].nunique()),
        "n_observations": int(len(long)),
        "pair_fixed_effects": int(pair_dummies.shape[1]),
        "exact_matching_controls": ["article type", "journal ISSN-L", "publication year", "paper age at pseudo-event"],
        "within_pair_covariates": ["individual pre-event baseline citation level"],
        "inference": "standard errors clustered by matched pair",
    }
    return result, meta, effect


def make_inference(data: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    diff = data["log_brokerage_difference"].to_numpy(float)
    mean_diff = float(diff.mean())
    ci_low, ci_high = bootstrap_mean_ci(diff, args.bootstrap, args.seed)
    permutation_p = sign_flip_permutation(diff, args.permutations, args.seed)
    nonzero = diff[diff != 0]
    if len(nonzero):
        wilcoxon = stats.wilcoxon(diff, zero_method="pratt", alternative="two-sided", method="auto")
        rank_biserial = float((np.sum(nonzero > 0) - np.sum(nonzero < 0)) / len(nonzero))
        wilcoxon_stat, wilcoxon_p = float(wilcoxon.statistic), float(wilcoxon.pvalue)
    else:
        wilcoxon_stat, wilcoxon_p, rank_biserial = np.nan, 1.0, 0.0
    b = int(((data["treated_any_brokerage"] == 1) & (data["control_any_brokerage"] == 0)).sum())
    c = int(((data["treated_any_brokerage"] == 0) & (data["control_any_brokerage"] == 1)).sum())
    odds_ratio, mcnemar_p = mcnemar_exact(b, c)
    adjusted, model_meta, adjusted_effect = adjusted_model(data)
    raw_p = {
        "Primary paired mean difference (sign-flip permutation)": permutation_p,
        "Secondary paired rank test (Wilcoxon)": wilcoxon_p,
        "Secondary brokerage-presence test (exact McNemar)": mcnemar_p,
        "Adjusted treatment effect (pair-fixed-effects OLS)": float(adjusted_effect["p_value"]),
    }
    adjusted_p = holm_adjust(raw_p)
    rows = [
        {
            "test": "Primary: paired mean difference in log(1 + focal-path brokerage)",
            "outcome": "log(1 + focal-path brokerage), retracted minus control",
            "estimate": mean_diff, "ci_95_lower": ci_low, "ci_95_upper": ci_high,
            "statistic": np.nan, "effect_size": mean_diff, "p_value": permutation_p,
            "holm_adjusted_p_value": adjusted_p["Primary paired mean difference (sign-flip permutation)"],
            "n_pairs": len(data), "note": f"{args.permutations:,} paired sign-flip permutations; {args.bootstrap:,} pair bootstrap draws",
        },
        {
            "test": "Secondary: paired rank comparison", "outcome": "log(1 + focal-path brokerage)",
            "estimate": np.nan, "ci_95_lower": np.nan, "ci_95_upper": np.nan,
            "statistic": wilcoxon_stat, "effect_size": rank_biserial, "p_value": wilcoxon_p,
            "holm_adjusted_p_value": adjusted_p["Secondary paired rank test (Wilcoxon)"],
            "n_pairs": len(data), "note": "Wilcoxon signed-rank, Pratt zero handling; effect size is matched-pair rank-biserial correlation",
        },
        {
            "test": "Secondary: any non-zero focal-path brokerage", "outcome": "1(brokerage > 0)",
            "estimate": odds_ratio, "ci_95_lower": np.nan, "ci_95_upper": np.nan,
            "statistic": b + c, "effect_size": odds_ratio, "p_value": mcnemar_p,
            "holm_adjusted_p_value": adjusted_p["Secondary brokerage-presence test (exact McNemar)"],
            "n_pairs": len(data), "note": f"Exact McNemar; discordant pairs: retracted-only={b}, control-only={c}; Haldane--Anscombe corrected odds ratio",
        },
        {
            "test": "Adjusted: retracted-paper effect with pair fixed effects", "outcome": "log(1 + focal-path brokerage)",
            "estimate": float(adjusted_effect["average_adjusted_treatment_effect"]),
            "ci_95_lower": float(adjusted_effect["ci_95_lower"]),
            "ci_95_upper": float(adjusted_effect["ci_95_upper"]),
            "statistic": float(adjusted_effect["t_statistic"]), "effect_size": float(adjusted_effect["average_adjusted_treatment_effect"]),
            "p_value": float(adjusted_effect["p_value"]),
            "holm_adjusted_p_value": adjusted_p["Adjusted treatment effect (pair-fixed-effects OLS)"],
            "n_pairs": int(model_meta["n_pairs"]),
            "note": "Pair fixed effects; pair-clustered standard errors; adjusted for individual pre-event citations. Pair effects absorb journal-level disciplinary context.",
        },
    ]
    summary = {
        "primary_outcome": "log(1 + unnormalized focal-path brokerage count)",
        "n_matched_pairs": int(len(data)),
        "exact_matching": model_meta["exact_matching_controls"],
        "adjusted_model": model_meta,
        "any_brokerage_pair_counts": {
            "both_zero": int(((data["treated_any_brokerage"] == 0) & (data["control_any_brokerage"] == 0)).sum()),
            "both_nonzero": int(((data["treated_any_brokerage"] == 1) & (data["control_any_brokerage"] == 1)).sum()),
            "retracted_only": b, "control_only": c,
        },
        "multiple_testing": "Holm adjustment is reported across the primary, secondary, and adjusted H1 comparisons; the primary inference remains the paired mean difference.",
    }
    return pd.DataFrame(rows), summary, adjusted


def write_latex_table(inference: pd.DataFrame, output_dir: Path) -> None:
    newline = r"\\"
    rows = [
        r"\begin{table}[!htbp]", r"\centering", r"\scriptsize",
        r"\caption{Matched and adjusted tests of H1 (boundary positioning). The primary outcome is $\log(1+\text{focal-path brokerage})$, where focal-path brokerage is the unnormalized count of sampled shortest directed citation paths mediated by the focal paper. Holm-adjusted $p$-values are reported across the four listed tests.}",
        r"\label{tab:h1_boundary}", r"\begin{tabular}{p{0.34\linewidth}rrrr}", r"\toprule",
        r"Test & Estimate & 95\% CI & $p$ & Holm $p$ " + newline, r"\midrule",
    ]
    for _, row in inference.iterrows():
        estimate = "" if pd.isna(row["estimate"]) else f"{row['estimate']:.3f}"
        ci = "" if pd.isna(row["ci_95_lower"]) else f"[{row['ci_95_lower']:.3f}, {row['ci_95_upper']:.3f}]"
        p = "" if pd.isna(row["p_value"]) else f"{row['p_value']:.3g}"
        holm = "" if pd.isna(row["holm_adjusted_p_value"]) else f"{row['holm_adjusted_p_value']:.3g}"
        rows.append(f"{latex_escape(row['test'])} & {estimate} & {ci} & {p} & {holm} " + newline)
    rows += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (output_dir / "table_step7_h1_boundary.tex").write_text("\n".join(rows), encoding="utf-8")


def make_figure(data: pd.DataFrame, inference: pd.DataFrame, output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    diff = data["log_brokerage_difference"].to_numpy(float)
    primary = inference.iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    displayed = data.iloc[rng.choice(len(data), size=min(130, len(data)), replace=False)].copy()
    for _, row in displayed.iterrows():
        ax.plot([0, 1], [row["log_control_brokerage"], row["log_treated_brokerage"]], color="#BDBDBD", alpha=0.32, linewidth=0.7, zorder=1)
    jitter = rng.normal(0, 0.035, len(data))
    ax.scatter(np.zeros(len(data)) + jitter, data["log_control_brokerage"], color="#377EB8", alpha=0.55, s=15, label="Matched controls", zorder=2)
    ax.scatter(np.ones(len(data)) + jitter, data["log_treated_brokerage"], color="#E41A1C", alpha=0.55, s=15, label="Retracted papers", zorder=2)
    for xpos, column, color in [(0, "log_control_brokerage", "#084594"), (1, "log_treated_brokerage", "#A50F15")]:
        mean = float(data[column].mean())
        low, high = bootstrap_mean_ci(data[column].to_numpy(float), 3000, seed + int(xpos))
        ax.errorbar(xpos, mean, yerr=[[mean-low], [high-mean]], color=color, fmt="D", markersize=6, capsize=4, zorder=4)
    ax.set_xlim(-0.35, 1.35)
    ax.set_xticks([0, 1], ["Matched\ncontrols", "Retracted\npapers"])
    ax.set_ylabel("log(1 + focal-path brokerage)")
    ax.set_title("A. Matched focal-paper comparison")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    ax = axes[1]
    bins = np.linspace(min(-1.0, diff.min()), max(1.0, diff.max()), 34)
    ax.hist(diff, bins=bins, color="#9ECAE1", edgecolor="white")
    ax.axvline(0, color="#333333", linewidth=1.1)
    ax.axvline(primary["estimate"], color="#B2182B", linewidth=2.0, label=f"Mean = {primary['estimate']:.3f}")
    ax.set_xlabel("Retracted − control: log(1 + brokerage)")
    ax.set_ylabel("Matched pairs")
    ax.set_title("B. Within-pair differences")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.suptitle("H1: Boundary Positioning in the Balanced Citation-Network Sample", y=1.02, fontsize=13)
    fig.tight_layout()
    save_dual_figure(fig, output_dir / "fig_step7_h1_boundary_positioning")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = project_root()
    dirs = revision_dirs(root)
    output_dir, data_dir = dirs["output"], dirs["data"]
    data = make_pair_data(output_dir, data_dir)
    if len(data) < 30:
        raise RuntimeError("Too few complete matched pairs for Step 7 inference.")
    inference, summary, adjusted = make_inference(data, args)
    data.to_csv(output_dir / "step7_h1_pair_level.csv", index=False)
    inference.to_csv(output_dir / "step7_h1_inference.csv", index=False)
    adjusted.to_csv(output_dir / "step7_h1_adjusted_regression.csv", index=False)
    write_latex_table(inference, output_dir)
    make_figure(data, inference, output_dir, args.seed)
    summary.update({
        "created_at": now_utc(),
        "bootstrap_replications": args.bootstrap,
        "sign_flip_permutations": args.permutations,
        "outputs": [
            "step7_h1_pair_level.csv", "step7_h1_inference.csv", "step7_h1_adjusted_regression.csv",
            "table_step7_h1_boundary.tex", "fig_step7_h1_boundary_positioning.png/.pdf",
        ],
    })
    write_json(output_dir / "step7_h1_summary.json", summary)
    print(f"Step 7 complete: {len(data):,} matched focal pairs tested. Inspect step7_h1_summary.json and inference table before interpreting H1.")


if __name__ == "__main__":
    main()
