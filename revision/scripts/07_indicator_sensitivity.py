#!/usr/bin/env python3
"""Step 5: sensitivity analysis for the post-retraction citation-persistence indicators.

This script replaces the earlier one-shot indicator calculation with a transparent
set of common-follow-up analyses. It evaluates how Exposure, Contamination,
Persistence, and Recovery change when the following analytical choices vary:

1. Common symmetric event window H in {1, 2, 3, 4};
2. Minimum papers per scholarly unit in {10, 30, 50}; and
3. Minimum annual citation denominator used to calculate Q_U(tau) in {1, 10}.

The source is the frozen Step 1 master corpus. A paper is eligible for a given H
only when all years r_p-H through r_p+H are in OpenAlex's common annual citation
history. Therefore, papers retracted too recently never contribute to a longer
persistence window. All reported indicators are window-specific and should be
labelled with their H value in the manuscript.

Definitions for a unit U and tau = 1,...,H:
  A_U(tau) = sum_{p in U} C_{p, r_p+tau}
  B_U(tau) = sum_{p in U} C_{p, r_p-tau}
  Q_U(tau) = A_U(tau) / [A_U(tau) + B_U(tau)]
  Exposure_H(U) = sum_tau A_U(tau) / sum_tau [A_U(tau) + B_U(tau)]
  Contamination_H(U) = mean_tau Q_U(tau), for eligible denominators
  Persistence_H(U) = max{tau: A_U(tau) > 0}; zero if no post-event citation
  Recovery_H(U) = OLS slope of Q_U(tau) on tau

Run from the project root after Step 1:
    python revision/scripts/run_step_05.py

No network/API calls are made. Outputs are written under revision/output/.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    find_column,
    latex_escape,
    now_utc,
    project_root,
    read_master,
    revision_dirs,
    save_dual_figure,
    write_json,
)

DEFAULT_WINDOWS = (1, 2, 3, 4)
DEFAULT_MIN_UNIT_PAPERS = (10, 30, 50)
DEFAULT_MIN_DENOMINATORS = (1, 10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run common-follow-up sensitivity analysis for citation-persistence indicators."
    )
    parser.add_argument("--windows", nargs="+", type=int, default=list(DEFAULT_WINDOWS),
                        help="Positive symmetric follow-up half-windows H (default: 1 2 3 4).")
    parser.add_argument("--min-unit-papers", nargs="+", type=int,
                        default=list(DEFAULT_MIN_UNIT_PAPERS),
                        help="Minimum papers per scholarly unit (default: 10 30 50).")
    parser.add_argument("--min-denominators", nargs="+", type=int,
                        default=list(DEFAULT_MIN_DENOMINATORS),
                        help="Minimum A_U(tau)+B_U(tau) for Q_U(tau) (default: 1 10).")
    parser.add_argument("--as-of-year", type=int, default=date.today().year - 1,
                        help="Last complete citation year (default: previous calendar year).")
    parser.add_argument("--citation-history-start-year", type=int, default=None,
                        help="First complete year in counts_by_year; default: as-of year minus 9.")
    parser.add_argument("--baseline-window", type=int, default=4,
                        help="Window H used as the stability-comparison baseline (default: 4).")
    parser.add_argument("--baseline-min-unit-papers", type=int, default=30,
                        help="Minimum unit size for the stability-comparison baseline (default: 30).")
    parser.add_argument("--baseline-min-denominator", type=int, default=10,
                        help="Minimum denominator for the stability-comparison baseline (default: 10).")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Top-k threshold for ranking-overlap sensitivity (default: 20).")
    return parser.parse_args()


def history_start_year(args: argparse.Namespace) -> int:
    return int(args.citation_history_start_year) if args.citation_history_start_year is not None else int(args.as_of_year - 9)


def parse_counts(value: Any) -> dict[int, int]:
    if isinstance(value, (dict, list)):
        raw = value
    else:
        try:
            raw = json.loads(value) if isinstance(value, str) else []
        except (TypeError, ValueError, json.JSONDecodeError):
            raw = []
    output: dict[int, int] = {}
    if isinstance(raw, dict):
        iterator = raw.items()
        for year, count in iterator:
            try:
                output[int(year)] = int(count)
            except (TypeError, ValueError):
                continue
    elif isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                output[int(item["year"])] = int(item["cited_by_count"])
            except (KeyError, TypeError, ValueError):
                continue
    return output


def infer_columns(df: pd.DataFrame) -> dict[str, str]:
    out = {
        "paper_id": find_column(df.columns, ["revision_openalex_id", "openalex_id", "id", "doi_normalised"]),
        "retraction_year": find_column(df.columns, ["retraction_year", "revision_retraction_year"]),
        "retraction_date": find_column(df.columns, ["RetractionDate", "revision_retraction_date"]),
        "counts": find_column(df.columns, ["counts_by_year_json", "counts_by_year"]),
        "journal": find_column(df.columns, ["journal_name", "Journal"]),
        "publisher": find_column(df.columns, ["Publisher", "publisher"]),
        "country": find_column(df.columns, ["Country", "country"]),
        "discipline": find_column(df.columns, ["top_concept", "concepts_top", "primary_topic"]),
    }
    for key in ("paper_id", "retraction_year", "counts"):
        if not out[key]:
            raise ValueError(f"Frozen master corpus lacks the required {key} column.")
    return out  # type: ignore[return-value]


def prepare_data(master: pd.DataFrame, cols: dict[str, str]) -> pd.DataFrame:
    df = master.copy()
    df["paper_id"] = df[cols["paper_id"]].fillna("").astype(str).str.strip()
    df["ret_year"] = pd.to_numeric(df[cols["retraction_year"]], errors="coerce")
    if df["ret_year"].isna().all() and cols.get("retraction_date"):
        df["ret_year"] = pd.to_datetime(df[cols["retraction_date"]], errors="coerce").dt.year
    df["citation_history"] = df[cols["counts"]].apply(parse_counts)
    for unit, source_col in (("journal", cols.get("journal")),
                             ("publisher", cols.get("publisher")),
                             ("country", cols.get("country")),
                             ("discipline", cols.get("discipline"))):
        if source_col:
            df[unit] = df[source_col].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
        else:
            df[unit] = "Unknown"
    df = df[df["paper_id"].ne("") & df["ret_year"].notna()].copy()
    df["ret_year"] = df["ret_year"].astype(int)
    return df


def eligible_for_window(df: pd.DataFrame, h: int, args: argparse.Namespace) -> pd.DataFrame:
    start = history_start_year(args)
    eligible = df[
        (df["ret_year"] - h >= start) &
        (df["ret_year"] + h <= args.as_of_year)
    ].copy()
    return eligible


def unit_event_totals(sub: pd.DataFrame, h: int) -> tuple[np.ndarray, np.ndarray]:
    post = np.zeros(h, dtype=float)
    pre = np.zeros(h, dtype=float)
    # Each row retains its paper-specific retraction year for event-time alignment.
    for _, row in sub[["ret_year", "citation_history"]].iterrows():
        ret_year = int(row["ret_year"])
        counts = row["citation_history"]
        for tau in range(1, h + 1):
            post[tau - 1] += max(0, int(counts.get(ret_year + tau, 0)))
            pre[tau - 1] += max(0, int(counts.get(ret_year - tau, 0)))
    return post, pre


def calculate_indicators(sub: pd.DataFrame, h: int, min_denominator: int) -> dict[str, Any]:
    post, pre = unit_event_totals(sub, h)
    denominator = post + pre
    valid = denominator >= min_denominator
    ratios = np.divide(post, denominator, out=np.full(h, np.nan, dtype=float), where=denominator > 0)
    valid_ratios = ratios[valid & np.isfinite(ratios)]
    exposure_denom = denominator.sum()
    exposure = float(post.sum() / exposure_denom) if exposure_denom > 0 else np.nan
    contamination = float(valid_ratios.mean()) if len(valid_ratios) else np.nan
    persistence = int(np.max(np.where(post > 0)[0]) + 1) if np.any(post > 0) else 0
    recovery = float(np.polyfit(np.arange(1, h + 1)[valid & np.isfinite(ratios)], valid_ratios, 1)[0]) if len(valid_ratios) >= 2 else np.nan
    return {
        "exposure": exposure,
        "contamination": contamination,
        "persistence_years": persistence,
        "recovery_slope": recovery,
        "total_post_citations": int(post.sum()),
        "total_pre_citations": int(pre.sum()),
        "eligible_event_times": int(valid.sum()),
    }


def compute_specification(
    eligible: pd.DataFrame,
    h: int,
    min_unit_papers: int,
    min_denominator: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    unit_columns = {"journal": "journal", "publisher": "publisher", "country": "country", "discipline": "discipline"}
    for unit_type, col in unit_columns.items():
        group_sizes = eligible.groupby(col, dropna=False).size()
        valid_units = group_sizes[group_sizes >= min_unit_papers].index
        for unit in valid_units:
            sub = eligible[eligible[col].eq(unit)]
            values = calculate_indicators(sub, h, min_denominator)
            rows.append({
                "unit_type": unit_type,
                "unit": str(unit),
                "followup_window_h": h,
                "min_unit_papers": min_unit_papers,
                "min_event_denominator": min_denominator,
                "n_papers": int(len(sub)),
                **values,
            })
    return pd.DataFrame(rows)


def global_summary(eligible: pd.DataFrame, h: int, min_denominator: int) -> dict[str, Any]:
    values = calculate_indicators(eligible, h, min_denominator)
    return {"followup_window_h": h, "min_event_denominator": min_denominator, "n_eligible_papers": int(len(eligible)), **values}


def top_k_set(df: pd.DataFrame, metric: str, k: int) -> set[str]:
    use = df.dropna(subset=[metric]).sort_values(metric, ascending=False).head(k)
    return set(use["unit"].astype(str))


def compute_stability(estimates: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    base = estimates[
        (estimates["followup_window_h"] == args.baseline_window) &
        (estimates["min_unit_papers"] == args.baseline_min_unit_papers) &
        (estimates["min_event_denominator"] == args.baseline_min_denominator)
    ].copy()
    rows: list[dict[str, Any]] = []
    metrics = ("exposure", "contamination", "persistence_years", "recovery_slope")
    specifications = estimates[["followup_window_h", "min_unit_papers", "min_event_denominator"]].drop_duplicates()
    for _, spec in specifications.iterrows():
        alt = estimates[
            (estimates["followup_window_h"] == spec["followup_window_h"]) &
            (estimates["min_unit_papers"] == spec["min_unit_papers"]) &
            (estimates["min_event_denominator"] == spec["min_event_denominator"])
        ]
        for unit_type in sorted(estimates["unit_type"].unique()):
            b = base[base["unit_type"].eq(unit_type)]
            a = alt[alt["unit_type"].eq(unit_type)]
            for metric in metrics:
                joined = b[["unit", metric]].merge(a[["unit", metric]], on="unit", suffixes=("_base", "_alt")).dropna()
                if (
                    len(joined) >= 3
                    and joined[f"{metric}_base"].nunique() > 1
                    and joined[f"{metric}_alt"].nunique() > 1
                ):
                    spearman_rho, spearman_p = stats.spearmanr(joined[f"{metric}_base"], joined[f"{metric}_alt"])
                else:
                    spearman_rho = spearman_p = np.nan
                base_top = top_k_set(b, metric, args.top_k)
                alt_top = top_k_set(a, metric, args.top_k)
                union = base_top | alt_top
                jaccard = len(base_top & alt_top) / len(union) if union else np.nan
                rows.append({
                    "unit_type": unit_type,
                    "metric": metric,
                    "baseline_window_h": args.baseline_window,
                    "alternative_window_h": int(spec["followup_window_h"]),
                    "alternative_min_unit_papers": int(spec["min_unit_papers"]),
                    "alternative_min_event_denominator": int(spec["min_event_denominator"]),
                    "n_common_units": int(len(joined)),
                    "spearman_rho": spearman_rho,
                    "spearman_p_value": spearman_p,
                    "top_k": args.top_k,
                    "top_k_jaccard": jaccard,
                })
    return pd.DataFrame(rows)


def make_summary_table(globals_df: pd.DataFrame, output_dir: Path, args: argparse.Namespace) -> None:
    newline = r"\\"
    rows = [
        r"\begin{table}[!htbp]",
        r"\centering",
        rf"\caption{{Sensitivity of corpus-level post-retraction citation-persistence indicators to the common symmetric follow-up window $H$. All papers included in a row have complete citation histories from $r_p-H$ through $r_p+H$. Contamination and Recovery use an event-time denominator threshold of {args.baseline_min_denominator} citations.}}",
        r"\label{tab:indicator_sensitivity}",
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        "$H$ & Eligible papers & Exposure & Contamination & Persistence & Recovery " + newline,
        r"\midrule",
    ]
    for _, r in globals_df[globals_df["min_event_denominator"].eq(args.baseline_min_denominator)].sort_values("followup_window_h").iterrows():
        recovery = "" if pd.isna(r["recovery_slope"]) else f"{r['recovery_slope']:.3f}"
        rows.append(
            f"{int(r['followup_window_h'])} & {int(r['n_eligible_papers']):,} & "
            f"{r['exposure']:.3f} & {r['contamination']:.3f} & "
            f"{int(r['persistence_years'])} & {recovery} " + newline
        )
    rows += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (output_dir / "table_step5_indicator_sensitivity.tex").write_text("\n".join(rows), encoding="utf-8")


def make_stability_table(stability: pd.DataFrame, output_dir: Path, args: argparse.Namespace) -> None:
    # Concise table: H-only changes at the selected baseline unit and denominator threshold,
    # pooled as the median across scholarly-unit types.
    subset = stability[
        (stability["alternative_min_unit_papers"] == args.baseline_min_unit_papers) &
        (stability["alternative_min_event_denominator"] == args.baseline_min_denominator)
    ].copy()
    med = (subset.groupby(["metric", "alternative_window_h"], as_index=False)
           .agg(median_spearman_rho=("spearman_rho", "median"),
                median_top20_jaccard=("top_k_jaccard", "median")))
    newline = r"\\"
    rows = [
        r"\begin{table}[!htbp]",
        r"\centering",
        rf"\caption{{Median ranking stability across journals, publishers, countries, and disciplines when the follow-up window varies. Values compare each specification with the $H={args.baseline_window}$, minimum-unit-size {args.baseline_min_unit_papers}, minimum-denominator {args.baseline_min_denominator} baseline.}}",
        r"\label{tab:indicator_stability}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        "Indicator & $H$ & Median Spearman $\\rho$ & Median top-20 Jaccard " + newline,
        r"\midrule",
    ]
    for _, r in med.sort_values(["metric", "alternative_window_h"]).iterrows():
        metric = latex_escape(str(r["metric"]).replace("_", " ").title())
        rho = "" if pd.isna(r["median_spearman_rho"]) else f"{r['median_spearman_rho']:.3f}"
        jac = "" if pd.isna(r["median_top20_jaccard"]) else f"{r['median_top20_jaccard']:.3f}"
        rows.append(f"{metric} & {int(r['alternative_window_h'])} & {rho} & {jac} " + newline)
    rows += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (output_dir / "table_step5_indicator_stability.tex").write_text("\n".join(rows), encoding="utf-8")


def plot_sensitivity(globals_df: pd.DataFrame, stability: pd.DataFrame, output_dir: Path, args: argparse.Namespace) -> None:
    base = globals_df[globals_df["min_event_denominator"].eq(args.baseline_min_denominator)].sort_values("followup_window_h")
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
    panels = [
        ("exposure", "Exposure$_H$", "#2166AC"),
        ("contamination", "Contamination$_H$", "#B2182B"),
        ("persistence_years", "Persistence$_H$ (years)", "#4D9221"),
        ("recovery_slope", "Recovery$_H$ slope", "#762A83"),
    ]
    for ax, (metric, label, color) in zip(axes.ravel(), panels):
        ax.plot(base["followup_window_h"], base[metric], marker="o", linewidth=2.4, color=color)
        ax.set_xlabel("Common follow-up half-window, $H$ (years)")
        ax.set_ylabel(label)
        ax.grid(axis="y", linestyle=":", alpha=0.55)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Sensitivity of Corpus-Level Citation-Persistence Indicators to Common Follow-up Windows", y=0.99, fontsize=13)
    fig.tight_layout()
    save_dual_figure(fig, output_dir / "fig_step5_indicator_window_sensitivity")
    plt.close(fig)

    # Heatmap of median rank correlations across unit types as H varies.
    subset = stability[
        (stability["alternative_min_unit_papers"] == args.baseline_min_unit_papers) &
        (stability["alternative_min_event_denominator"] == args.baseline_min_denominator)
    ].copy()
    matrix = subset.groupby(["metric", "alternative_window_h"])["spearman_rho"].median().unstack()
    matrix = matrix.reindex(["exposure", "contamination", "persistence_years", "recovery_slope"])
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    values = matrix.to_numpy(dtype=float)
    image = ax.imshow(values, vmin=-1, vmax=1, cmap="RdYlBu", aspect="auto")
    ax.set_xticks(range(len(matrix.columns)), [f"H={int(x)}" for x in matrix.columns])
    ax.set_yticks(range(len(matrix.index)), [x.replace("_", " ").title() for x in matrix.index])
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            text = "NA" if not np.isfinite(values[i, j]) else f"{values[i, j]:.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=10)
    fig.colorbar(image, ax=ax, label="Median Spearman rank correlation vs. H=4 baseline")
    ax.set_title("Ranking Stability Under Alternative Follow-up Windows")
    fig.tight_layout()
    save_dual_figure(fig, output_dir / "fig_step5_indicator_ranking_stability")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if any(h <= 0 for h in args.windows):
        raise ValueError("All follow-up windows must be positive integers.")
    if args.baseline_window not in args.windows:
        raise ValueError("--baseline-window must be one of the values passed to --windows.")
    if args.baseline_min_unit_papers not in args.min_unit_papers:
        raise ValueError("--baseline-min-unit-papers must be one of --min-unit-papers.")
    if args.baseline_min_denominator not in args.min_denominators:
        raise ValueError("--baseline-min-denominator must be one of --min-denominators.")

    root = project_root()
    dirs = revision_dirs(root)
    output_dir = dirs["output"]
    master = read_master(root)
    cols = infer_columns(master)
    data = prepare_data(master, cols)
    print(f"Frozen corpus: {len(data):,} records with a retraction year and citation-history field.")

    estimates_frames: list[pd.DataFrame] = []
    global_rows: list[dict[str, Any]] = []
    eligibility_rows: list[dict[str, Any]] = []
    for h in sorted(set(args.windows)):
        eligible = eligible_for_window(data, h, args)
        eligibility_rows.append({
            "followup_window_h": h,
            "citation_history_start_year": history_start_year(args),
            "as_of_year": args.as_of_year,
            "n_eligible_papers": len(eligible),
            "n_excluded_for_incomplete_window": len(data) - len(eligible),
        })
        print(f"H={h}: {len(eligible):,} papers with a complete common event window.")
        if eligible.empty:
            continue
        for min_denom in sorted(set(args.min_denominators)):
            global_rows.append(global_summary(eligible, h, min_denom))
            for min_n in sorted(set(args.min_unit_papers)):
                result = compute_specification(eligible, h, min_n, min_denom)
                if not result.empty:
                    estimates_frames.append(result)

    if not estimates_frames:
        raise RuntimeError("No indicator estimates were produced. Check citation-history coverage and eligibility settings.")

    estimates = pd.concat(estimates_frames, ignore_index=True)
    globals_df = pd.DataFrame(global_rows)
    eligibility_df = pd.DataFrame(eligibility_rows)
    stability = compute_stability(estimates, args)

    estimates.to_csv(output_dir / "step5_indicator_estimates_long.csv", index=False)
    globals_df.to_csv(output_dir / "step5_indicator_global_sensitivity.csv", index=False)
    eligibility_df.to_csv(output_dir / "step5_indicator_eligibility.csv", index=False)
    stability.to_csv(output_dir / "step5_indicator_ranking_stability.csv", index=False)
    make_summary_table(globals_df, output_dir, args)
    make_stability_table(stability, output_dir, args)
    plot_sensitivity(globals_df, stability, output_dir, args)

    manifest = {
        "created_at": now_utc(),
        "source": "revision/data/revision_master.csv.gz",
        "citation_history_start_year": history_start_year(args),
        "as_of_year": args.as_of_year,
        "windows": sorted(set(args.windows)),
        "min_unit_papers": sorted(set(args.min_unit_papers)),
        "min_event_denominators": sorted(set(args.min_denominators)),
        "baseline_specification": {
            "followup_window_h": args.baseline_window,
            "min_unit_papers": args.baseline_min_unit_papers,
            "min_event_denominator": args.baseline_min_denominator,
        },
        "formulas": {
            "exposure": "sum_tau A_U(tau) / sum_tau [A_U(tau)+B_U(tau)]",
            "contamination": "mean_tau A_U(tau)/[A_U(tau)+B_U(tau)] over denominators meeting threshold",
            "persistence": "max tau in 1..H with A_U(tau)>0, else 0",
            "recovery": "OLS slope of Q_U(tau) on tau over eligible event times",
        },
        "outputs": [
            "step5_indicator_estimates_long.csv",
            "step5_indicator_global_sensitivity.csv",
            "step5_indicator_eligibility.csv",
            "step5_indicator_ranking_stability.csv",
            "table_step5_indicator_sensitivity.tex",
            "table_step5_indicator_stability.tex",
            "fig_step5_indicator_window_sensitivity.png/.pdf",
            "fig_step5_indicator_ranking_stability.png/.pdf",
        ],
    }
    write_json(output_dir / "step5_indicator_sensitivity_manifest.json", manifest)
    print("Step 5 complete. Inspect the global sensitivity and ranking-stability files before reporting indicators.")


if __name__ == "__main__":
    main()
