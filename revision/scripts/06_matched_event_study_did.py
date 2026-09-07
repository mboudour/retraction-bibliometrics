#!/usr/bin/env python3
"""Step 4: matched-control event study for post-retraction citation persistence.

This script replaces the earlier within-treated-paper comparison that was
incorrectly described as difference-in-differences (DiD). It constructs a
non-retracted control group from OpenAlex and estimates matched-pair event-time
DiD contrasts.

Matching design
---------------
* Treated papers: records in revision/data/revision_master.csv.gz.
* Candidate controls: OpenAlex works with is_retracted:false.
* Exact matching: venue ISSN-L and publication year.
* Nearest-neighbour matching without replacement: mean log(1 + citations) in
  the treated paper's three calendar years before retraction (t = -3,-2,-1).
* Each control receives the treated paper's retraction year as a pseudo-event
  year. This aligns treated and control papers by both publication cohort and
  calendar time.

Estimator
---------
For each matched pair m and event time tau, the outcome is log(1 + annual
citations). Let d_m(tau) = y_T,m(tau) - y_C,m(tau). The estimated event-time
DiD contrast is d_m(tau) minus the pair's mean d_m(-3:-1). Standard errors are
computed across matched pairs, which is equivalent to clustering at matched-pair
level for this one-observation-per-pair-per-event-time estimator.

IMPORTANT: Citation counts at t=0 are calendar-year counts and may include both
pre- and post-notice citations. The script reports them but does not interpret
them as a clean post-retraction treatment effect. Effects are observational
unless the matching balance and pre-trend diagnostics support a stronger
interpretation.

Run from the project root after Step 1:
    python revision/scripts/06_matched_event_study_did.py --mode all \
      --api-key "$OPENALEX_API_KEY" --email your.email@university.edu

The first run can take substantial time because it obtains and caches candidate
controls. It resumes safely after interruption. Re-run with --mode analyze after
controls have been cached.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    find_column,
    latex_escape,
    normalise_doi,
    now_utc,
    project_root,
    read_master,
    revision_dirs,
    save_dual_figure,
    write_json,
)

API_URL = "https://api.openalex.org/works"
SOURCE_API_URL = "https://api.openalex.org/sources/issn:"
BASELINE_TIMES = (-3, -2, -1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construct matched non-retracted controls and run event-time DiD."
    )
    parser.add_argument("--mode", choices=("fetch", "analyze", "all"), default="all",
                        help="fetch controls only, analyze an existing cache, or do both.")
    parser.add_argument("--api-key", default=os.getenv("OPENALEX_API_KEY", ""),
                        help="OpenAlex API key; defaults to OPENALEX_API_KEY.")
    parser.add_argument("--email", default=os.getenv("OPENALEX_EMAIL", ""),
                        help="Contact email for OpenAlex polite-pool requests.")
    parser.add_argument("--event-window", type=int, default=10,
                        help="Symmetric event window in years (default: 10).")
    parser.add_argument("--candidate-multiplier", type=int, default=3,
                        help="Candidate controls requested per treated paper in each stratum.")
    parser.add_argument("--max-candidates-per-stratum", type=int, default=1000,
                        help="Hard API retrieval ceiling per venue-year stratum.")
    parser.add_argument("--baseline-caliper", type=float, default=1.00,
                        help="Maximum absolute difference in mean log(1+citations) at t=-3:-1.")
    parser.add_argument("--as-of-year", type=int, default=date.today().year,
                        help="Do not analyze event years after this calendar year.")
    parser.add_argument("--sleep", type=float, default=0.12,
                        help="Seconds between API requests (default: 0.12).")
    parser.add_argument("--seed", type=int, default=20260907,
                        help="Random seed used only for deterministic tie-breaking.")
    parser.add_argument("--max-treated", type=int, default=None,
                        help="Optional small test run; never use for final analysis.")
    parser.add_argument("--force-refetch", action="store_true",
                        help="Ignore completed-stratum fetch checkpoints. Cached records are retained.")
    return parser.parse_args()


def parse_counts(value: Any) -> dict[int, int]:
    """Return an annual citation dictionary from an OpenAlex counts_by_year value."""
    if isinstance(value, (dict, list)):
        raw = value
    else:
        try:
            raw = json.loads(value) if isinstance(value, str) else []
        except (TypeError, ValueError, json.JSONDecodeError):
            raw = []
    if isinstance(raw, dict):
        items = raw.items()
        output: dict[int, int] = {}
        for year, count in items:
            try:
                output[int(year)] = int(count)
            except (TypeError, ValueError):
                continue
        return output
    output = {}
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                output[int(item["year"])] = int(item["cited_by_count"])
            except (KeyError, TypeError, ValueError):
                continue
    return output


def canonical_openalex_id(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).rstrip("/").strip()


def infer_columns(df: pd.DataFrame) -> dict[str, str | None]:
    columns = df.columns
    out = {
        "oa_id": find_column(columns, ["revision_openalex_id", "openalex_id", "id"]),
        "doi": find_column(columns, ["doi_normalised", "doi", "revision_doi"]),
        "pub_year": find_column(columns, ["publication_year", "pub_year", "orig_year"]),
        "ret_year": find_column(columns, ["retraction_year", "revision_retraction_year"]),
        "ret_date": find_column(columns, ["RetractionDate", "revision_retraction_date"]),
        "issn_l": find_column(columns, [
            "journal_issn_l", "primary_location_source_issn_l", "issn_l", "ISSN"
        ]),
        "journal": find_column(columns, ["journal_name", "Journal", "Title"]),
        "concept": find_column(columns, ["top_concept", "concepts_top", "primary_topic"]),
        "counts": find_column(columns, ["counts_by_year_json", "counts_by_year"]),
    }
    missing = [key for key in ("oa_id", "pub_year", "ret_year", "issn_l", "counts") if not out[key]]
    if missing:
        raise ValueError(
            "The frozen master dataset lacks required columns: " + ", ".join(missing) +
            ". Required variables are OpenAlex ID, publication year, retraction year, "
            "venue ISSN-L, and counts_by_year_json."
        )
    return out


def prepare_treated(master: pd.DataFrame, cols: dict[str, str | None], args: argparse.Namespace) -> pd.DataFrame:
    df = master.copy()
    df["treated_id"] = df[cols["oa_id"]].map(canonical_openalex_id)
    df["treated_doi"] = normalise_doi(df[cols["doi"]]) if cols["doi"] else pd.NA
    df["pub_year_match"] = pd.to_numeric(df[cols["pub_year"]], errors="coerce")
    df["ret_year_match"] = pd.to_numeric(df[cols["ret_year"]], errors="coerce")
    if df["ret_year_match"].isna().all() and cols["ret_date"]:
        df["ret_year_match"] = pd.to_datetime(df[cols["ret_date"]], errors="coerce").dt.year
    df["issn_l_match"] = df[cols["issn_l"]].fillna("").astype(str).str.strip().str.upper()
    df["journal_match"] = df[cols["journal"]].fillna("").astype(str).str.strip() if cols["journal"] else ""
    df["concept_match"] = df[cols["concept"]].fillna("Unknown").astype(str).str.strip() if cols["concept"] else "Unknown"
    df["treated_counts"] = df[cols["counts"]].apply(parse_counts)
    df["paper_age_at_retraction"] = df["ret_year_match"] - df["pub_year_match"]

    # A common three-year baseline is required for a valid matched DiD contrast.
    eligible = df[
        df["treated_id"].ne("") &
        df["pub_year_match"].notna() &
        df["ret_year_match"].notna() &
        df["issn_l_match"].ne("") &
        (df["paper_age_at_retraction"] >= 3)
    ].copy()
    eligible["pub_year_match"] = eligible["pub_year_match"].astype(int)
    eligible["ret_year_match"] = eligible["ret_year_match"].astype(int)
    eligible["paper_age_at_retraction"] = eligible["paper_age_at_retraction"].astype(int)
    eligible["stratum"] = eligible["issn_l_match"] + "|" + eligible["pub_year_match"].astype(str)
    eligible["baseline_log_cites"] = eligible.apply(
        lambda row: baseline_log_citations(row["treated_counts"], int(row["ret_year_match"])), axis=1
    )
    eligible = eligible.replace([np.inf, -np.inf], np.nan).dropna(subset=["baseline_log_cites"])
    eligible = eligible.sort_values(["stratum", "ret_year_match", "treated_id"]).reset_index(drop=True)
    if args.max_treated:
        eligible = eligible.head(args.max_treated).copy()
    return eligible


def baseline_log_citations(counts: dict[int, int], event_year: int) -> float:
    values = [math.log1p(max(0, int(counts.get(event_year + t, 0)))) for t in BASELINE_TIMES]
    return float(np.mean(values))


def extract_control_record(record: dict[str, Any]) -> dict[str, Any] | None:
    source = ((record.get("primary_location") or {}).get("source") or {})
    issn_l = str(source.get("issn_l") or "").strip().upper()
    work_id = canonical_openalex_id(record.get("id"))
    pub_year = record.get("publication_year")
    if not work_id or not issn_l or pub_year is None:
        return None
    try:
        pub_year = int(pub_year)
    except (TypeError, ValueError):
        return None
    doi = record.get("doi") or ""
    return {
        "control_id": work_id,
        "control_doi": str(doi).replace("https://doi.org/", "").lower().strip(),
        "control_pub_year": pub_year,
        "control_issn_l": issn_l,
        "control_journal": str(source.get("display_name") or "").strip(),
        "control_concept": str(((record.get("primary_topic") or {}).get("display_name") or "")).strip(),
        "control_counts": parse_counts(record.get("counts_by_year") or []),
        "control_cited_by_count": record.get("cited_by_count"),
    }


def load_control_cache(path: Path) -> dict[tuple[str, int], list[dict[str, Any]]]:
    index: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    if not path.exists():
        return index
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            item = extract_control_record(rec)
            if not item or item["control_id"] in seen:
                continue
            seen.add(item["control_id"])
            index[(item["control_issn_l"], item["control_pub_year"])].append(item)
    return index


def request_json(
    session: requests.Session,
    params: dict[str, Any],
    retries: int = 6,
    url: str = API_URL,
) -> dict[str, Any]:
    for attempt in range(retries):
        try:
            response = session.get(url, params=params, timeout=60)
            if response.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise RuntimeError(f"OpenAlex request failed after {retries} attempts: {exc}") from exc
            time.sleep(min(60, 2 ** (attempt + 1)))
    raise RuntimeError("Unreachable retry state")


def fetch_controls(
    treated: pd.DataFrame,
    control_cache: Path,
    progress_path: Path,
    source_id_map_path: Path,
    args: argparse.Namespace,
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    """Retrieve non-retracted venue-year candidates and append raw records to cache."""
    if not args.api_key and not args.email:
        raise ValueError("Supply --api-key or --email before fetching OpenAlex controls.")

    cache = load_control_cache(control_cache)
    if source_id_map_path.exists():
        try:
            source_id_map = json.loads(source_id_map_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            source_id_map = {}
    else:
        source_id_map = {}
    completed: set[str] = set()
    if progress_path.exists() and not args.force_refetch:
        try:
            completed = set(json.loads(progress_path.read_text(encoding="utf-8")).get("completed_strata", []))
        except (OSError, ValueError, json.JSONDecodeError):
            completed = set()

    treated_ids = set(treated["treated_id"].astype(str))
    treated_dois = set(treated["treated_doi"].dropna().astype(str))
    grouped = list(treated.groupby(["issn_l_match", "pub_year_match"], sort=True))
    session = requests.Session()
    session.headers.update({"User-Agent": f"RetractionBibliometrics/Revision-Step4 ({args.email or 'OpenAlex API key'})"})
    select = "id,doi,publication_year,primary_location,primary_topic,counts_by_year,cited_by_count,is_retracted,type"

    control_cache.parent.mkdir(parents=True, exist_ok=True)
    with control_cache.open("a", encoding="utf-8") as fout:
        for number, ((issn_l, pub_year), group) in enumerate(grouped, start=1):
            stratum = f"{issn_l}|{int(pub_year)}"
            desired = min(args.max_candidates_per_stratum,
                          max(10, len(group) * args.candidate_multiplier))
            existing = cache.get((issn_l, int(pub_year)), [])
            if stratum in completed and len(existing) >= min(desired, 10):
                continue

            source_id = source_id_map.get(str(issn_l))
            if not source_id:
                source_params: dict[str, Any] = {}
                if args.api_key:
                    source_params["api_key"] = args.api_key
                else:
                    source_params["mailto"] = args.email
                try:
                    source = request_json(
                        session,
                        source_params,
                        url=SOURCE_API_URL + str(issn_l),
                    )
                    source_id = canonical_openalex_id(source.get("id", "")).split("/")[-1]
                except RuntimeError as exc:
                    print(f"[WARN] Could not resolve ISSN-L {issn_l}: {exc}")
                    source_id = ""
                source_id_map[str(issn_l)] = source_id
                source_id_map_path.write_text(
                    json.dumps(source_id_map, indent=2, sort_keys=True), encoding="utf-8"
                )
            if not source_id:
                completed.add(stratum)
                progress_path.write_text(
                    json.dumps({"completed_strata": sorted(completed), "updated_at": now_utc()}, indent=2),
                    encoding="utf-8",
                )
                continue

            candidates: list[dict[str, Any]] = []
            cursor = "*"
            while len(candidates) < desired and cursor:
                params: dict[str, Any] = {
                    "filter": (
                        f"is_retracted:false,type:article,has_doi:true,"
                        f"publication_year:{int(pub_year)},"
                        f"primary_location.source.id:{source_id}"
                    ),
                    "select": select,
                    "per-page": min(200, desired - len(candidates)),
                    "cursor": cursor,
                }
                if args.api_key:
                    params["api_key"] = args.api_key
                else:
                    params["mailto"] = args.email
                data = request_json(session, params)
                page = data.get("results", [])
                if not page:
                    break
                for raw in page:
                    item = extract_control_record(raw)
                    if not item:
                        continue
                    if item["control_id"] in treated_ids or item["control_doi"] in treated_dois:
                        continue
                    candidates.append(raw)
                    fout.write(json.dumps(raw, ensure_ascii=False) + "\n")
                cursor = (data.get("meta") or {}).get("next_cursor")
                time.sleep(args.sleep)

            # Reload only the new stratum in a simple robust way; the cache remains the source of truth.
            cache = load_control_cache(control_cache)
            completed.add(stratum)
            progress_path.write_text(
                json.dumps({"completed_strata": sorted(completed), "updated_at": now_utc()}, indent=2),
                encoding="utf-8",
            )
            if number % 100 == 0 or number == len(grouped):
                print(f"Fetched/checkpointed {number:,}/{len(grouped):,} venue-year strata.")
    return cache


def greedy_match(
    treated: pd.DataFrame,
    candidates: dict[tuple[str, int], list[dict[str, Any]]],
    caliper: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    pair_rows: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []

    for (issn_l, pub_year), group in treated.groupby(["issn_l_match", "pub_year_match"], sort=True):
        pool = candidates.get((str(issn_l), int(pub_year)), [])
        available = {item["control_id"]: item for item in pool}
        # Treat papers with the least common baseline profiles first.
        work = group.sample(frac=1.0, random_state=int(rng.integers(1, 2**31 - 1)))
        for _, row in work.iterrows():
            event_year = int(row["ret_year_match"])
            t_base = float(row["baseline_log_cites"])
            scored: list[tuple[float, float, dict[str, Any]]] = []
            for control in available.values():
                c_base = baseline_log_citations(control["control_counts"], event_year)
                distance = abs(t_base - c_base)
                if distance <= caliper:
                    scored.append((distance, rng.random(), control))
            if not scored:
                unmatched.append({
                    "treated_id": row["treated_id"],
                    "treated_doi": row["treated_doi"],
                    "issn_l": issn_l,
                    "publication_year": int(pub_year),
                    "retraction_year": event_year,
                    "reason": "no_unused_control_within_baseline_caliper",
                    "treated_baseline_log_cites": t_base,
                    "candidate_controls_in_stratum": len(pool),
                })
                continue
            scored.sort(key=lambda item: (item[0], item[1]))
            distance, _, control = scored[0]
            del available[control["control_id"]]
            pair_rows.append({
                "pair_id": f"pair_{len(pair_rows)+1:07d}",
                "treated_id": row["treated_id"],
                "treated_doi": row["treated_doi"],
                "control_id": control["control_id"],
                "control_doi": control["control_doi"],
                "issn_l": issn_l,
                "journal": row["journal_match"],
                "publication_year": int(pub_year),
                "pseudo_event_year": event_year,
                "paper_age_at_event": int(row["paper_age_at_retraction"]),
                "concept": row["concept_match"],
                "treated_baseline_log_cites": t_base,
                "control_baseline_log_cites": baseline_log_citations(control["control_counts"], event_year),
                "baseline_abs_distance": distance,
                "treated_counts_json": json.dumps(row["treated_counts"], sort_keys=True),
                "control_counts_json": json.dumps(control["control_counts"], sort_keys=True),
            })
    return pd.DataFrame(pair_rows), pd.DataFrame(unmatched)


def standardized_mean_difference(x: pd.Series, y: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    y = pd.to_numeric(y, errors="coerce").dropna()
    if not len(x) or not len(y):
        return np.nan
    pooled = math.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
    return 0.0 if pooled == 0 else float((x.mean() - y.mean()) / pooled)


def make_balance_table(pairs: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame()
    rows = [
        {
            "variable": "Publication year (exact match)",
            "treated_mean": pairs["publication_year"].mean(),
            "control_mean": pairs["publication_year"].mean(),
            "standardized_mean_difference": 0.0,
            "note": "Exact venue-year matching",
        },
        {
            "variable": "Venue ISSN-L (exact match)",
            "treated_mean": np.nan,
            "control_mean": np.nan,
            "standardized_mean_difference": 0.0,
            "note": f"{pairs['issn_l'].nunique():,} matched venue strata",
        },
        {
            "variable": "Mean log(1 + citations), t=-3:-1",
            "treated_mean": pairs["treated_baseline_log_cites"].mean(),
            "control_mean": pairs["control_baseline_log_cites"].mean(),
            "standardized_mean_difference": standardized_mean_difference(
                pairs["treated_baseline_log_cites"], pairs["control_baseline_log_cites"]
            ),
            "note": "Nearest-neighbour matching without replacement",
        },
        {
            "variable": "Absolute baseline distance",
            "treated_mean": pairs["baseline_abs_distance"].mean(),
            "control_mean": np.nan,
            "standardized_mean_difference": np.nan,
            "note": "Mean absolute treated-control difference",
        },
    ]
    return pd.DataFrame(rows)


def build_event_estimates(pairs: pd.DataFrame, event_window: int, as_of_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    pair_pretrend: list[dict[str, Any]] = []
    for _, pair in pairs.iterrows():
        t_counts = parse_counts(pair["treated_counts_json"])
        c_counts = parse_counts(pair["control_counts_json"])
        event_year = int(pair["pseudo_event_year"])
        pub_year = int(pair["publication_year"])
        baseline_diffs: list[float] = []
        for tau in BASELINE_TIMES:
            calendar_year = event_year + tau
            if calendar_year < pub_year or calendar_year > as_of_year:
                continue
            baseline_diffs.append(math.log1p(t_counts.get(calendar_year, 0)) - math.log1p(c_counts.get(calendar_year, 0)))
        if len(baseline_diffs) != len(BASELINE_TIMES):
            continue
        baseline = float(np.mean(baseline_diffs))
        pre_rows: list[tuple[int, float]] = []
        for tau in range(-event_window, event_window + 1):
            calendar_year = event_year + tau
            if calendar_year < pub_year or calendar_year > as_of_year:
                continue
            t_value = math.log1p(t_counts.get(calendar_year, 0))
            c_value = math.log1p(c_counts.get(calendar_year, 0))
            raw_difference = t_value - c_value
            did_difference = raw_difference - baseline
            rows.append({
                "pair_id": pair["pair_id"],
                "event_time": tau,
                "calendar_year": calendar_year,
                "treated_log_citations": t_value,
                "control_log_citations": c_value,
                "raw_treated_minus_control": raw_difference,
                "baseline_pair_difference": baseline,
                "did_pair_contrast": did_difference,
            })
            if tau <= -4:
                pre_rows.append((tau, raw_difference))
        if len(pre_rows) >= 2:
            x = np.array([item[0] for item in pre_rows], dtype=float)
            y = np.array([item[1] for item in pre_rows], dtype=float)
            slope = float(np.polyfit(x, y, 1)[0])
            pair_pretrend.append({"pair_id": pair["pair_id"], "pretrend_slope": slope})

    panel = pd.DataFrame(rows)
    if panel.empty:
        return pd.DataFrame(), pd.DataFrame()
    estimates: list[dict[str, Any]] = []
    for tau, group in panel.groupby("event_time", sort=True):
        values = group["did_pair_contrast"].astype(float)
        n = len(values)
        estimate = values.mean()
        se = values.std(ddof=1) / math.sqrt(n) if n > 1 else np.nan
        if n > 1 and np.isfinite(se) and se > 0:
            t_stat = estimate / se
            p_value = 2 * stats.t.sf(abs(t_stat), df=n - 1)
            crit = stats.t.ppf(0.975, df=n - 1)
            ci_low = estimate - crit * se
            ci_high = estimate + crit * se
        else:
            t_stat = p_value = ci_low = ci_high = np.nan
        estimates.append({
            "event_time": int(tau),
            "n_matched_pairs": int(n),
            "did_estimate_log1p_citations": estimate,
            "pair_clustered_se": se,
            "ci_95_lower": ci_low,
            "ci_95_upper": ci_high,
            "t_statistic": t_stat,
            "p_value": p_value,
            "mean_treated_log_citations": group["treated_log_citations"].mean(),
            "mean_control_log_citations": group["control_log_citations"].mean(),
        })
    pretrend = pd.DataFrame(pair_pretrend)
    estimates_df = pd.DataFrame(estimates)
    if not pretrend.empty and len(pretrend) > 1:
        result = stats.ttest_1samp(pretrend["pretrend_slope"], popmean=0.0, nan_policy="omit")
        estimates_df.attrs["pretrend"] = {
            "n_pairs": int(pretrend["pretrend_slope"].notna().sum()),
            "mean_slope": float(pretrend["pretrend_slope"].mean()),
            "t_statistic": float(result.statistic),
            "p_value": float(result.pvalue),
        }
    else:
        estimates_df.attrs["pretrend"] = {"n_pairs": 0, "mean_slope": np.nan, "t_statistic": np.nan, "p_value": np.nan}
    return panel, estimates_df


def write_latex_tables(balance: pd.DataFrame, estimates: pd.DataFrame, output_dir: Path) -> None:
    newline = r"\\"
    b_rows = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Balance diagnostics for the matched event-study sample. Venue and publication year are exact matching variables; the citation baseline is mean log(1+annual citations) over event times $-3$ to $-1$.}",
        r"\label{tab:did_balance}",
        r"\begin{tabular}{p{0.40\linewidth}rrr}",
        r"\toprule",
        "Variable & Treated & Control & SMD " + newline,
        r"\midrule",
    ]
    for _, row in balance.iterrows():
        treated = "" if pd.isna(row["treated_mean"]) else f"{row['treated_mean']:.3f}"
        control = "" if pd.isna(row["control_mean"]) else f"{row['control_mean']:.3f}"
        smd = "" if pd.isna(row["standardized_mean_difference"]) else f"{row['standardized_mean_difference']:.3f}"
        b_rows.append(f"{latex_escape(row['variable'])} & {treated} & {control} & {smd} " + newline)
    b_rows += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (output_dir / "table_step4_matching_balance.tex").write_text("\n".join(b_rows), encoding="utf-8")

    e_rows = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Matched-pair event-study difference-in-differences estimates. The outcome is log(1+annual citations). Each estimate compares the treated--control difference at event time $\tau$ with the pair-specific mean difference over $\tau=-3,-2,-1$. Standard errors are clustered at matched-pair level.}",
        r"\label{tab:did_eventstudy}",
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"Event time & Pairs & Estimate & Pair-clustered SE & 95\% CI & $p$ " + newline,
        r"\midrule",
    ]
    for _, row in estimates.iterrows():
        ci = f"[{row['ci_95_lower']:.3f}, {row['ci_95_upper']:.3f}]" if pd.notna(row["ci_95_lower"]) else ""
        pvalue = f"{row['p_value']:.3g}" if pd.notna(row["p_value"]) else ""
        e_rows.append(
            f"{int(row['event_time'])} & {int(row['n_matched_pairs']):,} & "
            f"{row['did_estimate_log1p_citations']:.3f} & {row['pair_clustered_se']:.3f} & {ci} & {pvalue} " + newline
        )
    e_rows += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (output_dir / "table_step4_event_study.tex").write_text("\n".join(e_rows), encoding="utf-8")


def plot_event_study(estimates: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.3))
    x = estimates["event_time"].to_numpy()
    y = estimates["did_estimate_log1p_citations"].to_numpy()
    low = estimates["ci_95_lower"].to_numpy()
    high = estimates["ci_95_upper"].to_numpy()
    ax.fill_between(x, low, high, color="#2C7FB8", alpha=0.20, label="95% confidence interval")
    ax.plot(x, y, color="#045A8D", linewidth=2.4, marker="o", markersize=4.5,
            label="Matched-pair DiD estimate")
    ax.axhline(0, color="#333333", linewidth=1.0)
    ax.axvline(0, color="#B2182B", linewidth=1.3, linestyle="--", label="Retraction year")
    ax.axvspan(-3.2, -0.8, color="#DDDDDD", alpha=0.35, label="Reference window ($t=-3$ to $-1$)")
    ax.set_xlabel("Years relative to retraction / matched pseudo-event")
    ax.set_ylabel("DiD contrast in log(1 + annual citations)")
    ax.set_title("Matched-Control Event Study of Post-Retraction Citation Persistence")
    ax.legend(frameon=False, fontsize=9, loc="best")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    save_dual_figure(fig, output_dir / "fig_step4_matched_event_study")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = project_root()
    dirs = revision_dirs(root)
    output_dir = dirs["output"]
    data_dir = dirs["data"]
    cache_path = data_dir / "step4_nonretracted_control_cache.jsonl"
    progress_path = data_dir / "step4_control_fetch_progress.json"
    source_id_map_path = data_dir / "step4_issn_to_openalex_source_id.json"

    print("Loading frozen Step 1 master corpus …")
    master = read_master(root)
    cols = infer_columns(master)
    treated = prepare_treated(master, cols, args)
    if treated.empty:
        raise RuntimeError("No eligible treated records after applying common three-year baseline requirements.")
    treated.to_csv(output_dir / "step4_treated_eligibility.csv", index=False)
    print(f"Eligible treated papers: {len(treated):,}; venue-year strata: {treated['stratum'].nunique():,}")

    candidates = load_control_cache(cache_path)
    if args.mode in {"fetch", "all"}:
        candidates = fetch_controls(treated, cache_path, progress_path, source_id_map_path, args)
        print(f"Cached candidate controls: {sum(len(v) for v in candidates.values()):,}")
    if args.mode == "fetch":
        print("Fetch complete. Re-run with --mode analyze to perform matching and estimation.")
        return

    print("Matching treated papers to non-retracted controls …")
    pairs, unmatched = greedy_match(treated, candidates, args.baseline_caliper, args.seed)
    if pairs.empty:
        raise RuntimeError(
            "No matched pairs were formed. Inspect candidate availability and consider a larger "
            "--max-candidates-per-stratum or a wider --baseline-caliper."
        )
    pairs.to_csv(output_dir / "step4_matched_pairs.csv", index=False)
    unmatched.to_csv(output_dir / "step4_unmatched_treated.csv", index=False)
    print(f"Matched pairs: {len(pairs):,}; unmatched eligible treated papers: {len(unmatched):,}")

    balance = make_balance_table(pairs)
    balance.to_csv(output_dir / "step4_matching_balance.csv", index=False)

    print("Estimating event-time matched-pair DiD contrasts …")
    panel, estimates = build_event_estimates(pairs, args.event_window, args.as_of_year)
    if estimates.empty:
        raise RuntimeError("No event-study estimates were produced from the matched pairs.")
    panel.to_csv(output_dir / "step4_matched_event_panel.csv", index=False)
    estimates.to_csv(output_dir / "step4_matched_event_study_estimates.csv", index=False)
    write_latex_tables(balance, estimates, output_dir)
    plot_event_study(estimates, output_dir)

    pretrend = estimates.attrs.get("pretrend", {})
    summary = {
        "created_at": now_utc(),
        "design": {
            "treated_source": "revision_master.csv.gz",
            "control_source": "OpenAlex API, is_retracted:false",
            "exact_matching": ["venue ISSN-L", "publication year"],
            "nearest_neighbour_variable": "mean log(1+annual citations) at event times -3,-2,-1",
            "matching_with_replacement": False,
            "baseline_caliper": args.baseline_caliper,
            "event_window": [-args.event_window, args.event_window],
            "as_of_year": args.as_of_year,
            "outcome": "log(1 + annual citations)",
        },
        "n_frozen_master": int(len(master)),
        "n_eligible_treated": int(len(treated)),
        "n_matched_pairs": int(len(pairs)),
        "n_unmatched_eligible_treated": int(len(unmatched)),
        "matching_rate": float(len(pairs) / len(treated)),
        "pretrend_slope_test": pretrend,
        "outputs": {
            "pairs": "step4_matched_pairs.csv",
            "unmatched": "step4_unmatched_treated.csv",
            "balance": "step4_matching_balance.csv",
            "event_panel": "step4_matched_event_panel.csv",
            "estimates": "step4_matched_event_study_estimates.csv",
            "figure": "fig_step4_matched_event_study.png/.pdf",
            "latex_tables": ["table_step4_matching_balance.tex", "table_step4_event_study.tex"],
        },
    }
    write_json(output_dir / "step4_matched_event_study_summary.json", summary)
    print("Step 4 complete. Inspect matching balance and pre-trend diagnostics before interpreting effects.")


if __name__ == "__main__":
    main()
