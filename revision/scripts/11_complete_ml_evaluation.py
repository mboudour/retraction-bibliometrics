#!/usr/bin/env python3
"""Step 9: complete, leakage-conscious evaluation of retraction-risk classifiers.

This script evaluates four prespecified classifiers under three protocols
(random, temporal, and cross-discipline transfer) and two feature sets:
(1) at-publication metadata only and (2) an explicitly labelled one-year
post-publication early-warning extension. It reports prevalence, balancing,
confusion-matrix counts, ROC-AUC, PR-AUC, calibration diagnostics, and all
requested threshold metrics. It does not include structural network variables,
post-retraction variables, publication-year features, or total citation counts.

The outcome is a retrospective class label, not a causal estimate or an editorial
screening rule. The non-retracted comparison sample is publication-year and broad
field matched to the retracted sample, sampled without replacement where possible.

Run from the project root:
  python revision/scripts/11_complete_ml_evaluation.py --api-key "$OPENALEX_API_KEY" --email "..."
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, log_loss, matthews_corrcoef, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import find_column, now_utc, project_root, read_master, revision_dirs, save_dual_figure, write_json  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
OPENALEX_WORKS = "https://api.openalex.org/works"
SEED = 20260910
AT_PUBLICATION = ["title_length", "log_n_authors", "log_n_references", "is_oa", "has_abstract"]
EARLY_WARNING = AT_PUBLICATION + ["log_early_citations_year1"]
FEATURE_LABELS = {
    "title_length": "Title length",
    "log_n_authors": "log(1 + authors)",
    "log_n_references": "log(1 + references)",
    "is_oa": "Open access",
    "has_abstract": "Abstract available",
    "log_early_citations_year1": "log(1 + citations in year 1)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 9 complete predictive-model reporting.")
    parser.add_argument("--mode", choices=("all", "fetch", "analyze"), default="all")
    parser.add_argument("--api-key", default="", help="OpenAlex API key; required only for --mode all/fetch when the cache is insufficient.")
    parser.add_argument("--email", default="", help="Contact email for OpenAlex polite pool.")
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--temporal-cutoff", type=int, default=2021, help="Train through this publication year; test afterward.")
    parser.add_argument("--control-multiplier", type=float, default=2.0, help="Cached negative candidates targeted per eligible positive paper.")
    parser.add_argument("--batch-size", type=int, default=200, choices=(100, 200))
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def json_or_literal(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return None


def citation_counts(value: Any) -> dict[int, int]:
    raw = json_or_literal(value)
    result: dict[int, int] = {}
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                result[int(item.get("year"))] = int(item.get("cited_by_count", 0))
            except (ValueError, TypeError):
                continue
    elif isinstance(raw, dict):
        for year, count in raw.items():
            try:
                result[int(year)] = int(count)
            except (ValueError, TypeError):
                continue
    return result


def bool_value(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0
    return int(str(value).strip().lower() in {"1", "true", "yes", "y"})


def broad_field(value: Any) -> str:
    """Map legacy concepts and current OpenAlex field/topic labels to common broad fields."""
    text = str(value or "").strip().lower()
    if not text or text in {"nan", "none", "unknown"}:
        return "Other/Unknown"
    mapping = [
        ("Medicine/Health", ("medicine", "health", "clinical", "nursing", "pharmacy", "dent", "veterinar", "biomed")),
        ("Biology/Life sciences", ("biology", "life science", "genetic", "ecology", "biochem", "neuroscience", "agricultur")),
        ("Chemistry", ("chemistry", "chemical")),
        ("Physics/Astronomy", ("physics", "astronomy", "space science")),
        ("Materials science", ("materials", "metallurgy", "polymer")),
        ("Computer science", ("computer", "information science", "artificial intelligence", "data science")),
        ("Engineering", ("engineering", "technology", "robotics")),
        ("Mathematics/Statistics", ("mathematics", "statistic")),
        ("Earth/Environmental sciences", ("earth", "environment", "geology", "geography", "climate", "ocean")),
        ("Social sciences", ("social", "economics", "politic", "education", "sociology", "business", "management", "law")),
        ("Psychology", ("psychology", "cognitive science")),
        ("Humanities", ("history", "philosophy", "linguistic", "literature", "art")),
    ]
    for label, terms in mapping:
        if any(term in text for term in terms):
            return label
    return "Other/Unknown"


def field_from_openalex(record: dict[str, Any]) -> str:
    primary = record.get("primary_topic") or {}
    if isinstance(primary, dict):
        field = primary.get("field") or {}
        if isinstance(field, dict):
            name = field.get("display_name")
            if name:
                return broad_field(name)
        name = primary.get("display_name")
        if name:
            return broad_field(name)
    concepts = record.get("concepts") or []
    if concepts and isinstance(concepts, list) and isinstance(concepts[0], dict):
        return broad_field(concepts[0].get("display_name"))
    return "Other/Unknown"


def feature_row_from_master(row: pd.Series, columns: dict[str, str]) -> dict[str, Any]:
    pub_year = int(row[columns["year"]])
    cby = citation_counts(row.get(columns.get("citation_series", ""), None))
    concept = row.get(columns.get("discipline", ""), "Other/Unknown")
    return {
        "work_id": str(row.get(columns.get("work_id", ""), "")),
        "publication_year": pub_year,
        "broad_field": broad_field(concept),
        "title_length": len(str(row.get(columns.get("title", ""), "") or "")),
        "log_n_authors": float(np.log1p(pd.to_numeric(row.get(columns.get("authors", ""), 0), errors="coerce") if pd.notna(pd.to_numeric(row.get(columns.get("authors", ""), 0), errors="coerce")) else 0)),
        "log_n_references": float(np.log1p(pd.to_numeric(row.get(columns.get("references", ""), 0), errors="coerce") if pd.notna(pd.to_numeric(row.get(columns.get("references", ""), 0), errors="coerce")) else 0)),
        "is_oa": bool_value(row.get(columns.get("oa", ""), 0)),
        "has_abstract": bool_value(row.get(columns.get("abstract", ""), 0)),
        "log_early_citations_year1": float(np.log1p(cby.get(pub_year + 1, 0))),
        "label": 1,
    }


def feature_row_from_openalex(record: dict[str, Any]) -> dict[str, Any] | None:
    try:
        year = int(record.get("publication_year"))
    except (TypeError, ValueError):
        return None
    authorships = record.get("authorships") or []
    references = record.get("referenced_works") or []
    open_access = record.get("open_access") or {}
    abstract = record.get("abstract_inverted_index") or record.get("abstract") or ""
    cby = citation_counts(record.get("counts_by_year") or [])
    return {
        "work_id": str(record.get("id") or ""),
        "publication_year": year,
        "broad_field": field_from_openalex(record),
        "title_length": len(str(record.get("title") or record.get("display_name") or "")),
        "log_n_authors": float(np.log1p(len(authorships))),
        "log_n_references": float(np.log1p(len(references))),
        "is_oa": int(bool(open_access.get("is_oa"))),
        "has_abstract": int(bool(abstract)),
        "log_early_citations_year1": float(np.log1p(cby.get(year + 1, 0))),
        "label": 0,
    }


def master_columns(df: pd.DataFrame) -> dict[str, str]:
    specs = {
        "year": ("publication_year", "pub_year", "orig_year"),
        "title": ("title", "display_name"),
        "authors": ("authorships_count", "n_authors", "author_count"),
        "references": ("referenced_works_count", "n_references", "reference_count"),
        "oa": ("is_oa", "open_access"),
        "abstract": ("has_abstract", "abstract_inverted_index"),
        "citation_series": ("counts_by_year_json", "counts_by_year"),
        "discipline": ("top_concept", "primary_topic_field", "concept"),
        "work_id": ("openalex_id", "id", "work_id"),
        "work_type": ("type", "work_type"),
    }
    result = {key: find_column(df.columns, options) for key, options in specs.items()}
    required = ("year", "title", "authors", "references", "oa", "abstract", "citation_series")
    missing = [key for key in required if result.get(key) is None]
    if missing:
        raise ValueError("Revision master lacks required predictive fields: " + ", ".join(missing))
    return {key: value for key, value in result.items() if value is not None}


def make_positive_dataset(root: Path, args: argparse.Namespace) -> pd.DataFrame:
    master = read_master(root)
    columns = master_columns(master)
    years = pd.to_numeric(master[columns["year"]], errors="coerce")
    eligible = master.loc[years.between(args.start_year, args.end_year)].copy()
    if "work_type" in columns:
        type_values = eligible[columns["work_type"]].fillna("").astype(str).str.lower()
        # Restrict only when the source provides a meaningful type value.
        if (type_values != "").mean() > 0.90:
            eligible = eligible.loc[type_values.isin(("article", "journal-article", "research-article"))]
    rows = [feature_row_from_master(row, columns) for _, row in eligible.iterrows()]
    positives = pd.DataFrame(rows)
    positives = positives.replace([np.inf, -np.inf], np.nan).dropna(subset=EARLY_WARNING)
    positives = positives.drop_duplicates(subset="work_id").reset_index(drop=True)
    if len(positives) < 500:
        raise RuntimeError(f"Only {len(positives)} eligible positive papers after 2016–2024 filtering.")
    return positives


def read_cache(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            oid = str(record.get("id") or "")
            if oid and oid not in seen:
                seen.add(oid)
                records.append(record)
    return records


def api_get(session: requests.Session, params: dict[str, Any]) -> dict[str, Any]:
    last: Exception | None = None
    for attempt in range(6):
        try:
            response = session.get(OPENALEX_WORKS, params=params, timeout=60)
            if response.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(30, 1.8 ** attempt))
                continue
            response.raise_for_status()
            return response.json()
        except requests.RequestException as error:
            last = error
            time.sleep(min(30, 1.8 ** attempt))
    raise RuntimeError(f"OpenAlex request failed after 6 attempts: {last}")


def fetch_negatives(positives: pd.DataFrame, cache_path: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    target = int(math.ceil(len(positives) * args.control_multiplier))
    records = read_cache(cache_path)
    if len(records) >= target:
        return records
    if not args.api_key:
        raise RuntimeError(f"The Step 9 control cache has {len(records):,}/{target:,} records. Supply --api-key to fetch the remainder.")
    session = requests.Session()
    session.headers.update({"User-Agent": f"RetractBibliometrics/Step9 (mailto:{args.email or 'research@example.org'})"})
    select = "id,publication_year,type,title,display_name,authorships,referenced_works,open_access,abstract_inverted_index,counts_by_year,primary_topic,concepts"
    with cache_path.open("a", encoding="utf-8") as handle:
        draw = 0
        seen = {str(record.get("id") or "") for record in records}
        while len(records) < target:
            draw += 1
            params = {
                "filter": f"is_retracted:false,type:article,from_publication_date:{args.start_year}-01-01,to_publication_date:{args.end_year}-12-31,has_doi:true",
                "sample": args.batch_size,
                "seed": args.seed + draw,
                "per-page": args.batch_size,
                "select": select,
            }
            if args.api_key:
                params["api_key"] = args.api_key
            if args.email:
                params["mailto"] = args.email
            response = api_get(session, params)
            new = 0
            for record in response.get("results", []):
                oid = str(record.get("id") or "")
                if not oid or oid in seen:
                    continue
                seen.add(oid)
                records.append(record)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                new += 1
                if len(records) >= target:
                    break
            print(f"Step 9 control candidates: {len(records):,}/{target:,} (draw {draw}; added {new})")
            if new == 0:
                time.sleep(1.0)
            time.sleep(0.12)
    return records


def stratified_control_sample(positives: pd.DataFrame, raw_controls: list[dict[str, Any]], seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    controls = pd.DataFrame([row for record in raw_controls if (row := feature_row_from_openalex(record)) is not None])
    controls = controls.replace([np.inf, -np.inf], np.nan).dropna(subset=EARLY_WARNING).drop_duplicates("work_id")
    rng = np.random.default_rng(seed)
    used: set[int] = set()
    selected: list[int] = []
    records: list[dict[str, Any]] = []
    by_stratum: dict[tuple[int, str], list[int]] = defaultdict(list)
    for index, row in controls.iterrows():
        by_stratum[(int(row.publication_year), str(row.broad_field))].append(index)
    for _, pos in positives.sample(frac=1, random_state=seed).iterrows():
        year, field = int(pos.publication_year), str(pos.broad_field)
        candidates = [i for i in by_stratum[(year, field)] if i not in used]
        # Strict matching is intentional. A year-only fallback would retain a much
        # larger sample but would reintroduce field composition as a classifier.
        match_level = "year_field"
        if not candidates:
            records.append({"publication_year": year, "broad_field": field, "status": "unmatched"})
            continue
        choice = int(rng.choice(candidates))
        used.add(choice)
        selected.append(choice)
        records.append({"publication_year": year, "broad_field": field, "status": match_level})
    selected_controls = controls.loc[selected].copy().reset_index(drop=True)
    match_audit = pd.DataFrame(records)
    return selected_controls, match_audit


def make_models() -> dict[str, Any]:
    return {
        "Logistic regression": Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(C=1.0, max_iter=2000, class_weight=None, random_state=SEED))]),
        "Random forest": RandomForestClassifier(n_estimators=400, max_features="sqrt", min_samples_leaf=2, class_weight=None, n_jobs=-1, random_state=SEED),
        "Gradient boosting": GradientBoostingClassifier(n_estimators=250, learning_rate=0.05, max_depth=3, min_samples_leaf=5, random_state=SEED),
        "RBF SVM": Pipeline([("scale", StandardScaler()), ("model", SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, random_state=SEED))]),
    }


def probability(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    values = model.decision_function(x)
    return 1.0 / (1.0 + np.exp(-values))


def calibration_data(y: np.ndarray, p: np.ndarray, bins: int = 10) -> tuple[float, pd.DataFrame]:
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(p, edges, right=True) - 1, 0, bins - 1)
    rows = []
    ece = 0.0
    for index in range(bins):
        mask = which == index
        if not mask.any():
            continue
        mean_pred = float(np.mean(p[mask]))
        observed = float(np.mean(y[mask]))
        weight = float(mask.mean())
        ece += weight * abs(observed - mean_pred)
        rows.append({"bin": index + 1, "n": int(mask.sum()), "mean_predicted": mean_pred, "observed_rate": observed})
    return float(ece), pd.DataFrame(rows)


def evaluate(model: Any, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, meta: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    fitted = clone(model).fit(x_train, y_train)
    p = probability(fitted, x_test)
    pred = (p >= 0.5).astype(int)
    ece, calibration = calibration_data(y_test, p)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    row = {
        **meta,
        "n_train": int(len(y_train)), "n_test": int(len(y_test)),
        "train_prevalence": float(np.mean(y_train)), "test_prevalence": float(np.mean(y_test)),
        "accuracy": float(accuracy_score(y_test, pred)), "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)), "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, p)), "pr_auc": float(average_precision_score(y_test, p)),
        "log_loss": float(log_loss(y_test, p, labels=[0, 1])), "mcc": float(matthews_corrcoef(y_test, pred)),
        "brier_score": float(brier_score_loss(y_test, p)), "ece_10_bins": ece,
        "tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
    }
    calibration = calibration.assign(**meta)
    return row, calibration, p


def protocol_splits(data: pd.DataFrame, protocol: str, cutoff: int) -> list[tuple[str, np.ndarray, np.ndarray]]:
    y = data.label.to_numpy(int)
    n = len(data)
    if protocol == "random":
        train, test = train_test_split(np.arange(n), test_size=0.2, stratify=y, random_state=SEED)
        return [("Random 80/20 split", train, test)]
    if protocol == "temporal":
        train = np.flatnonzero(data.publication_year.to_numpy(int) <= cutoff)
        test = np.flatnonzero(data.publication_year.to_numpy(int) > cutoff)
        if len(np.unique(y[train])) < 2 or len(np.unique(y[test])) < 2:
            raise RuntimeError("Temporal split lacks both classes; change --temporal-cutoff or inspect matched controls.")
        return [(f"Train <= {cutoff}; test > {cutoff}", train, test)]
    if protocol == "cross_discipline":
        splits = []
        counts = data.groupby("broad_field").label.agg(["count", "sum"])
        eligible = counts[(counts["count"] >= 100) & (counts["sum"] >= 25) & ((counts["count"] - counts["sum"]) >= 25)].index.tolist()
        for field in eligible:
            test = np.flatnonzero(data.broad_field.to_numpy(str) == str(field))
            train = np.flatnonzero(data.broad_field.to_numpy(str) != str(field))
            if len(np.unique(y[train])) == 2 and len(np.unique(y[test])) == 2:
                splits.append((f"Held-out field: {field}", train, test))
        if not splits:
            raise RuntimeError("No broad discipline has at least 25 papers in each class for cross-discipline validation.")
        return splits
    raise ValueError(protocol)


def run_protocol(data: pd.DataFrame, feature_set: str, features: list[str], protocol: str, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]], list[tuple[str, Any, np.ndarray, np.ndarray]]]:
    x = data[features].to_numpy(float)
    y = data.label.to_numpy(int)
    rows, calibration_parts, curves, fitted_runs = [], [], {}, []
    split_list = protocol_splits(data, protocol, args.temporal_cutoff)
    for model_name, model in make_models().items():
        all_y, all_p = [], []
        for fold, (split_label, train, test) in enumerate(split_list, 1):
            result, calibration, p = evaluate(model, x[train], y[train], x[test], y[test], {
                "feature_set": feature_set, "protocol": protocol, "model": model_name,
                "fold": fold, "split_definition": split_label,
            })
            rows.append(result)
            calibration_parts.append(calibration)
            all_y.append(y[test]); all_p.append(p)
            fitted_runs.append((f"{protocol}|{feature_set}|{model_name}|{fold}", clone(model).fit(x[train], y[train]), x[test], y[test]))
        if len(split_list) > 1:
            # Pooled out-of-field predictions provide one transparent cross-discipline metric per model.
            yy, pp = np.concatenate(all_y), np.concatenate(all_p)
            result, calibration, _ = evaluate(model, x[split_list[0][1]], y[split_list[0][1]], x[split_list[0][2]], y[split_list[0][2]], {
                "feature_set": feature_set, "protocol": protocol, "model": model_name,
                "fold": 0, "split_definition": "Pooled held-out broad-field predictions",
            })
            # overwrite metrics from pooled predictions without refitting.
            pred = (pp >= 0.5).astype(int); ece, cal = calibration_data(yy, pp); cm = confusion_matrix(yy, pred, labels=[0, 1])
            for metric, value in {
                "n_train": np.nan, "n_test": len(yy), "train_prevalence": np.nan, "test_prevalence": yy.mean(),
                "accuracy": accuracy_score(yy,pred), "precision": precision_score(yy,pred,zero_division=0), "recall": recall_score(yy,pred,zero_division=0),
                "f1": f1_score(yy,pred,zero_division=0), "roc_auc": roc_auc_score(yy,pp), "pr_auc": average_precision_score(yy,pp),
                "log_loss": log_loss(yy,pp,labels=[0,1]), "mcc": matthews_corrcoef(yy,pred), "brier_score": brier_score_loss(yy,pp), "ece_10_bins": ece,
                "tn": cm[0,0], "fp": cm[0,1], "fn": cm[1,0], "tp": cm[1,1],
            }.items(): result[metric] = float(value) if isinstance(value, (float, np.floating)) else int(value) if pd.notna(value) else np.nan
            rows.append(result); calibration_parts.append(cal.assign(feature_set=feature_set, protocol=protocol, model=model_name, fold=0, split_definition=result["split_definition"]))
            curves[f"{protocol}|{feature_set}|{model_name}"] = (yy, pp)
        else:
            curves[f"{protocol}|{feature_set}|{model_name}"] = (all_y[0], all_p[0])
    return pd.DataFrame(rows), pd.concat(calibration_parts, ignore_index=True), curves, fitted_runs


def latex_table(results: pd.DataFrame, output: Path) -> None:
    primary = results[((results.protocol != "cross_discipline") & (results.fold == 1)) | ((results.protocol == "cross_discipline") & (results.fold == 0))].copy()
    primary = primary.sort_values(["feature_set", "protocol", "model"])
    nl = r"\\"
    lines = [r"\begin{table*}[!htbp]", r"\centering", r"\scriptsize", r"\caption{Complete predictive-model evaluation. The data are class-balanced by construction (one retracted and one non-retracted paper per selected control match). Random and temporal rows use one held-out test set; cross-discipline rows pool predictions from eligible held-out broad-field folds. PR-AUC is the precision--recall area under the curve; ECE is 10-bin expected calibration error.}", r"\label{tab:ml_complete}", r"\begin{tabular}{llrrrrrrrrr}", r"\toprule", r"Feature set & Protocol/model & Accuracy & Precision & Recall & F1 & ROC-AUC & PR-AUC & MCC & Brier & ECE " + nl, r"\midrule"]
    for _, row in primary.iterrows():
        protocol = str(row.protocol).replace("_", " ").title()
        lines.append(f"{str(row.feature_set).replace('_', ' ').title()} & {protocol}: {row.model} & {row.accuracy:.3f} & {row.precision:.3f} & {row.recall:.3f} & {row.f1:.3f} & {row.roc_auc:.3f} & {row.pr_auc:.3f} & {row.mcc:.3f} & {row.brier_score:.3f} & {row.ece_10_bins:.3f} " + nl)
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    (output / "table_step9_complete_ml_results.tex").write_text("\n".join(lines), encoding="utf-8")


def figures(results: pd.DataFrame, calibration: pd.DataFrame, curves: dict[str, tuple[np.ndarray, np.ndarray]], importance: pd.DataFrame, output: Path) -> None:
    primary = results[((results.protocol != "cross_discipline") & (results.fold == 1)) | ((results.protocol == "cross_discipline") & (results.fold == 0))].copy()
    # Protocol / AUC comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), sharey=True)
    for ax, feature_set in zip(axes, ("at_publication", "early_warning")):
        subset = primary[primary.feature_set.eq(feature_set)]
        pivot = subset.pivot(index="model", columns="protocol", values="roc_auc").reindex(list(make_models()))
        pivot.plot(kind="bar", ax=ax, color=["#2166AC", "#B2182B", "#4D9221"], width=0.78)
        ax.set_ylim(0.45, 1.0); ax.axhline(0.5, color="black", linewidth=0.8, linestyle="--")
        ax.set_title("At-publication model" if feature_set == "at_publication" else "One-year early-warning model")
        ax.set_ylabel("ROC-AUC" if feature_set == "at_publication" else "")
        ax.set_xlabel(""); ax.tick_params(axis="x", rotation=22); ax.legend(title="Protocol", fontsize=8)
    fig.suptitle("Predictive Performance Across Validation Protocols", y=1.02, fontsize=13)
    fig.tight_layout(); save_dual_figure(fig, output / "fig_step9_protocol_auc"); plt.close(fig)

    # Calibration: best random model under each feature set and same model temporal
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), sharex=True, sharey=True)
    for ax, feature_set in zip(axes, ("at_publication", "early_warning")):
        subset = primary[(primary.feature_set.eq(feature_set)) & primary.protocol.eq("random")]
        best = subset.sort_values("roc_auc", ascending=False).iloc[0].model
        for protocol, color in (("random", "#2166AC"), ("temporal", "#B2182B")):
            key = f"{protocol}|{feature_set}|{best}"
            if key not in curves: continue
            yy, pp = curves[key]
            ece, cal = calibration_data(yy, pp)
            ax.plot(cal.mean_predicted, cal.observed_rate, marker="o", color=color, label=f"{protocol.title()} (ECE={ece:.3f})")
        ax.plot([0,1],[0,1],"k--",linewidth=0.8); ax.set_title(f"{feature_set.replace('_',' ').title()}\n{best}")
        ax.set_xlabel("Mean predicted probability"); ax.grid(linestyle=":", alpha=0.4); ax.legend(fontsize=8)
    axes[0].set_ylabel("Observed retraction rate")
    fig.suptitle("Calibration on Random and Temporal Hold-Out Sets", y=1.02, fontsize=13)
    fig.tight_layout(); save_dual_figure(fig, output / "fig_step9_calibration"); plt.close(fig)

    # Relative permutation importance
    if not importance.empty:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        plot = importance.sort_values("relative_permutation_importance")
        ax.barh([FEATURE_LABELS.get(value, value) for value in plot.feature], plot.relative_permutation_importance, color="#4D9221")
        ax.set_xlabel("Relative permutation importance (sum = 1)")
        ax.set_title("Relative Feature Importance: Best Random-Split Early-Warning Model")
        ax.grid(axis="x", linestyle=":", alpha=0.4)
        fig.tight_layout(); save_dual_figure(fig, output / "fig_step9_relative_importance"); plt.close(fig)


def main() -> None:
    args = parse_args(); root = project_root(); dirs = revision_dirs(root); output = dirs["output"]
    positive = make_positive_dataset(root, args)
    cache = dirs["data"] / "step9_nonretracted_candidate_cache.jsonl"
    if args.mode == "fetch":
        fetch_negatives(positive, cache, args); return
    if args.mode == "all":
        raw_controls = fetch_negatives(positive, cache, args)
    else:
        raw_controls = read_cache(cache)
        if not raw_controls:
            raise FileNotFoundError(f"No Step 9 cache at {cache}; run with --mode all and an API key first.")
    controls, match_audit = stratified_control_sample(positive, raw_controls, args.seed)
    # Retain only positive rows successfully assigned a control, one-to-one.
    positive_order = positive.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    matched_flags = match_audit.status.ne("unmatched").to_numpy()
    matched_positive = positive_order.loc[matched_flags].reset_index(drop=True)
    if len(matched_positive) != len(controls):
        # Match audit rows preserve the shuffled positive order; this is an integrity guard.
        raise RuntimeError("Control matching produced inconsistent treated/control rows.")
    data = pd.concat([matched_positive, controls], ignore_index=True).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    if data.label.value_counts().min() < 500:
        raise RuntimeError("Too few matched examples for Step 9 model evaluation.")

    results_parts, calibration_parts, all_curves = [], [], {}
    for feature_set, features in (("at_publication", AT_PUBLICATION), ("early_warning", EARLY_WARNING)):
        for protocol in ("random", "temporal", "cross_discipline"):
            print(f"Step 9: evaluating {feature_set.replace('_', ' ')} features under {protocol.replace('_', ' ')} validation …", flush=True)
            result, calibration, curves, _ = run_protocol(data, feature_set, features, protocol, args)
            results_parts.append(result); calibration_parts.append(calibration); all_curves.update(curves)
    results = pd.concat(results_parts, ignore_index=True)
    calibration = pd.concat(calibration_parts, ignore_index=True)

    # Permutation importance uses the best early-warning random-split model, fitted to its training partition.
    split = protocol_splits(data, "random", args.temporal_cutoff)[0]
    _, train_idx, test_idx = split
    early_results = results[(results.feature_set.eq("early_warning")) & (results.protocol.eq("random")) & (results.fold.eq(1))]
    best_model_name = early_results.sort_values("roc_auc", ascending=False).iloc[0].model
    best_model = make_models()[best_model_name]
    x = data[EARLY_WARNING].to_numpy(float); y = data.label.to_numpy(int)
    fitted = clone(best_model).fit(x[train_idx], y[train_idx])
    importance_result = permutation_importance(fitted, x[test_idx], y[test_idx], scoring="roc_auc", n_repeats=20, random_state=args.seed, n_jobs=-1)
    imp = pd.DataFrame({"feature": EARLY_WARNING, "mean_auc_decrease": importance_result.importances_mean, "std_auc_decrease": importance_result.importances_std})
    imp["positive_auc_decrease"] = imp.mean_auc_decrease.clip(lower=0)
    total = imp.positive_auc_decrease.sum()
    imp["relative_permutation_importance"] = imp.positive_auc_decrease / total if total > 0 else np.nan
    imp["model"] = best_model_name; imp["protocol"] = "random"; imp["feature_set"] = "early_warning"

    # Reports
    data.to_csv(output / "step9_ml_analysis_dataset.csv", index=False)
    match_audit.to_csv(output / "step9_control_matching_audit.csv", index=False)
    results.to_csv(output / "step9_complete_model_results.csv", index=False)
    calibration.to_csv(output / "step9_calibration_bins.csv", index=False)
    imp.to_csv(output / "step9_relative_permutation_importance.csv", index=False)
    latex_table(results, output)
    figures(results, calibration, all_curves, imp, output)
    write_json(output / "step9_ml_summary.json", {
        "created_at": now_utc(), "study_design": "Retrospective binary classification of retracted versus non-retracted research articles.",
        "interpretation_limit": "Model results do not estimate causal determinants of retraction and should not be interpreted as a deployment-ready screening system.",
        "positive_eligible": int(len(positive)), "matched_pairs": int(len(matched_positive)), "analysis_rows": int(len(data)),
        "class_prevalence": float(data.label.mean()),         "control_matching": "One-to-one sampling without replacement using exact publication-year and broad-field matches only; unmatched retracted papers are excluded and recorded in step9_control_matching_audit.csv.",

        "feature_sets": {"at_publication": AT_PUBLICATION, "early_warning_year1": EARLY_WARNING},
        "excluded_features": ["publication year", "retraction year", "total citations", "post-retraction citations", "citation-network features", "author-level historical retraction counts"],
        "validation_protocols": {"random": "stratified 80/20 split", "temporal": f"train publication year <= {args.temporal_cutoff}; test > {args.temporal_cutoff}", "cross_discipline": "leave-one-eligible-broad-field-out; pooled held-out predictions"},
        "hyperparameter_policy": "Four model configurations were prespecified and not tuned on the held-out evaluation sets. Exact parameters are stored in the script and model configuration below.",
        "models": {name: str(model) for name, model in make_models().items()},
        "outputs": ["step9_complete_model_results.csv", "step9_calibration_bins.csv", "step9_control_matching_audit.csv", "step9_relative_permutation_importance.csv", "table_step9_complete_ml_results.tex", "fig_step9_protocol_auc.png/.pdf", "fig_step9_calibration.png/.pdf", "fig_step9_relative_importance.png/.pdf"],
    })
    print(f"Step 9 complete: {len(matched_positive):,} matched retracted/non-retracted pairs; {len(results)} evaluation rows.")


if __name__ == "__main__":
    main()
