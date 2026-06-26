#!/usr/bin/env python3
"""
WP5 (v5 — leakage-free, no structural features): Predictive Modelling
======================================================================
Trains four classifiers to distinguish retracted from non-retracted papers
using ONLY features that are genuinely available before or shortly after
publication — no structural network features (which are only computed for
the retracted corpus and would trivially separate the two classes).

Feature set (6 features):
  Pre-publication:
    log_n_authors       log(1 + number of authors)
    log_n_references    log(1 + number of references)
    is_oa               open-access flag (0/1)
    has_abstract        abstract present (0/1)
    title_length        number of characters in title
  Early post-publication:
    log_early_cites     log(1 + citations in years 1-2)

Negative class: fetched live from OpenAlex API (non-retracted papers),
cached in output/wp5_neg_class_cache.jsonl to avoid re-fetching.

Models: Logistic Regression, Random Forest, Gradient Boosting, SVM (RBF)
Evaluation: 5-fold stratified cross-validation
Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, Log-Loss, MCC

Outputs (in wp5_prediction/output/):
    wp5_features.csv
    wp5_model_evaluation.csv
    wp5_feature_importance.csv
    fig_roc_curves.png
    fig_feature_importance.png
    fig_confusion_matrices.png
"""

import sys, os, json, time, requests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "shared"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble      import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm           import SVC
from sklearn.pipeline      import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.metrics       import (roc_auc_score, roc_curve,
                                   precision_score, recall_score,
                                   f1_score, accuracy_score,
                                   log_loss, matthews_corrcoef,
                                   confusion_matrix)
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

from utils import load_retraction_watch, load_openalex, load_openalex_jsonl

OUT = Path(__file__).resolve().parent.parent / "output"
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#2563EB"; RED = "#DC2626"; GRAY = "#6B7280"
plt.rcParams.update({
    "figure.figsize": (10, 5),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "font.family": "DejaVu Sans",
})

def norm_doi(d):
    return str(d).replace("https://doi.org/", "").lower().strip() if d else ""

def parse_cby(raw):
    """Parse counts_by_year into {year: count} dict."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return {}
    if not isinstance(raw, list):
        return {}
    out = {}
    for d in raw:
        try:
            out[int(d["year"])] = int(d["cited_by_count"])
        except Exception:
            pass
    return out

def extract_features(rec, pub_year=None, label=1):
    """Extract leakage-free features from an OpenAlex record dict."""
    if pub_year is None:
        pub_year = rec.get("publication_year") or 0
    try:
        pub_year = int(pub_year)
    except Exception:
        pub_year = 0

    authors = rec.get("authorships") or []
    n_authors = len(authors)

    refs = rec.get("referenced_works") or []
    n_refs = len(refs)

    is_oa = 1 if rec.get("open_access", {}).get("is_oa") else 0

    abstract = rec.get("abstract_inverted_index") or rec.get("abstract") or ""
    has_abstract = 1 if abstract else 0

    title = rec.get("title") or rec.get("display_name") or ""
    title_length = len(str(title))

    # Early citations: years 1 and 2 after publication
    cby = parse_cby(rec.get("counts_by_year") or [])
    early_cites = sum(cby.get(pub_year + i, 0) for i in [1, 2])

    return {
        "log_n_authors":    np.log1p(n_authors),
        "log_n_references": np.log1p(n_refs),
        "is_oa":            is_oa,
        "has_abstract":     has_abstract,
        "title_length":     title_length,
        "log_early_cites":  np.log1p(early_cites),
        "label":            label,
    }

# ---------------------------------------------------------------------------
# Load positive class (retracted papers)
# ---------------------------------------------------------------------------
print("Loading retracted papers …")
rw      = load_retraction_watch()
oa      = load_openalex()
records = load_openalex_jsonl()

retracted_ids = set(oa["openalex_id"].dropna().tolist())
print(f"  Retracted OpenAlex IDs: {len(retracted_ids):,}")

# Build pub_year lookup from OA CSV
pub_year_map = dict(zip(oa["openalex_id"].fillna(""),
                        pd.to_numeric(oa["publication_year"], errors="coerce").fillna(0).astype(int)))

pos_rows = []
for rec in records:
    oa_id = rec.get("id", "")
    pub_yr = pub_year_map.get(oa_id, rec.get("publication_year") or 0)
    pos_rows.append(extract_features(rec, pub_year=pub_yr, label=1))

pos_df = pd.DataFrame(pos_rows)
print(f"  Positive samples: {len(pos_df):,}")

# ---------------------------------------------------------------------------
# Load / fetch negative class (non-retracted papers)
# ---------------------------------------------------------------------------
NEG_CACHE = OUT / "wp5_neg_class_cache.jsonl"
N_NEG_TARGET = min(len(pos_df) * 3, 60_000)

def fetch_negatives(n_target, cache_path, email="retraction.study@example.com"):
    """Fetch non-retracted papers from OpenAlex API with cursor pagination."""
    neg_records = []
    if cache_path.exists():
        print(f"  Loading negatives from cache: {cache_path}")
        with open(cache_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        neg_records.append(json.loads(line))
                    except Exception:
                        pass
        if len(neg_records) >= n_target:
            print(f"  Cache has {len(neg_records):,} records — sufficient.")
            return neg_records[:n_target]
        print(f"  Cache has {len(neg_records):,} records — fetching more …")

    print(f"  Fetching up to {n_target:,} non-retracted papers from OpenAlex …")
    per_page = 200
    cursor = "*"
    session = requests.Session()
    session.headers.update({"User-Agent": f"RetractStudy/1.0 (mailto:{email})"})

    with open(cache_path, "a") as fout:
        while len(neg_records) < n_target:
            params = {
                "filter":   "is_retracted:false,type:article,has_doi:true",
                "select":   ("id,doi,publication_year,authorships,referenced_works,"
                             "open_access,abstract_inverted_index,title,cited_by_count,"
                             "counts_by_year"),
                "per-page": per_page,
                "cursor":   cursor,
                "mailto":   email,
            }
            for attempt in range(5):
                try:
                    r = session.get("https://api.openalex.org/works",
                                    params=params, timeout=30)
                    if r.status_code == 429:
                        time.sleep(10 * (attempt + 1))
                        continue
                    r.raise_for_status()
                    data = r.json()
                    break
                except Exception as e:
                    if attempt == 4:
                        print(f"  API error after 5 attempts: {e}")
                        return neg_records
                    time.sleep(5)

            results = data.get("results", [])
            if not results:
                break

            for rec in results:
                oa_id = rec.get("id", "")
                if oa_id not in retracted_ids:
                    neg_records.append(rec)
                    fout.write(json.dumps(rec) + "\n")

            cursor = data.get("meta", {}).get("next_cursor")
            if not cursor:
                break

            fetched = len(neg_records)
            if fetched % 5000 < per_page:
                print(f"    {fetched:,} / {n_target:,} negatives fetched …")
            time.sleep(0.12)

    print(f"  Total negatives fetched: {len(neg_records):,}")
    return neg_records[:n_target]

neg_records = fetch_negatives(N_NEG_TARGET, NEG_CACHE)

neg_rows = []
for rec in neg_records:
    pub_yr = rec.get("publication_year") or 0
    neg_rows.append(extract_features(rec, pub_year=pub_yr, label=0))

neg_df = pd.DataFrame(neg_rows)
print(f"  Negative samples: {len(neg_df):,}")

# ---------------------------------------------------------------------------
# Build balanced dataset
# ---------------------------------------------------------------------------
FEATS = ["log_n_authors", "log_n_references", "is_oa",
         "has_abstract", "title_length", "log_early_cites"]

pos_clean = pos_df.dropna(subset=FEATS)
neg_clean = neg_df.dropna(subset=FEATS)

# 1:3 balance
n_pos = len(pos_clean)
n_neg = min(n_pos * 3, len(neg_clean))
neg_sample = neg_clean.sample(n_neg, random_state=42)

balanced = pd.concat([pos_clean, neg_sample], ignore_index=True).sample(
    frac=1, random_state=42)

X = balanced[FEATS].values
y = balanced["label"].values
print(f"\nBalanced dataset: {n_pos:,} retracted + {n_neg:,} non-retracted = {len(balanced):,} total")

balanced[FEATS + ["label"]].to_csv(OUT / "wp5_features.csv", index=False)
print("  Saved wp5_features.csv")

# ---------------------------------------------------------------------------
# Define models
# ---------------------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

if XGB_AVAILABLE:
    gb_model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                              use_label_encoder=False, eval_metric="logloss",
                              random_state=42, n_jobs=-1)
    gb_name = "XGBoost"
else:
    gb_model = GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                          learning_rate=0.1, random_state=42)
    gb_name = "Gradient Boosting"

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)),
    ]),
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
    ]),
    gb_name: Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    gb_model),
    ]),
    "SVM (RBF)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    SVC(kernel="rbf", probability=True, random_state=42)),
    ]),
}

# ---------------------------------------------------------------------------
# Evaluate models
# ---------------------------------------------------------------------------
print("\nTraining and evaluating models (5-fold CV) …")
eval_rows   = []
roc_data    = {}
cm_data     = {}

for name, model in models.items():
    print(f"  {name} …")
    y_prob = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc  = roc_auc_score(y, y_prob)
    acc  = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec  = recall_score(y, y_pred, zero_division=0)
    f1   = f1_score(y, y_pred, zero_division=0)
    ll   = log_loss(y, y_prob)
    mcc  = matthews_corrcoef(y, y_pred)

    eval_rows.append({
        "Model": name, "Accuracy": round(acc, 4),
        "Precision": round(prec, 4), "Recall": round(rec, 4),
        "F1": round(f1, 4), "ROC-AUC": round(auc, 4),
        "Log-Loss": round(ll, 4), "MCC": round(mcc, 4),
    })
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_data[name] = (fpr, tpr, auc)
    cm_data[name]  = (confusion_matrix(y, y_pred), auc)
    print(f"    AUC={auc:.3f}  F1={f1:.3f}  MCC={mcc:.3f}")

eval_df = pd.DataFrame(eval_rows)
eval_df.to_csv(OUT / "wp5_model_evaluation.csv", index=False)
print("\n  Saved wp5_model_evaluation.csv")
print(eval_df.to_string(index=False))

# ---------------------------------------------------------------------------
# Feature importance (Random Forest)
# ---------------------------------------------------------------------------
print("\nFitting Random Forest for feature importance …")
rf_pipe = models["Random Forest"]
rf_pipe.fit(X, y)
importances = rf_pipe.named_steps["clf"].feature_importances_

feat_imp = pd.DataFrame({
    "Feature":    FEATS,
    "Importance": importances,
    "Color":      [BLUE, BLUE, BLUE, BLUE, BLUE, "#16A34A"],
}).sort_values("Importance", ascending=True)
feat_imp.to_csv(OUT / "wp5_feature_importance.csv", index=False)

# ---------------------------------------------------------------------------
# Figure 1: ROC curves
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))
colors = [BLUE, RED, "#16A34A", "#7C3AED"]
for (name, (fpr, tpr, auc)), col in zip(roc_data.items(), colors):
    ax.plot(fpr, tpr, color=col, linewidth=2,
            label=f"{name}  (AUC = {auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", linewidth=1)
ax.set_title("ROC Curves — Retraction Early-Warning Model\n"
             "(5-Fold Cross-Validation, Pre-Publication Features Only)")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(fontsize=9, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "fig_roc_curves.png", dpi=150)
plt.close(fig)
print("  Saved fig_roc_curves.png")

# ---------------------------------------------------------------------------
# Figure 2: Feature importance
# ---------------------------------------------------------------------------
FEAT_LABELS = {
    "log_n_authors":    "No. of Authors (log)",
    "log_n_references": "No. of References (log)",
    "is_oa":            "Open Access",
    "has_abstract":     "Has Abstract",
    "title_length":     "Title Length",
    "log_early_cites":  "Early Citations yrs 1-2 (log)",
}
feat_imp["Label"] = feat_imp["Feature"].map(FEAT_LABELS)

fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.barh(feat_imp["Label"], feat_imp["Importance"],
               color=feat_imp["Color"], edgecolor="white")
for bar, val in zip(bars, feat_imp["Importance"]):
    ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=9)
ax.set_title("Feature Importance (Random Forest)\n"
             "(Blue = pre-publication  |  Green = early post-pub)")
ax.set_xlabel("Mean Decrease in Impurity")
fig.tight_layout()
fig.savefig(OUT / "fig_feature_importance.png", dpi=150)
plt.close(fig)
print("  Saved fig_feature_importance.png")

# ---------------------------------------------------------------------------
# Figure 3: Confusion matrices
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, len(cm_data), figsize=(4 * len(cm_data), 4))
if len(cm_data) == 1:
    axes = [axes]
for ax, (name, (cm, auc)) in zip(axes, cm_data.items()):
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(f"{name}\nAUC={auc:.3f}", fontsize=10)
    tick_marks = [0, 1]
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(["Non-retracted", "Retracted"], fontsize=8)
    ax.set_yticklabels(["Non-retracted", "Retracted"], fontsize=8)
    ax.set_xlabel("Predicted label", fontsize=8)
    ax.set_ylabel("True label", fontsize=8)
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "#2563EB",
                    fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "fig_confusion_matrices.png", dpi=150)
plt.close(fig)
print("  Saved fig_confusion_matrices.png")

print("\n=== WP5 Complete ===")
print(f"  Features used: {FEATS}")
print(f"  Positive (retracted):     {n_pos:,}")
print(f"  Negative (non-retracted): {n_neg:,}")
print(f"  Best AUC: {max(r['ROC-AUC'] for r in eval_rows):.3f}")
print("\nAll outputs in wp5_prediction/output/")
