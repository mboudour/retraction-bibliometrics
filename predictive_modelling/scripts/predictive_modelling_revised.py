"""
Predictive Modelling — WP5 Revision
=====================================
Adds to the original analysis:
  1. Temporal train/test split (train ≤2019, test 2020–2024)
  2. Cross-discipline validation (train on top discipline, test on others)
  3. Ablation study (remove one feature at a time)
  4. SHAP analysis (XGBoost)

Outputs figures and CSV tables to predictive_modelling/output/
"""

import warnings, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, matthews_corrcoef, confusion_matrix,
    RocCurveDisplay
)
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import shap

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = Path("/home/ubuntu/final_review")
DATA = BASE / "descriptive_statistics/output/merged_dataset.csv"
OUT  = BASE / "predictive_modelling/output"
OUT.mkdir(parents=True, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

# ── Load and prepare data ────────────────────────────────────────────────────
print("Loading data …")
df = pd.read_csv(DATA, low_memory=False)
df["publication_year"] = pd.to_numeric(df["publication_year"], errors="coerce")
df["retraction_year"]  = pd.to_numeric(df["retraction_year"],  errors="coerce")

# Parse early citation velocity (citations in first year)
def early_velocity(row):
    try:
        cby = json.loads(row["counts_by_year_json"])
        pub_yr = int(row["publication_year"])
        for e in cby:
            if int(e["year"]) == pub_yr + 1:
                return int(e["cited_by_count"])
        return 0
    except Exception:
        return 0

df["early_velocity"] = df.apply(early_velocity, axis=1)

# Feature engineering
df["title_length"]   = df["title"].fillna("").str.len()
df["n_authors_log"]  = np.log1p(df["authorships_count"].fillna(0))
df["n_refs_log"]     = np.log1p(df["referenced_works_count"].fillna(0))
df["is_oa_bin"]      = df["is_oa"].fillna(False).astype(int)
df["has_abstract"]   = df["has_abstract"].fillna(False).astype(int)
df["early_vel_log"]  = np.log1p(df["early_velocity"])

FEATURES = ["title_length","n_authors_log","n_refs_log",
            "is_oa_bin","has_abstract","early_vel_log"]
FEAT_LABELS = ["Title Length","log(Authors+1)","log(Refs+1)",
               "Open Access","Has Abstract","log(Early Velocity+1)"]

# Build balanced dataset: all retracted + matched non-retracted
retracted = df[df["is_retracted"] == True].dropna(subset=FEATURES)
non_ret   = df[df["is_retracted"] == False].dropna(subset=FEATURES)
np.random.seed(42)
non_sample = non_ret.sample(n=min(len(retracted), len(non_ret)), random_state=42)
balanced = pd.concat([retracted, non_sample]).sample(frac=1, random_state=42)
balanced["label"] = (balanced["is_retracted"] == True).astype(int)

X = balanced[FEATURES].values
y = balanced["label"].values
years = balanced["publication_year"].fillna(0).values
disciplines = balanced["top_concept"].fillna("Unknown").values

print(f"  Dataset: {len(balanced):,} papers ({y.sum():,} retracted)")

# ── Helper: evaluate model ────────────────────────────────────────────────────
def evaluate(model, X_tr, y_tr, X_te, y_te, name):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:,1] if hasattr(model,"predict_proba") else \
             model.decision_function(X_te)
    return {
        "model": name,
        "accuracy":  round(accuracy_score(y_te, y_pred), 4),
        "precision": round(precision_score(y_te, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_te, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_te, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_te, y_prob), 4),
        "log_loss":  round(log_loss(y_te, y_prob), 4),
        "mcc":       round(matthews_corrcoef(y_te, y_pred), 4),
    }

# ── Models ────────────────────────────────────────────────────────────────────
def make_models():
    return {
        "Logistic Regression": Pipeline([
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42, n_jobs=-1),
        "SVM (RBF)": Pipeline([
            ("sc", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=42))
        ]),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, n_jobs=-1, verbosity=0),
    }

# ── 1. Random 80/20 split (original) ─────────────────────────────────────────
print("\n1. Random 80/20 split …")
from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                            stratify=y, random_state=42)
results_random = []
for name, model in make_models().items():
    r = evaluate(model, X_tr, y_tr, X_te, y_te, name)
    results_random.append(r)
    print(f"  {name}: AUC={r['roc_auc']:.4f}  F1={r['f1']:.4f}")

df_random = pd.DataFrame(results_random)
df_random["split"] = "random_80_20"
df_random.to_csv(OUT / "wp5_results_random_split.csv", index=False)

# ── 2. Temporal split (train ≤2019, test 2020–2024) ──────────────────────────
print("\n2. Temporal split (train ≤2019, test 2020–2024) …")
mask_train = years <= 2019
mask_test  = (years >= 2020) & (years <= 2024)
X_tr_t, y_tr_t = X[mask_train], y[mask_train]
X_te_t, y_te_t = X[mask_test],  y[mask_test]
print(f"  Train: {mask_train.sum():,}  |  Test: {mask_test.sum():,}")

results_temporal = []
if mask_train.sum() > 100 and mask_test.sum() > 100:
    for name, model in make_models().items():
        r = evaluate(model, X_tr_t, y_tr_t, X_te_t, y_te_t, name)
        results_temporal.append(r)
        print(f"  {name}: AUC={r['roc_auc']:.4f}  F1={r['f1']:.4f}")
    df_temporal = pd.DataFrame(results_temporal)
    df_temporal["split"] = "temporal_2019_2024"
    df_temporal.to_csv(OUT / "wp5_results_temporal_split.csv", index=False)
else:
    print("  Insufficient data for temporal split")
    df_temporal = pd.DataFrame()

# ── 3. Cross-discipline validation ───────────────────────────────────────────
print("\n3. Cross-discipline validation …")
top_discs = pd.Series(disciplines).value_counts().head(5).index.tolist()
results_disc = []
for train_disc in top_discs:
    mask_tr = disciplines == train_disc
    mask_te = disciplines != train_disc
    if mask_tr.sum() < 100 or mask_te.sum() < 100:
        continue
    # Use XGBoost only for speed
    model = xgb.XGBClassifier(n_estimators=100, max_depth=5,
                               use_label_encoder=False, eval_metric="logloss",
                               random_state=42, n_jobs=-1, verbosity=0)
    r = evaluate(model, X[mask_tr], y[mask_tr], X[mask_te], y[mask_te],
                 f"XGB train={train_disc[:20]}")
    r["train_discipline"] = train_disc
    r["n_train"] = mask_tr.sum()
    r["n_test"]  = mask_te.sum()
    results_disc.append(r)
    print(f"  Train on {train_disc[:25]}: AUC={r['roc_auc']:.4f}")

df_disc = pd.DataFrame(results_disc)
df_disc.to_csv(OUT / "wp5_results_discipline_transfer.csv", index=False)

# ── 4. Ablation study ────────────────────────────────────────────────────────
print("\n4. Ablation study …")
ablation_rows = []
# Baseline (all features)
model_base = xgb.XGBClassifier(n_estimators=200, max_depth=6,
                                use_label_encoder=False, eval_metric="logloss",
                                random_state=42, n_jobs=-1, verbosity=0)
r_base = evaluate(model_base, X_tr, y_tr, X_te, y_te, "All features")
r_base["removed_feature"] = "None (baseline)"
ablation_rows.append(r_base)

for i, feat in enumerate(FEATURES):
    feat_mask = [j for j in range(len(FEATURES)) if j != i]
    model_abl = xgb.XGBClassifier(n_estimators=200, max_depth=6,
                                   use_label_encoder=False, eval_metric="logloss",
                                   random_state=42, n_jobs=-1, verbosity=0)
    r = evaluate(model_abl, X_tr[:, feat_mask], y_tr,
                 X_te[:, feat_mask], y_te, f"−{FEAT_LABELS[i]}")
    r["removed_feature"] = FEAT_LABELS[i]
    ablation_rows.append(r)
    print(f"  −{FEAT_LABELS[i]}: AUC={r['roc_auc']:.4f}  (Δ={r['roc_auc']-r_base['roc_auc']:+.4f})")

df_ablation = pd.DataFrame(ablation_rows)
df_ablation.to_csv(OUT / "wp5_ablation_study.csv", index=False)

# Ablation figure
fig, ax = plt.subplots(figsize=(8, 5))
base_auc = df_ablation.iloc[0]["roc_auc"]
sub = df_ablation.iloc[1:].sort_values("roc_auc")
delta = sub["roc_auc"] - base_auc
colors = ["#d62728" if d < 0 else "#2ca02c" for d in delta]
ax.barh(sub["removed_feature"], delta, color=colors, height=0.6)
ax.axvline(0, color="black", linewidth=1)
ax.set_xlabel("ΔAUC vs. baseline (all features)", fontsize=11)
ax.set_title(f"Ablation Study: Impact of Removing Each Feature\n(Baseline AUC = {base_auc:.4f})",
             fontsize=11, fontweight="bold", pad=10)
plt.tight_layout()
fig.savefig(OUT / "fig_ablation_study.png", bbox_inches="tight")
plt.close(fig)
print("  Saved fig_ablation_study.png")

# ── 5. SHAP analysis ─────────────────────────────────────────────────────────
print("\n5. SHAP analysis …")
xgb_model = xgb.XGBClassifier(n_estimators=200, max_depth=6,
                               use_label_encoder=False, eval_metric="logloss",
                               random_state=42, n_jobs=-1, verbosity=0)
xgb_model.fit(X_tr, y_tr)

# Compute SHAP values on test set (subsample for speed)
X_te_shap = X_te[:2000] if len(X_te) > 2000 else X_te
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_te_shap)

# SHAP summary bar plot
fig, ax = plt.subplots(figsize=(8, 5))
mean_abs_shap = np.abs(shap_values).mean(axis=0)
order = np.argsort(mean_abs_shap)
colors_shap = ["#1f77b4"] * len(FEAT_LABELS)
ax.barh([FEAT_LABELS[i] for i in order], mean_abs_shap[order],
        color=[colors_shap[i] for i in order], height=0.6)
ax.set_xlabel("Mean |SHAP value|", fontsize=11)
ax.set_title("SHAP Feature Importance (XGBoost)\nMean absolute SHAP value across test set",
             fontsize=11, fontweight="bold", pad=10)
plt.tight_layout()
fig.savefig(OUT / "fig_shap_importance.png", bbox_inches="tight")
plt.close(fig)
print("  Saved fig_shap_importance.png")

# ── 6. Comparison figure: random vs temporal AUC ─────────────────────────────
if not df_temporal.empty:
    print("\n6. AUC comparison: random vs temporal split …")
    models_list = df_random["model"].tolist()
    auc_rand = df_random.set_index("model")["roc_auc"]
    auc_temp = df_temporal.set_index("model")["roc_auc"]

    x = np.arange(len(models_list))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width/2, [auc_rand.get(m, 0) for m in models_list],
                   width, label="Random 80/20 split", color="#1f77b4")
    bars2 = ax.bar(x + width/2, [auc_temp.get(m, 0) for m in models_list],
                   width, label="Temporal split (≤2019 → 2020–2024)", color="#ff7f0e")
    ax.set_ylabel("ROC-AUC", fontsize=11)
    ax.set_title("Temporal Validation: AUC Degradation Under Concept Drift",
                 fontsize=11, fontweight="bold", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(models_list, rotation=10, ha="right")
    ax.set_ylim(0.5, 1.0)
    ax.legend(fontsize=9)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1, alpha=0.5)
    plt.tight_layout()
    fig.savefig(OUT / "fig_temporal_validation.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig_temporal_validation.png")

# ── 7. Combined results table ─────────────────────────────────────────────────
all_results = pd.concat([df_random,
                          df_temporal if not df_temporal.empty else pd.DataFrame()],
                         ignore_index=True)
all_results.to_csv(OUT / "wp5_all_results.csv", index=False)
print("\n  Saved wp5_all_results.csv")

# Print summary table
print("\n=== Summary: Random 80/20 Split ===")
print(df_random[["model","accuracy","precision","recall","f1","roc_auc","log_loss","mcc"]].to_string(index=False))
if not df_temporal.empty:
    print("\n=== Summary: Temporal Split ===")
    print(df_temporal[["model","accuracy","precision","recall","f1","roc_auc","log_loss","mcc"]].to_string(index=False))

print("\nDone — revised predictive modelling complete.")
