"""
04_model.py
===========
Trains three model variants and scores all NJ zip codes.

Models
------
  Model 1 — Desert classifier
      Targets : is_food_desert  (zip-level binary)
                is_desert_5mi   (5-mile buffer — primary)
      Inputs  : census demographics only (no store counts, no proximity)
      Outputs : desert_probability, predicted_desert, predicted_desert_tuned

  Model 2 — Access typology classifier
      Target  : trains on 4 classes (NaN rows get .dropna()'d before the split).
      Inputs  : store counts + proximity + demographics
      Outputs : predicted_typology + per-class probabilities

  Model 3 — Health outcome regressors
      Targets : HEALTH_TARGETS list has 20 outcomes.
      Inputs  : food environment features + demographic controls
      Outputs : predicted_{outcome} columns + feature importance CSVs

Validation
----------
  - Stratified k-fold CV (5 folds)
  - Bootstrapped confusion matrix (2 000 resamples, 95 % CI)
  - Leave-one-county-out spatial CV
  - Threshold tuning table (precision / recall / F1 vs threshold)

Outputs
-------
  data/nj_zip_scores1.csv          — all zips scored (primary deliverable)
  data/model_feature_importance.csv
  data/health_importance_{outcome}.csv
  data/pipeline_metadata.json
  data/bootstrap_metrics.csv
  data/threshold_tuning.csv
  data/spatial_cv_results.csv
"""

from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
from datetime import date
from scipy import stats
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from pipeline_utils import section, normalize_zip

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

ROOT_DIR  = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT_DIR / "data"
INPUT     = DATA_DIR / "nj_zip_features_v5.csv"  # output of 03_features.py
COUNTY_XW = DATA_DIR / "nj_zip_complete.csv"

OUTPUT_SCORES  = DATA_DIR / "nj_zip_scores_1.csv"
OUTPUT_FI      = DATA_DIR / "model_feature_importance.csv"
OUTPUT_META    = DATA_DIR / "pipeline_metadata.json"
OUTPUT_BOOT    = DATA_DIR / "bootstrap_metrics.csv"
OUTPUT_THRESH  = DATA_DIR / "threshold_tuning.csv"
OUTPUT_SCV     = DATA_DIR / "spatial_cv_results.csv"

RANDOM_SEED = 42
N_BOOTSTRAP = 2_000


# section() and normalize_zip() now live in pipeline_utils.py (imported above)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
section("LOAD")

df = pd.read_csv(INPUT, dtype={"zip": str})
print(f"Loaded: {df.shape[0]} zips × {df.shape[1]} columns")

# Attach county for spatial CV
df["zip"] = normalize_zip(df["zip"])

if "county_fips" in df.columns:
    print(f"  county_fips already present: {df['county_fips'].nunique()} counties")
    df["county_fips"] = df["county_fips"].fillna("Unknown").astype(str)
elif COUNTY_XW.exists():
    xwalk = pd.read_csv(COUNTY_XW, dtype={"zip": str})
    xwalk["zip"] = normalize_zip(xwalk["zip"])
    df = df.merge(xwalk[["zip", "county_fips"]], on="zip", how="left")
    print(f"  county_fips matched: {df['county_fips'].notna().sum()} / {len(df)}")
    df["county_fips"] = df["county_fips"].fillna("Unknown").astype(str)
    print(f"  County crosswalk joined: {df['county_fips'].nunique()} counties")
else:
    df["county_fips"] = "Unknown"
    print(f"[WARNING] county crosswalk not found at {COUNTY_XW} — spatial CV skipped")
# ─────────────────────────────────────────────────────────────────────────────
# FEATURE SETS
# ─────────────────────────────────────────────────────────────────────────────
section("FEATURE SETS")

# ── Model 1: Desert — census demographics only ────────────────────────────
# NO store counts, NO proximity features (those define the target → leakage)
DESERT_FEATURES = [
    "median_income",
    "pct_poverty",
    "pct_snap",
    "pct_transit",
    "pct_no_vehicle",
    "pct_college",
    "pct_elderly",
]

# Leakage guard for desert model
PROXIMITY_COLS = ["nearest_supermarket_miles", "supermarkets_within_5mi"]
leaked = [f for f in PROXIMITY_COLS if f in DESERT_FEATURES]
assert not leaked, f"DATA LEAKAGE: proximity features in DESERT_FEATURES: {leaked}"

DESERT_SWAMP_OUTCOMES = {
    "Coronary Heart Disease % (Adults)":      "coronary_heart_disease",
    "High Blood Pressure % (Adults)":         "high_blood_pressure",
    "Depression % (Adults)":                  "depression",
    "High Cholesterol % (Adults)":            "high_cholesterol",
    "Diabetes % (Adults)":                    "diabetes",
    "Teeth Lost % (Adults 65+)":              "tooth_loss",
    "Obesity % (Adults)":                     "obesity",
    "Current Smoking % (Adults)":             "smoking",
    "COPD % (Adults)":                        "copd",
    "Stroke % (Adults)":                      "stroke",
    "Physical Inactivity % (Adults)":         "physical_inactivity",
    "Poor Mental Health % (Adults)":          "poor_mental_health",
    "Poor Physical Health % (Adults)":        "poor_physical_health",
    "Short Sleep Duration % (Adults)":        "short_sleep",
    "Current Asthma % (Adults)":              "asthma",
    "Food Insecurity % (Adults)":             "food_insecurity",
    "Housing Insecurity % (Adults)":          "housing_insecurity",
    "Social Isolation/Loneliness % (Adults)": "social_isolation",
    "Any Disability % (Adults)":              "any_disability",
    "Lack of Transportation % (Adults)":      "lack_of_transportation",
}

# ── Model 2: Typology — store counts + proximity + demographics ───────────
TYPOLOGY_FEATURES = [
    "supermarket", "fast_food", "convenience", "restaurant", "dollar_store",
    "nearest_supermarket_miles", "supermarkets_within_5mi",
    "snap_stores_per_10k", "snap_supermarkets_per_10k", "wic_stores_per_10k",
    "median_income", "pop_density", "pct_poverty", "pct_no_vehicle",
    "pct_elderly", "pct_transit",
]

# ── Model 3: Health outcomes — food environment + demographic controls ─────
# Never include one health outcome as a predictor of another
HEALTH_FEATURES = [
    "rfei", "mrfei", "nj_swamp_score",
    "nearest_supermarket_miles", "supermarkets_within_5mi",
    "snap_quality_ratio", "dollar_store_ratio",
    "snap_stores_per_10k", "wic_stores_per_10k",
    "pct_poverty", "median_income", "pct_no_vehicle",
    "pct_elderly", "pop_density",
]

HEALTH_TARGETS = [
    "Obesity % (Adults)",
    "Diabetes % (Adults)",
    "Food Insecurity % (Adults)",
    "Coronary Heart Disease % (Adults)",
    "High Blood Pressure % (Adults)",
    "Depression % (Adults)",
    "High Cholesterol % (Adults)",
    "Current Smoking % (Adults)",
    "COPD % (Adults)",
    "Stroke % (Adults)",
    "Physical Inactivity % (Adults)",
    "Poor Mental Health % (Adults)",
    "Poor Physical Health % (Adults)",
    "Short Sleep Duration % (Adults)",
    "Current Asthma % (Adults)",
    "Teeth Lost % (Adults 65+)",
    "Any Disability % (Adults)",
    "Social Isolation/Loneliness % (Adults)",
    "Housing Insecurity % (Adults)",
    "Lack of Transportation % (Adults)",
]

# Leakage guard for desert model
PROXIMITY_COLS = ["nearest_supermarket_miles", "supermarkets_within_5mi"]
leaked = [f for f in PROXIMITY_COLS if f in DESERT_FEATURES]
assert not leaked, f"DATA LEAKAGE: proximity features in DESERT_FEATURES: {leaked}"

# Filter to columns that actually exist
DESERT_FEATURES   = [f for f in DESERT_FEATURES   if f in df.columns]
TYPOLOGY_FEATURES = [f for f in TYPOLOGY_FEATURES if f in df.columns]
HEALTH_FEATURES   = [f for f in HEALTH_FEATURES   if f in df.columns]

print(f"Desert features   : {len(DESERT_FEATURES)}")
print(f"Typology features : {len(TYPOLOGY_FEATURES)}")
print(f"Health features   : {len(HEALTH_FEATURES)}")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_sample_weights(y: pd.Series) -> np.ndarray:
    """Per-sample weights that balance classes (for GBM which has no class_weight)."""
    counts = y.value_counts()
    total  = len(y)
    return y.map(lambda c: total / (2 * counts[c])).values


def bootstrap_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                       y_proba: np.ndarray,
                       n: int = N_BOOTSTRAP,
                       seed: int = RANDOM_SEED) -> dict:
    """
    Bootstrap 95 % CIs around accuracy, precision, recall, F1, AUC.
    Returns dict of {metric: (mean, lo, hi)}.
    """
    rng = np.random.default_rng(seed)
    store = {k: [] for k in [
        "accuracy", "precision_desert", "recall_desert", "f1_desert",
        "precision_access", "recall_access", "f1_access", "auc",
        "tn", "fp", "fn", "tp",
    ]}

    n_test = len(y_true)
    for _ in range(n):
        idx = rng.integers(0, n_test, size=n_test)
        yt  = y_true[idx]
        yp  = y_pred[idx]
        ypr = y_proba[idx]
        if len(np.unique(yt)) < 2:
            continue

        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        store["tn"].append(tn); store["fp"].append(fp)
        store["fn"].append(fn); store["tp"].append(tp)

        acc  = (tp + tn) / (tp + tn + fp + fn)
        p_d  = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        r_d  = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        p_a  = tn / (tn + fn) if (tn + fn) > 0 else np.nan
        r_a  = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        f1_d = 2 * p_d * r_d / (p_d + r_d) if (p_d and r_d and (p_d + r_d) > 0) else np.nan
        f1_a = 2 * p_a * r_a / (p_a + r_a) if (p_a and r_a and (p_a + r_a) > 0) else np.nan

        store["accuracy"].append(acc)
        store["precision_desert"].append(p_d)
        store["recall_desert"].append(r_d)
        store["f1_desert"].append(f1_d)
        store["precision_access"].append(p_a)
        store["recall_access"].append(r_a)
        store["f1_access"].append(f1_a)
        store["auc"].append(roc_auc_score(yt, ypr))

    def _ci(vals):
        arr = np.array([v for v in vals if not np.isnan(v)])
        if len(arr) == 0:
            return np.nan, np.nan, np.nan
        return arr.mean(), np.percentile(arr, 2.5), np.percentile(arr, 97.5)

    return {k: _ci(v) for k, v in store.items() if k not in ("tn","fp","fn","tp")}


def tune_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[float, pd.DataFrame]:
    """Return (best_threshold_by_f1, full_table)."""
    rows = []
    best_f1, best_thresh = 0.0, 0.5
    for thresh in np.arange(0.10, 0.71, 0.05):
        preds = (y_proba >= thresh).astype(int)
        tp = ((preds == 1) & (y_true == 1)).sum()
        fp = ((preds == 1) & (y_true == 0)).sum()
        fn = ((preds == 0) & (y_true == 1)).sum()
        prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1     = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0
        rows.append({"threshold": round(thresh, 2), "precision": round(prec, 3),
                     "recall": round(recall, 3), "f1": round(f1, 3),
                     "flagged": int(preds.sum())})
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh
    return best_thresh, pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1 — DESERT CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────
section("MODEL 1: Desert Classifier")

pre_drop = df[["zip"] + DESERT_FEATURES + ["is_food_desert"]]
print(f"Positives before dropna: {pre_drop['is_food_desert'].sum()}")
print(f"Positives after dropna:  {pre_drop.dropna()['is_food_desert'].sum()}")
missing_by_feature = pre_drop[pre_drop['is_food_desert']==1][DESERT_FEATURES].isna().sum()
print(f"Missingness among desert-positive zips:\n{missing_by_feature}")


# ── Data prep ─────────────────────────────────────────────────────────────
model1_data = (
    df[["zip", "county_fips"] + DESERT_FEATURES + ["is_food_desert"]]
    .dropna()
    .reset_index(drop=True)
)

print(f"Complete rows: {len(model1_data)}")

n_pos = model1_data["is_food_desert"].sum()
n_tot = len(model1_data)

print(f"Total desert positives: {n_pos} / {n_tot} ({n_pos/n_tot*100:.1f}%)")

X      = model1_data[DESERT_FEATURES]
y_a    = model1_data["is_food_desert"]  # Variant A: 5-mile buffer (no poverty gate)

X_tr_a, X_te_a, y_tr_a, y_te_a = train_test_split(
    X, y_a, test_size=0.20, random_state=RANDOM_SEED, stratify=y_a)

# ── Baseline drift check (needs df, not y_tr_a — safe to run anytime) ─────
n_pos = df["is_food_desert"].sum()
n_tot = len(df)
#baseline_n, baseline_pct = 84, 15.1
#drift = abs(n_pos - baseline_n)

print(f"\nVariant A — 5mi desert rate: {y_tr_a.mean():.3f}  "
      f"(pos_weight {(1-y_tr_a.mean())/y_tr_a.mean():.1f}x)")
#if drift > 5:
    #print(f"  ⚠ WARNING: {int(n_pos)} positives ({n_pos/n_tot*100:.1f}%) — "
          #f"differs from symposium baseline (~{baseline_n}, {baseline_pct}%) "
          #f"by {drift}. Investigate upstream before trusting results.")
#else:
    #print(f"  {int(n_pos)} positives ({n_pos/n_tot*100:.1f}%) — matches symposium baseline")


# ── Model zoo ─────────────────────────────────────────────────────────────
desert_models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced", max_iter=1_000, random_state=RANDOM_SEED)),
    ]),
    "Random Forest": Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            max_features="sqrt", random_state=RANDOM_SEED, n_jobs=-1)),
    ]),
    "Gradient Boosting": Pipeline([
        ("clf", GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05,
            max_depth=4, subsample=0.8, random_state=RANDOM_SEED)),
    ]),
}

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)


def train_variant(name, X_tr, X_te, y_tr, y_te, models):
    print(f"\n── {name}")
    results = {}
    for mname, model in models.items():
        cv_aucs = []
        for tr_idx, val_idx in cv5.split(X_tr, y_tr):
            m = clone(model)
            Xf, yf = X_tr.iloc[tr_idx], y_tr.iloc[tr_idx]
            Xv, yv = X_tr.iloc[val_idx], y_tr.iloc[val_idx]
            if mname == "Gradient Boosting":
                m.fit(Xf, yf, clf__sample_weight=get_sample_weights(yf))
            else:
                m.fit(Xf, yf)
            cv_aucs.append(roc_auc_score(yv, m.predict_proba(Xv)[:, 1]))

        m_final = clone(model)
        if mname == "Gradient Boosting":
            m_final.fit(X_tr, y_tr, clf__sample_weight=get_sample_weights(y_tr))
        else:
            m_final.fit(X_tr, y_tr)

        y_pred  = m_final.predict(X_te)
        y_proba = m_final.predict_proba(X_te)[:, 1]
        auc     = roc_auc_score(y_te, y_proba)

        results[mname] = {
            "model":       m_final,
            "cv_mean":     np.mean(cv_aucs),
            "cv_std":      np.std(cv_aucs),
            "test_auc":    auc,
            "y_pred":      y_pred,
            "y_proba":     y_proba,
        }
        print(f"  {mname:<22} CV={np.mean(cv_aucs):.3f}±{np.std(cv_aucs):.3f}  "
              f"TestAUC={auc:.3f}")
        print(classification_report(y_te, y_pred,
                                    target_names=["Has Access", "Desert"], digits=3))
    return results


results_a = train_variant("Variant A (zip-level binary)", X_tr_a, X_te_a, y_tr_a, y_te_a, desert_models)

# Add these lines:
best_name_b = max(results_a, key=lambda k: results_a[k]["test_auc"])
best_result  = results_a[best_name_b]
best_model   = best_result["model"]
print(f"\nBest model: {best_name_b}  AUC={best_result['test_auc']:.3f}")

# ── Permutation importance ────────────────────────────────────────────────
print("\nPermutation importance...")
perm = permutation_importance(
    best_model, X_te_a, y_te_a,
    n_repeats=20, random_state=RANDOM_SEED, scoring="roc_auc", n_jobs=-1,
)
fi_df = pd.DataFrame({
    "feature":    X_te_a.columns,
    "importance": perm.importances_mean,
    "std":        perm.importances_std,
}).sort_values("importance", ascending=False)

print(fi_df.to_string(index=False))
fi_df.to_csv(OUTPUT_FI, index=False)
print(f"Saved: {OUTPUT_FI.name}")

# ── Bootstrap ─────────────────────────────────────────────────────────────
print(f"\nBootstrap ({N_BOOTSTRAP} resamples)...")
boot = bootstrap_metrics(
    np.array(y_te_a), best_result["y_pred"], best_result["y_proba"]
)

print(f"\n{'Metric':<28} {'Mean':>7}  {'95% CI':>18}")
print("─" * 58)
for label, key in [
    ("AUC",                     "auc"),
    ("Accuracy",                "accuracy"),
    ("Precision  [Desert]",     "precision_desert"),
    ("Recall     [Desert]",     "recall_desert"),
    ("F1         [Desert]",     "f1_desert"),
    ("Precision  [Has Access]", "precision_access"),
    ("Recall     [Has Access]", "recall_access"),
    ("F1         [Has Access]", "f1_access"),
]:
    m, lo, hi = boot[key]
    print(f"  {label:<26} {m:>6.3f}   [{lo:.3f}, {hi:.3f}]")

boot_rows = [{"metric": k, "mean": round(v[0], 4),
              "ci_lo": round(v[1], 4), "ci_hi": round(v[2], 4)}
             for k, v in boot.items()]
pd.DataFrame(boot_rows).to_csv(OUTPUT_BOOT, index=False)
print(f"\nSaved: {OUTPUT_BOOT.name}")

# ── Threshold tuning ─────────────────────────────────────────────────────
print("\nThreshold tuning...")
best_thresh, thresh_df = tune_threshold(
    np.array(y_te_a), best_result["y_proba"]
)
print(thresh_df.to_string(index=False))
print(f"\nBest threshold by F1: {best_thresh:.2f}")
thresh_df.to_csv(OUTPUT_THRESH, index=False)
print(f"Saved: {OUTPUT_THRESH.name}")

# ── Spatial CV (leave-one-county-out) ─────────────────────────────────────
section("Spatial CV — Leave-One-County-Out")

counties    = model1_data["county_fips"].values
X_arr       = model1_data[DESERT_FEATURES].values
y_arr       = model1_data["is_food_desert"].values
all_probs   = np.full(len(y_arr), np.nan)
scv_rows    = []

for county in sorted(np.unique(counties)):
    if county == "Unknown":
        continue
    test_mask  = counties == county
    train_mask = ~test_mask
    n_pos = y_arr[test_mask].sum()
    if test_mask.sum() < 2 or n_pos == 0:
        print(f"  {county:<25} skipped (no desert cases)")
        continue

    rf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced",
        max_features="sqrt", random_state=RANDOM_SEED, n_jobs=-1,
    )
    rf.fit(X_arr[train_mask], y_arr[train_mask])
    probs = rf.predict_proba(X_arr[test_mask])[:, 1]
    all_probs[test_mask] = probs

    auc = roc_auc_score(y_arr[test_mask], probs)
    scv_rows.append({
        "county":  county,
        "n":       int(test_mask.sum()),
        "deserts": int(n_pos),
        "auc":     round(auc, 3),
    })
    print(f"  {county:<25} n={test_mask.sum():>4}  deserts={n_pos:>3}  AUC={auc:.3f}")

valid_mask        = ~np.isnan(all_probs)
spatial_auc       = roc_auc_score(y_arr[valid_mask], all_probs[valid_mask])
random_kfold_auc  = best_result["cv_mean"]

print(f"\nOverall spatial CV AUC : {spatial_auc:.3f}")
print(f"Random k-fold AUC      : {random_kfold_auc:.3f}  "
      f"(Δ = {spatial_auc - random_kfold_auc:+.3f})")

scv_df = pd.DataFrame(scv_rows)
scv_df.to_csv(OUTPUT_SCV, index=False)
print(f"Saved: {OUTPUT_SCV.name}")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2 — TYPOLOGY CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────
section("MODEL 2: Access Typology Classifier")

typo_data = (
    df[["zip"] + TYPOLOGY_FEATURES + ["access_typology"]]
    .dropna()
    .reset_index(drop=True)
)
print(f"Complete rows: {len(typo_data)}")
print("\nClass distribution:")
print(typo_data["access_typology"].value_counts().to_string())

le   = LabelEncoder()
y_t  = le.fit_transform(typo_data["access_typology"])
X_t  = typo_data[TYPOLOGY_FEATURES]

X_tr_t, X_te_t, y_tr_t, y_te_t = train_test_split(
    X_t, y_t, test_size=0.20, random_state=RANDOM_SEED, stratify=y_t,
)

typo_rf = RandomForestClassifier(
    n_estimators=300, class_weight="balanced",
    max_features="sqrt", random_state=RANDOM_SEED, n_jobs=-1,
)
typo_rf.fit(X_tr_t, y_tr_t)
y_pred_t = typo_rf.predict(X_te_t)

print(f"\nTypology classifier:")
print(classification_report(y_te_t, y_pred_t,
                             target_names=le.classes_, digits=3))

# Per-class probabilities for all zips
X_all_t     = df[TYPOLOGY_FEATURES].copy()
rows_valid   = X_all_t.dropna().index
typo_proba   = np.full((len(df), len(le.classes_)), np.nan)
typo_pred    = np.full(len(df), "", dtype=object)

if len(rows_valid) > 0:
    proba_valid = typo_rf.predict_proba(X_all_t.loc[rows_valid])
    pred_valid  = le.inverse_transform(typo_rf.predict(X_all_t.loc[rows_valid]))
    for i, row_idx in enumerate(rows_valid):
        typo_proba[row_idx] = proba_valid[i]
        typo_pred[row_idx]  = pred_valid[i]

df["predicted_typology"] = typo_pred
for cls in le.classes_:
    col_idx = list(le.classes_).index(cls)
    df[f"typo_prob_{cls.lower().replace(' ', '_')}"] = [
        round(typo_proba[i, col_idx], 4) if not np.isnan(typo_proba[i, 0]) else np.nan
        for i in range(len(df))
    ]

print("Typology predictions added to dataframe.")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 3 — HEALTH OUTCOME REGRESSORS
# ─────────────────────────────────────────────────────────────────────────────
section("MODEL 3: Health Outcome Regressors")

health_importance_dfs = {}

for src_col in HEALTH_TARGETS:
    if src_col not in df.columns:
        print(f"  SKIP {src_col} — column not found")
        continue
    out_col = src_col  # same name, used for file output

    h_data = (
        df[["zip"] + HEALTH_FEATURES + [src_col]]
        .dropna()
        .reset_index(drop=True)
    )
    if len(h_data) < 50:
        print(f"  SKIP {out_col} — only {len(h_data)} complete rows")
        continue

    print(f"\n── {out_col}  (n={len(h_data)})")

    X_h = h_data[HEALTH_FEATURES]
    y_h = h_data[src_col].astype(float)

    X_tr_h, X_te_h, y_tr_h, y_te_h = train_test_split(
        X_h, y_h, test_size=0.20, random_state=RANDOM_SEED,
    )

    rf_h = RandomForestRegressor(
        n_estimators=300, max_features="sqrt",
        random_state=RANDOM_SEED, n_jobs=-1,
    )
    rf_h.fit(X_tr_h, y_tr_h)
    y_hat = rf_h.predict(X_te_h)

    r2  = r2_score(y_te_h, y_hat)
    mae = mean_absolute_error(y_te_h, y_hat)
    print(f"  R² = {r2:.3f}   MAE = {mae:.3f}")

    # Permutation importance
    perm_h = permutation_importance(
        rf_h, X_te_h, y_te_h, n_repeats=20,
        random_state=RANDOM_SEED, n_jobs=-1,
    )
    hi_df = pd.DataFrame({
        "feature":    X_te_h.columns,
        "importance": perm_h.importances_mean,
        "std":        perm_h.importances_std,
    }).sort_values("importance", ascending=False)
    print(hi_df.head(5).to_string(index=False))

    safe_name = out_col.replace("/", "-").replace(" ", "_").replace("%", "pct").replace("(", "").replace(")", "")
    out_path = DATA_DIR / f"health_importance_{safe_name}.csv"
    hi_df.to_csv(out_path, index=False)
    health_importance_dfs[out_col] = hi_df
    print(f"  Saved: {out_path.name}")

    # Score all zips
    X_score_h   = df[HEALTH_FEATURES].copy()
    rows_h_valid = X_score_h.dropna().index
    preds_h      = np.full(len(df), np.nan)
    if len(rows_h_valid) > 0:
        preds_h[rows_h_valid] = rf_h.predict(X_score_h.loc[rows_h_valid])
    df[f"predicted_{out_col}"] = preds_h.round(3)


# ─────────────────────────────────────────────────────────────────────────────
# SCORE ALL ZIP CODES (Model 1 — primary deliverable)
# ─────────────────────────────────────────────────────────────────────────────
section("SCORE ALL ZIP CODES")

X_score         = df[DESERT_FEATURES].copy()
rows_valid_m1   = X_score.dropna().index
desert_prob     = np.full(len(df), np.nan)
predicted       = np.full(len(df), np.nan)
predicted_tuned = np.full(len(df), np.nan)

if len(rows_valid_m1) > 0:
    proba_all = best_model.predict_proba(X_score.loc[rows_valid_m1])[:, 1]
    desert_prob[rows_valid_m1]     = proba_all
    predicted[rows_valid_m1]       = (proba_all >= 0.50).astype(int)
    predicted_tuned[rows_valid_m1] = (proba_all >= best_thresh).astype(int)

df["desert_probability"]    = desert_prob.round(4)
df["predicted_desert"]      = predicted
df["predicted_desert_tuned"]= predicted_tuned

print(f"Predicted deserts (0.50 threshold)   : "
      f"{int(np.nansum(predicted))} ({np.nanmean(predicted)*100:.1f}%)")
print(f"Predicted deserts ({best_thresh:.2f} threshold) : "
      f"{int(np.nansum(predicted_tuned))} ({np.nanmean(predicted_tuned)*100:.1f}%)")

print(df[['is_food_desert', 'pct_elderly']].dropna().groupby('is_food_desert').size())
print(df[TYPOLOGY_FEATURES].isna().sum().sort_values(ascending=False))
# ─────────────────────────────────────────────────────────────────────────────
# SAVE nj_zip_scores.csv
# ─────────────────────────────────────────────────────────────────────────────
section("SAVE OUTPUTS")

# Columns to include in the scored output
SCORE_OUTPUT_COLS = (
    ["zip"]
    # Core identifiers
    + ["county_fips"]
    # Demographics
    + [c for c in DESERT_FEATURES if c in df.columns]
    # Ethnicity (EDA only — not model inputs)
    + [c for c in ["pct_acs_nh_white", "pct_acs_nh_black",
                   "pct_acs_nh_asian", "pct_acs_hispanic"] if c in df.columns]
    # Food environment counts
    + [c for c in ["supermarket", "fast_food", "convenience",
                   "restaurant", "dollar_store", "total_unhealthy",
                   "wic_stores", "snap_stores", "snap_supermarkets"] if c in df.columns]
    # Food environment indexes
    + [c for c in ["rfei", "rfei_full", "mrfei", "mrfei_wic", "mrfei_gap",
                   "nj_swamp_score", "swamp_ratio", "dollar_store_ratio",
                   "snap_quality_ratio", "snap_stores_per_10k",
                   "snap_supermarkets_per_10k", "wic_stores_per_10k"] if c in df.columns]
    # Proximity
    + [c for c in ["nearest_supermarket_miles", "supermarkets_within_5mi"] if c in df.columns]
    # Desert flags
    + [c for c in ["is_food_desert", "is_desert_5mi",
                   "usda_desert_flag"] if c in df.columns]
    # Swamp flags
    + [c for c in ["is_swamp_rfei", "is_swamp_mrfei", "is_swamp_mrfei_wic",
                   "is_swamp_nj", "is_swamp_consensus",
                   "swamp_method_count"] if c in df.columns]
    # Mirage / typology
    + [c for c in ["is_food_mirage_v2", "mirage_score", "mirage_barrier_count",
                   "food_mirage_score", "is_food_mirage", "is_rural_mirage",
                   "food_access_type", "access_typology",
                   "dollar_store_desert", "dollar_store_dominance"] if c in df.columns]
    # Vulnerability scores
    + [c for c in ["elderly_vuln_score", "high_elderly_vuln",
                   "novehicle_vuln_score", "high_novehicle_vuln",
                   "composite_vuln_index", "vuln_tier",
                   "health_burden_index", "housing_stress_index",
                   "tax_burden_index"] if c in df.columns]
    # Economic stress
    + [c for c in ["economic_stress_score", "need_burden",
                   "income_stress", "mobility_barrier"] if c in df.columns]
    # Model outputs
    + ["desert_probability", "predicted_desert", "predicted_desert_tuned"]
    + ["predicted_typology"]
    + [c for c in df.columns if c.startswith("typo_prob_")]
    + [f"predicted_{k}" for k in HEALTH_TARGETS if f"predicted_{k}" in df.columns]
)

# Deduplicate while preserving order
seen = set()
SCORE_OUTPUT_COLS = [c for c in SCORE_OUTPUT_COLS
                     if c not in seen and not seen.add(c)]

scored = df[SCORE_OUTPUT_COLS].copy()
scored["zip"] = normalize_zip(scored["zip"])
scored = scored.sort_values("desert_probability", ascending=False, na_position="last")
scored.to_csv(OUTPUT_SCORES, index=False)
print(f"Saved: {OUTPUT_SCORES.name}  ({len(scored)} zips, {len(scored.columns)} columns)")

print("\nTop 15 highest desert probability:")
print(scored.head(15)[["zip", "desert_probability", "predicted_desert_tuned",
                         "is_food_desert", "pct_poverty", "median_income"]].to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# USDA LILA EXTERNAL VALIDATION
# Compares Model 1's predictions against the independently-sourced USDA
# Food Access Research Atlas (LILA tracts aggregated to ZIP in 03_features.py
# as `is_desert_fara`). This is the "does our model agree with an external
# ground truth" check — separate from internal CV/AUC, which only tells you
# the model is consistent with its own training target.
# ─────────────────────────────────────────────────────────────────────────────
section("USDA LILA External Validation")

USDA_TARGET_COL = "is_desert_fara"   # ZIP has ≥1 LILA tract (FARA tract→ZIP join)

validation_cols = ["zip", "predicted_desert", "predicted_desert_tuned",
                    USDA_TARGET_COL, "is_food_desert",
                    "pct_poverty", "median_income", "desert_probability"]
validation_cols = [c for c in validation_cols if c in df.columns]

usda_compare = df[validation_cols].dropna(
    subset=["predicted_desert_tuned", USDA_TARGET_COL]
).copy()

print(f"Comparable zip codes: {len(usda_compare)} / {len(df)}")

if len(usda_compare) > 0:
    y_usda  = usda_compare[USDA_TARGET_COL].astype(int)
    y_model = usda_compare["predicted_desert_tuned"].astype(int)

    usda_agreement = (y_model == y_usda).mean()
    usda_kappa     = cohen_kappa_score(y_usda, y_model)

    print(f"\nModel predicts desert : {y_model.sum()}  ({y_model.mean()*100:.1f}%)")
    print(f"USDA flags as desert   : {y_usda.sum()}  ({y_usda.mean()*100:.1f}%)")
    print(f"\nAgreement rate : {usda_agreement*100:.1f}%")
    print(f"Cohen's Kappa  : {usda_kappa:.3f}")
    print("(0=chance, 0.2=fair, 0.4=moderate, 0.6=substantial, 0.8=almost perfect)")

    print("\nClassification report (USDA as ground truth):")
    print(classification_report(y_usda, y_model,
                                 target_names=["Non-Desert", "Desert"], digits=3))

    cm_usda = confusion_matrix(y_usda, y_model, labels=[0, 1])
    tn, fp, fn, tp = cm_usda.ravel()
    print("Confusion matrix (rows=USDA truth, cols=Model prediction):")
    print(f"                  Pred: Non-Desert   Pred: Desert")
    print(f"  True: Non-Desert    {tn:>10}       {fp:>10}")
    print(f"  True: Desert        {fn:>10}       {tp:>10}")

    fp_zips = usda_compare[(y_model == 1) & (y_usda == 0)]
    fn_zips = usda_compare[(y_model == 0) & (y_usda == 1)]

    print(f"\nFalse positives (model flags, USDA doesn't): {len(fp_zips)}")
    if len(fp_zips):
        print(fp_zips[["zip", "desert_probability", "pct_poverty",
                        "median_income"]].head(10).to_string(index=False))

    print(f"\nFalse negatives (USDA flags, model doesn't): {len(fn_zips)}")
    if len(fn_zips):
        print(fn_zips[["zip", "desert_probability", "pct_poverty",
                        "median_income"]].head(10).to_string(index=False))

    usda_compare["zip"] = normalize_zip(usda_compare["zip"])
    usda_compare.to_csv(DATA_DIR / "usda_model_comparison.csv", index=False)
    print(f"\nSaved: usda_model_comparison.csv ({len(usda_compare)} zips)")
else:
    usda_agreement, usda_kappa = np.nan, np.nan
    print("[WARNING] No overlapping zips between model predictions and "
          "USDA LILA data — skipping validation.")
# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE METADATA
# ─────────────────────────────────────────────────────────────────────────────
section("METADATA")

skipped_counties = [
    c for c in sorted(np.unique(counties))
    if c != "Unknown" and y_arr[counties == c].sum() == 0
]

metadata = {
    "run_date":                 str(date.today()),
    "n_zips_scored":            int(len(scored)),
    "n_model_features":         len(DESERT_FEATURES),
    "desert_features":          DESERT_FEATURES,
    # Desert model
    "best_desert_model":        best_name_b,
    "variant_b_test_auc":       round(best_result["test_auc"], 3),
    "variant_b_cv_auc_mean":    round(best_result["cv_mean"], 3),
    "variant_b_cv_auc_std":     round(best_result["cv_std"], 3),
    "spatial_cv_auc":           round(spatial_auc, 3),
    "random_kfold_auc":         round(random_kfold_auc, 3),
    "best_threshold":           round(best_thresh, 2),
    # Bootstrap
    "boot_auc_mean":            round(boot["auc"][0], 3),
    "boot_auc_lo":              round(boot["auc"][1], 3),
    "boot_auc_hi":              round(boot["auc"][2], 3),
    "boot_recall_desert_mean":  round(boot["recall_desert"][0], 3),
    # Top predictors
    "top_predictors":           fi_df.head(3)[["feature", "importance"]].assign(
                                    importance=lambda d: d["importance"].round(4)
                                ).to_dict("records"),
    # Variant A comparison
    "variant_a_best_model":     max(results_a, key=lambda k: results_a[k]["test_auc"]),
    "variant_a_test_auc":       round(
                                    results_a[max(results_a,
                                    key=lambda k: results_a[k]["test_auc"])]["test_auc"], 3),
    # Health models
    "health_targets": HEALTH_TARGETS,
    # Spatial CV
    "spatial_cv_skipped_counties": skipped_counties,
    # Counts
    "predicted_deserts_050":    int(np.nansum(predicted)),
    "predicted_deserts_tuned":  int(np.nansum(predicted_tuned)),

    # USDA LILA external validation
    "usda_validation_n":        int(len(usda_compare)),
    "usda_agreement_rate":      (round(float(usda_agreement), 3)
                                  if not np.isnan(usda_agreement) else None),
    "usda_cohen_kappa":         (round(float(usda_kappa), 3)
                                  if not np.isnan(usda_kappa) else None),
}

OUTPUT_META.write_text(json.dumps(metadata, indent=2))
print(f"Saved: {OUTPUT_META.name}")

# ─────────────────────────────────────────────────────────────────────────────
# DESERT vs SWAMP — HEALTH OUTCOME ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
section("DESERT vs SWAMP: Health Outcome Analysis")

try:
    import statsmodels.api as sm
    from sklearn.decomposition import PCA

    available_health = [c for c in HEALTH_TARGETS if c in df.columns]
    missing_health = [c for c in HEALTH_TARGETS if c not in df.columns]
    if missing_health:
        print(f"  ⚠ Dropping {len(missing_health)} missing health targets: {missing_health}")

    mirage_candidates = [c for c in ["is_food_mirage_v2", "is_food_mirage",
                                     "mirage_score", "mirage_composite"]
                         if c in df.columns]

    snap_extra = [c for c in ["SNAP/Food Stamp Use % (Adults)", "pct_snap"]
                  if c in df.columns]

    analysis_df = df[["zip", "desert_probability", "is_food_desert", "rfei", "swamp_method_count",
                      "pct_poverty", "median_income", "pct_snap"] + mirage_candidates
                     + available_health].copy()

    print(f"Analysis dataset: {len(analysis_df)} zips")


    # ── Deprivation index (PCA on poverty + income) ───────────────────
    # NOTE: This is a simple 2-variable economic confound control, NOT a
    # reproduction of NJDCA's official Composite Food Desert Factor Score.
    # NJDCA's score uses factor analysis across 40+ variables in 5 domains
    # (retail, demographics, economic, health, community) at the sub-
    # municipal Food Desert Community (FDC) level — not zip-code level.
    # This index exists only to absorb poverty/income multicollinearity
    # in the regressions below.


    econ = analysis_df[["pct_poverty", "median_income"]].dropna()
    pca  = PCA(n_components=1)
    pca.fit(
        (econ - econ.mean()) / econ.std()
    )
    analysis_df["deprivation_index"] = np.nan
    valid_idx = analysis_df[["pct_poverty", "median_income"]].dropna().index
    analysis_df.loc[valid_idx, "deprivation_index"] = pca.transform(
        (analysis_df.loc[valid_idx, ["pct_poverty", "median_income"]] -
         econ.mean()) / econ.std()
    )
    print(f"Deprivation index: PC1 explains "
          f"{pca.explained_variance_ratio_[0]*100:.1f}% of variance")

    # ── Outcomes to test ──────────────────────────────────────────────────────────
    OUTCOMES = {col: col for col in HEALTH_TARGETS if col in analysis_df.columns}

    results = []

    print(f"\n{'Outcome':<30} {'Desert coef':>12} {'Desert p':>10} "
          f"{'RFEI coef':>12} {'RFEI p':>10} {'Dominant':>10}")
    print("─" * 90)

    for col, label in OUTCOMES.items():
        if col not in analysis_df.columns:
            continue

        subset = analysis_df[["is_food_desert", "rfei", "swamp_method_count",
                              "deprivation_index", col]].dropna()
        if len(subset) < 50:
            continue

        try:
            subset = subset.copy()
            subset[col] = pd.to_numeric(subset[col], errors="coerce")
            subset = subset.dropna()

            X = sm.add_constant(subset[["is_food_desert", "rfei",
                                        "swamp_method_count", "deprivation_index"]])
            model_ols = sm.OLS(subset[col], X).fit()

            d_coef = model_ols.params.get("is_food_desert", np.nan)
            d_p = model_ols.pvalues.get("is_food_desert", np.nan)
            s_coef = model_ols.params.get("rfei", np.nan)
            s_p = model_ols.pvalues.get("rfei", np.nan)
            r2 = model_ols.rsquared

            d_sig = d_p < 0.05 if not np.isnan(d_p) else False
            s_sig = s_p < 0.05 if not np.isnan(s_p) else False

            if d_sig and not s_sig:
                dominant = "Desert" if d_coef > 0 else "Desert (protective)"
            elif s_sig and not d_sig:
                dominant = "Swamp" if s_coef > 0 else "Swamp (protective)"
            elif d_sig and s_sig:
                dominant = "Both"
            else:
                dominant = "Neither"

            results.append({
                "outcome": label,
                "col": col,
                "desert_coef": round(d_coef, 4),
                "desert_p": round(d_p, 4),
                "rfei_coef": round(s_coef, 4),
                "rfei_p": round(s_p, 4),
                "r2": round(r2, 3),
                "n": len(subset),
                "dominant": dominant,
            })

            d_str = f"{d_coef:+.3f} ({'✓' if d_sig else '✗'} p={d_p:.3f})"
            s_str = f"{s_coef:+.3f} ({'✓' if s_sig else '✗'} p={s_p:.3f})"
            print(f"  {label:<28} {d_str:>20} {s_str:>20} {dominant:>10}")

        except Exception as e:
            print(f"  [SKIP] {label}: {e}")

    results_df = pd.DataFrame(results)

    # ── Summary by dominant predictor ─────────────────────────────────────
    print(f"\nSummary:")
    for dom in ["Desert", "Desert (protective)", "Swamp", "Swamp (protective)", "Both", "Neither"]:
        subset = results_df[results_df["dominant"] == dom]["outcome"].tolist()
        if subset:
            print(f"  {dom:<25}: {', '.join(subset)}")

    OUTPUT_SWAMP = DATA_DIR / "desert_swamp_health.csv"
    results_df.to_csv(OUTPUT_SWAMP, index=False)
    print(f"\nSaved: {OUTPUT_SWAMP.name}")

    # ── Mirage health outcomes ─────────────────────────────────────────────
    print(f"\n── Mirage vs Non-Mirage: Health Outcome Comparison ─────────────")

    mirage_col = next((c for c in ["is_food_mirage_v2", "is_food_mirage"]
                       if c in analysis_df.columns), None)

    if mirage_col:
        mirage_outcomes = {}

        print(f"\n  Using mirage flag: {mirage_col}")
        print(f"  Mirages: {analysis_df[mirage_col].sum():.0f} | "
              f"Non-mirages: {(~analysis_df[mirage_col].astype(bool)).sum():.0f}")

        print(f"\n  {'Outcome':<30} {'Mirage mean':>12} {'Non-mirage mean':>16} "
              f"{'Δ':>8} {'p':>8} {'sig':>6}")
        print("  " + "─" * 84)

        for col, label in OUTCOMES.items():
            if col not in analysis_df.columns:
                continue

            sub = analysis_df[[mirage_col, col, "deprivation_index"]].copy()
            sub[col] = pd.to_numeric(sub[col], errors="coerce")
            sub[mirage_col] = sub[mirage_col].astype(float)
            sub = sub.dropna()
            if len(sub) < 50:
                continue

            mirage_vals = sub[sub[mirage_col] == 1][col]
            non_mirage_vals = sub[sub[mirage_col] == 0][col]

            if len(mirage_vals) < 5 or len(non_mirage_vals) < 5:
                continue

            from scipy import stats

            t_stat, p_val = stats.ttest_ind(mirage_vals, non_mirage_vals,
                                            equal_var=False)
            delta = mirage_vals.mean() - non_mirage_vals.mean()
            sig = "✓" if p_val < 0.05 else "✗"

            mirage_outcomes[col] = {
                "label": label,
                "mirage_mean": round(mirage_vals.mean(), 2),
                "non_mirage_mean": round(non_mirage_vals.mean(), 2),
                "delta": round(delta, 2),
                "p": round(p_val, 4),
                "significant": p_val < 0.05,
            }

            print(f"  {label:<30} {mirage_vals.mean():>12.2f} "
                  f"{non_mirage_vals.mean():>16.2f} "
                  f"{delta:>+8.2f} {p_val:>8.3f} {sig:>6}")

        # OLS: mirage → each outcome controlling for deprivation
        print(f"\n  OLS (controlling for deprivation index):")
        print(f"  {'Outcome':<30} {'Mirage coef':>12} {'p':>8} {'R²':>6}")
        print("  " + "─" * 60)

        mirage_ols_results = []
        for col, label in OUTCOMES.items():
            if col not in analysis_df.columns:
                continue

            sub = analysis_df[[mirage_col, col,
                               "deprivation_index"]].copy()
            sub[col] = pd.to_numeric(sub[col], errors="coerce")
            sub[mirage_col] = sub[mirage_col].astype(float)
            sub = sub.dropna()
            if len(sub) < 50:
                continue

            try:
                X = sm.add_constant(sub[[mirage_col, "deprivation_index"]])
                ols = sm.OLS(sub[col], X).fit()
                coef = ols.params.get(mirage_col, np.nan)
                p = ols.pvalues.get(mirage_col, np.nan)
                r2 = ols.rsquared
                sig = "✓" if p < 0.05 else "✗"
                print(f"  {label:<30} {coef:>+12.3f} {p:>8.3f} {r2:>6.3f}  {sig}")
                mirage_ols_results.append({
                    "outcome": label,
                    "col": col,
                    "mirage_coef": round(coef, 4),
                    "mirage_p": round(p, 4),
                    "r2": round(r2, 3),
                    "significant": p < 0.05,
                })
            except Exception:
                continue

        # Narrative interpretation
        sig_outcomes = [r["outcome"] for r in mirage_ols_results if r["significant"]]
        elevated = [r for r in mirage_ols_results
                    if r["significant"] and r["mirage_coef"] > 0]
        suppressed = [r for r in mirage_ols_results
                      if r["significant"] and r["mirage_coef"] < 0]

        print(f"\n  Interpretation:")
        if elevated:
            print(f"  ▲ Elevated in mirages (worse outcomes): "
                  f"{', '.join(r['outcome'] for r in elevated)}")
        if suppressed:
            print(f"  ▼ Suppressed in mirages (better outcomes): "
                  f"{', '.join(r['outcome'] for r in suppressed)}")
        if not sig_outcomes:
            print(f"  → No significant mirage health signal after controlling "
                  f"for deprivation.")
            print(f"    Mirage status may proxy for economic stress rather than "
                  f"independently worsening outcomes.")

        # Save mirage results
        mirage_df = pd.DataFrame(mirage_ols_results)
        OUTPUT_MIRAGE = DATA_DIR / "mirage_health.csv"
        mirage_df.to_csv(OUTPUT_MIRAGE, index=False)
        print(f"\n  Saved: {OUTPUT_MIRAGE.name}")

        # Append to metadata
        if OUTPUT_META.exists():
            meta = json.loads(OUTPUT_META.read_text())
            meta["mirage_health"] = {
                "mirage_flag_used": mirage_col,
                "n_mirages": int(analysis_df[mirage_col].sum()),
                "significant_outcomes": sig_outcomes,
                "elevated_outcomes": [r["outcome"] for r in elevated],
                "suppressed_outcomes": [r["outcome"] for r in suppressed],
            }
            OUTPUT_META.write_text(json.dumps(meta, indent=2))

    else:
        print("  [SKIP] No mirage flag found in dataset — "
              "ensure is_food_mirage_v2 or is_food_mirage is present")

    # ── SNAP Confounding Test, split by desert subtype ─────────────────
    print(f"\n── SNAP Confounding Test, by Desert Subtype ──────────────────")

    if "desert_subtype" in df.columns:
        # deprivation_index lives on analysis_df (built earlier), not df —
        # merge it back on zip before splitting by subtype.
        dep_lookup = analysis_df[["zip", "deprivation_index"]].dropna()
        df_with_dep = df.merge(dep_lookup, on="zip", how="left")

        for subtype in ["Structural Desert (isolated, not poor — e.g. elderly/retirement)",
                        "Socioeconomic Desert (isolated AND poor)"]:
            sub = df_with_dep[df_with_dep["desert_subtype"] == subtype]
            sub_data = sub[["desert_probability", "deprivation_index", "pct_snap"]].copy()
            sub_data["pct_snap"] = pd.to_numeric(sub_data["pct_snap"], errors="coerce")
            sub_data = sub_data.dropna()

            if len(sub_data) < 10:
                print(f"\n  {subtype}: n={len(sub_data)} — too few zips to test")
                continue

            X_sub = sm.add_constant(sub_data[["desert_probability", "deprivation_index"]])
            model_sub = sm.OLS(sub_data["pct_snap"], X_sub).fit()

            print(f"\n  {subtype}  (n={len(sub_data)})")
            print(f"    desert_probability: coef={model_sub.params['desert_probability']:+.3f}  "
                  f"p={model_sub.pvalues['desert_probability']:.4f}  "
                  f"R²={model_sub.rsquared:.3f}")
    else:
        print("  [SKIP] desert_subtype column not found — rerun 03_features.py first")

    # ── Elderly % by desert subtype ─────────────────────────────────────
    print(f"\n── Elderly % by Desert Subtype ─────────────────────────────────")
    if "desert_subtype" in df.columns:
        print(df.groupby("desert_subtype")["pct_elderly"].agg(["mean", "count"]).round(2))

    snap_col = next((c for c in ["SNAP/Food Stamp Use % (Adults)", "pct_snap"]
                     if c in analysis_df.columns), None)
    if snap_col is None:
        print("  [SKIP] No SNAP column found in analysis_df")


    if snap_col:
        snap_sub = analysis_df[["desert_probability", "deprivation_index",
                                 snap_col]].dropna()
        snap_sub[snap_col] = pd.to_numeric(snap_sub[snap_col], errors="coerce")
        snap_sub = snap_sub.dropna()

        X_snap = sm.add_constant(snap_sub[["desert_probability",
                                            "deprivation_index"]])
        snap_ols = sm.OLS(snap_sub[snap_col], X_snap).fit()

        d_coef = snap_ols.params.get("desert_probability", np.nan)
        d_p    = snap_ols.pvalues.get("desert_probability", np.nan)
        dep_coef = snap_ols.params.get("deprivation_index", np.nan)
        r2     = snap_ols.rsquared

        print(f"  Outcome: {snap_col}  (n={len(snap_sub)}, R²={r2:.3f})")
        print(f"  deprivation_index : coef={dep_coef:+.3f}  "
              f"(strongly predicts SNAP as expected)")
        print(f"  desert_probability: coef={d_coef:+.3f}  p={d_p:.3f}  "
              f"({'NOT ' if d_p >= 0.05 else ''}significant)")
        if d_p >= 0.05:
            print(f"  → Desert status does not independently predict SNAP use")
            print(f"    once economic deprivation is controlled.")
            print(f"    Both are downstream of poverty — co-occurrence, not causation.")

        # Save to metadata
        if OUTPUT_META.exists():
            meta = json.loads(OUTPUT_META.read_text())
            meta["snap_confounding"] = {
                "desert_coef":  round(float(d_coef), 4),
                "desert_p":     round(float(d_p), 4),
                "deprivation_coef": round(float(dep_coef), 4),
                "r2":           round(float(r2), 3),
                "interpretation": "significant" if d_p < 0.05 else "not significant",
            }
            if "desert_swamp_health" in dir():
                pass
            # Add dominant outcomes summary
            meta["desert_dominant_outcomes"] = results_df[
                results_df["dominant"] == "Desert"]["col"].tolist()
            meta["swamp_dominant_outcomes"] = results_df[
                results_df["dominant"] == "Swamp"]["col"].tolist()
            OUTPUT_META.write_text(json.dumps(meta, indent=2))
            print(f"\nUpdated: {OUTPUT_META.name}")

except FileNotFoundError:
    pass
except ImportError:
    print("[SKIP] statsmodels not installed — run: pip install statsmodels")
except Exception as e:
    print(f"[ERROR] Desert/swamp health analysis failed: {e}")
    import traceback
    traceback.print_exc()

# ── Elderly % by desert subtype ─────────────────────────────────────
print(f"\n── Elderly % by Desert Subtype ─────────────────────────────────")
if "desert_subtype" in df.columns:
    print(df.groupby("desert_subtype")["pct_elderly"].agg(["mean", "count"]).round(2))

# ── Compare Structural vs Socioeconomic Desert, controlling for income ─────
print(f"\n── Structural vs Socioeconomic Desert, controlling for income ────")

desert_only = df[df["desert_subtype"].isin([
    "Structural Desert (isolated, not poor — e.g. elderly/retirement)",
    "Socioeconomic Desert (isolated AND poor)"
])].copy()

# 1 = Socioeconomic, 0 = Structural (reference group)
desert_only["is_socioeconomic"] = (
    desert_only["desert_subtype"] == "Socioeconomic Desert (isolated AND poor)"
).astype(int)

print(f"\nGroup sizes: Structural={len(desert_only[desert_only['is_socioeconomic']==0])}, "
      f"Socioeconomic={len(desert_only[desert_only['is_socioeconomic']==1])}")

compare_outcomes = [
    "pct_elderly",
    "nearest_supermarket_miles",
    "pct_transit",
    "pct_no_vehicle",
    "snap_stores_per_10k",
    "wic_stores_per_10k",
]

for outcome in compare_outcomes:
    if outcome not in desert_only.columns:
        print(f"  [SKIP] {outcome} — column not found")
        continue

    sub = desert_only[["is_socioeconomic", "median_income", outcome]].copy()
    sub[outcome] = pd.to_numeric(sub[outcome], errors="coerce")
    sub = sub.dropna()

    if len(sub) < 15:
        print(f"  [SKIP] {outcome} — too few complete cases (n={len(sub)})")
        continue

    X_cmp = sm.add_constant(sub[["is_socioeconomic", "median_income"]])
    model_cmp = sm.OLS(sub[outcome], X_cmp).fit()

    coef = model_cmp.params["is_socioeconomic"]
    pval = model_cmp.pvalues["is_socioeconomic"]
    sig = "✓" if pval < 0.05 else "✗"

    print(f"\n  {outcome}  (n={len(sub)})")
    print(f"    Socioeconomic vs Structural: {coef:+.3f}  p={pval:.4f}  {sig}"
          f"  (positive = higher in Socioeconomic group)")

# ── Compare health outcomes: Structural vs Socioeconomic Desert ───────
print(f"\n── Health Outcomes: Structural vs Socioeconomic Desert ──────────")
print(f"(controlling for income — connects the distance/elderly finding to health cost)")

health_compare_outcomes = [
    "Obesity % (Adults)",
    "Diabetes % (Adults)",
    "Food Insecurity % (Adults)",
    "Poor Physical Health % (Adults)",
    "Poor Mental Health % (Adults)",
    "Coronary Heart Disease % (Adults)",
    "High Blood Pressure % (Adults)",
    "Social Isolation/Loneliness % (Adults)",
    "Lack of Transportation % (Adults)",
]

health_results = []

for outcome in health_compare_outcomes:
    if outcome not in desert_only.columns:
        print(f"  [SKIP] {outcome} — column not found")
        continue

    sub = desert_only[["is_socioeconomic", "median_income", outcome]].copy()
    sub[outcome] = pd.to_numeric(sub[outcome], errors="coerce")
    sub = sub.dropna()

    if len(sub) < 15:
        print(f"  [SKIP] {outcome} — too few complete cases (n={len(sub)})")
        continue

    X_h = sm.add_constant(sub[["is_socioeconomic", "median_income"]])
    model_h = sm.OLS(sub[outcome], X_h).fit()

    coef = model_h.params["is_socioeconomic"]
    pval = model_h.pvalues["is_socioeconomic"]
    sig = "✓" if pval < 0.05 else "✗"

    # Group means for context (easier to interpret than coef alone)
    mean_struct = sub[sub["is_socioeconomic"] == 0][outcome].mean()
    mean_socio  = sub[sub["is_socioeconomic"] == 1][outcome].mean()

    health_results.append({
        "outcome": outcome,
        "structural_mean": round(mean_struct, 2),
        "socioeconomic_mean": round(mean_socio, 2),
        "coef": round(coef, 3),
        "pval": round(pval, 4),
        "significant": pval < 0.05,
        "n": len(sub),
    })

    print(f"\n  {outcome}  (n={len(sub)})")
    print(f"    Structural mean:     {mean_struct:.2f}")
    print(f"    Socioeconomic mean:  {mean_socio:.2f}")
    print(f"    Coef (Socio-Struct, income-controlled): {coef:+.3f}  p={pval:.4f}  {sig}")

health_compare_df = pd.DataFrame(health_results)
if len(health_compare_df) > 0:
    OUTPUT_SUBTYPE_HEALTH = DATA_DIR / "desert_subtype_health_comparison.csv"
    health_compare_df.to_csv(OUTPUT_SUBTYPE_HEALTH, index=False)
    print(f"\nSaved: {OUTPUT_SUBTYPE_HEALTH.name}")

    sig_outcomes = health_compare_df[health_compare_df["significant"]]["outcome"].tolist()
    print(f"\nSignificant differences (Socioeconomic vs Structural): "
          f"{sig_outcomes if sig_outcomes else 'None'}")


# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
from scipy.stats import ttest_ind


section("FINAL SUMMARY")

print(f"Best desert model  : {best_name_b}")
print(f"Test AUC           : {best_result['test_auc']:.3f}")  # remove "Var B"
print(f"CV AUC             : {best_result['cv_mean']:.3f} ± {best_result['cv_std']:.3f}")
print(f"Spatial CV AUC     : {spatial_auc:.3f}")
print(f"Best threshold     : {best_thresh:.2f}")
print(f"\nTop 3 predictors:")
for _, row in fi_df.head(3).iterrows():
    print(f"  {row['feature']:<25} AUC drop = {row['importance']:.4f}")

print(f"\nOutputs:")
print("\n" + "="*50)
print("ELDERLY ACCESS ANALYSIS")
print("="*50)

print("\nElderly vs No Vehicle")
print(df[['pct_elderly', 'pct_no_vehicle']].corr())

print("\nElderly vs Distance to Supermarket")
print(df[['pct_elderly', 'nearest_supermarket_miles']].corr())

print("\nElderly vs Supermarkets Within 5 Miles")
print(df[['pct_elderly', 'supermarkets_within_5mi']].corr())

print("\nMean Elderly % by Food Desert Status")
print(df.groupby('is_food_desert')['pct_elderly'].mean())

desert    = df[df['is_food_desert'] == 1]['pct_elderly'].dropna()
nondesert = df[df['is_food_desert'] == 0]['pct_elderly'].dropna()

t_stat, p_val = ttest_ind(desert, nondesert)

print("\nFood Desert T-Test")
print(f"t = {t_stat:.3f}")
print(f"p = {p_val:.6f}")

print("\nMean Elderly % by Food Swamp Status")
print(df.groupby('swamp_rfei_flag')['pct_elderly'].mean())

swamp    = df[df['swamp_rfei_flag'] == 1]['pct_elderly'].dropna()
nonswamp = df[df['swamp_rfei_flag'] == 0]['pct_elderly'].dropna()

t_stat, p_val = ttest_ind(swamp, nonswamp)

print("\nFood Swamp T-Test")
print(f"t = {t_stat:.3f}")
print(f"p = {p_val:.6f}")

print("\nMean Elderly % by Food Mirage Status")
print(df.groupby('is_food_mirage')['pct_elderly'].mean())

mirage    = df[df['is_food_mirage'] == 1]['pct_elderly'].dropna()
nonmirage = df[df['is_food_mirage'] == 0]['pct_elderly'].dropna()

t_stat, p_val = ttest_ind(mirage, nonmirage)

print("\nFood Mirage T-Test")
print(f"t = {t_stat:.3f}")
print(f"p = {p_val:.6f}")

demo_vars = [
    'pct_elderly',
    'pct_poverty',
    'median_income',
    'pct_snap',
    'pct_no_vehicle',
    'pct_college',
    'pct_transit'
]

summary = (
    df.groupby('access_typology')[demo_vars]
      .mean()
      .round(2)
)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
print(df.groupby('access_typology')[demo_vars].mean().round(2))
df.groupby('access_typology')[demo_vars].mean().round(2).to_csv(
    "typology_summary.csv"
)