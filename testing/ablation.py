"""
Ablation test for label circularity in the food-desert classifier.

Logic:
  A correlation coefficient between a candidate leak feature (e.g. pop_density)
  and the outcome-adjacent variable (e.g. nearest_supermarket_miles) is weak
  evidence about circularity, because a model — especially a tree-based one —
  can exploit a *threshold* relationship (e.g. pop_density >= 1000) that a
  linear correlation coefficient won't fully capture.

  The stronger test: train the SAME classifier with and without the
  suspect feature(s), using proper cross-validation, and compare predictive
  performance (AUC) and feature importance. If performance barely changes
  when the suspect feature is removed, it wasn't doing much work — weak
  evidence of circularity. If performance drops sharply, or the suspect
  feature dominates importance rankings, that's a real leakage signal.

Usage:
    python ablation_circularity_test.py
Edit the CONFIG block below to match your file path, target column, and
feature lists before running.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

# ═════════════════════════════════════════════════════════════════════════
# CONFIG — edit these to match your setup
# ═════════════════════════════════════════════════════════════════════════
ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "nj_zip_features_v5.csv"
DATA_DIR = ROOT / "data"

#DATA_PATH = "data/nj_zip_features_v5.csv"

# The label being predicted. Use the CLEAN distance-only definition
# (is_desert_5mi), not is_desert_usda, since the latter's threshold logic
# (urban 1mi / rural 10mi via pop_density) bakes pop_density into the
# label itself — that would make any circularity test meaningless.
TARGET_COL = "is_desert_5mi"

# Full candidate feature set: genuinely socioeconomic / demographic only.
# Do NOT include anything derived from retailer counts, distances, or
# retail density — those are the outcome's own ingredients, not predictors
# of it in a non-circular sense.
BASE_FEATURES = [
    "pct_poverty",
    "pct_snap",
    "pct_no_vehicle",
    "pct_elderly",
    "pct_transit",
    "pct_college",
    "median_income",
    "population",
]

# Features under suspicion of leaking the label's own construction logic.
# pop_density: used directly in the urban/rural threshold split for
#   is_desert_usda (not the target here, but density-adjacent features can
#   still act as a proxy for retail infrastructure generally).
# supermarkets_within_5mi: flagged in the pipeline as a same-ZIP count
#   proxy, not a true spatial radius count — effectively re-derived from
#   the same retailer data used to compute nearest_supermarket_miles.
SUSPECT_FEATURES = [
    "pop_density",
    #"supermarkets_within_5mi",
]

N_FOLDS = 5
RANDOM_STATE = 42


# ═════════════════════════════════════════════════════════════════════════
# Load
# ═════════════════════════════════════════════════════════════════════════
df = pd.read_csv(DATA_PATH, dtype={"zip": str})
print(f"Loaded {df.shape[0]} rows x {df.shape[1]} cols")

missing_target = TARGET_COL not in df.columns
if missing_target:
    raise SystemExit(f"Target column '{TARGET_COL}' not found — check DATA_PATH / column name.")

available_base = [c for c in BASE_FEATURES if c in df.columns]
missing_base = [c for c in BASE_FEATURES if c not in df.columns]
available_suspect = [c for c in SUSPECT_FEATURES if c in df.columns]
missing_suspect = [c for c in SUSPECT_FEATURES if c not in df.columns]

if missing_base:
    print(f"[WARN] base features missing, skipped: {missing_base}")
if missing_suspect:
    print(f"[WARN] suspect features missing, skipped: {missing_suspect}")

y = df[TARGET_COL]
print(f"\nTarget prevalence: {y.mean():.3f} ({y.sum()} positive / {len(y)} total)")


corr_check = df[["pop_density", "nearest_supermarket_miles",
                  "supermarkets_within_5mi", "supermarket",
                  "fast_food", "convenience"]].corr()["pop_density"]

def build_pipeline(model):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", model),
    ])


def run_cv(features, label):
    X = df[features]
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    results = {}
    for name, model in [
        ("LogisticRegression", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ("RandomForest", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)),
    ]:
        pipe = build_pipeline(model)
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
        results[name] = (scores.mean(), scores.std())
        print(f"    {name:20} AUC = {scores.mean():.3f} (+/- {scores.std():.3f})")

    return results


print("\n" + "=" * 70)
print(f"  ABLATION TEST — target: {TARGET_COL}")
print("=" * 70)

print(f"\n[A] Base socioeconomic features only ({len(available_base)} features)")
print(f"    {available_base}")
results_base = run_cv(available_base, "base")

if available_suspect:
    full_features = available_base + available_suspect
    print(f"\n[B] Base + suspect features ({len(full_features)} features)")
    print(f"    + suspects: {available_suspect}")
    results_full = run_cv(full_features, "full")

    print("\n" + "-" * 70)
    print("  DELTA (full - base AUC)")
    print("-" * 70)
    for name in results_base:
        delta = results_full[name][0] - results_base[name][0]
        flag = "  <-- meaningful jump, investigate" if abs(delta) > 0.03 else ""
        print(f"    {name:20} delta AUC = {delta:+.3f}{flag}")

    # Feature importance from the RF trained on the full set, refit once
    # on all data (not cross-validated) purely to inspect relative
    # importance rankings, not to report as a generalization metric.
    rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)
    pipe = build_pipeline(rf)
    pipe.fit(df[full_features].fillna(df[full_features].median()), y)
    importances = pd.Series(
        pipe.named_steps["clf"].feature_importances_, index=full_features
    ).sort_values(ascending=False)

    print("\n  Feature importances (full model, RandomForest):")
    for feat, imp in importances.items():
        marker = "  <-- suspect feature" if feat in available_suspect else ""
        print(f"    {feat:30} {imp:.3f}{marker}")

    print("\n" + "=" * 70)
    print("  INTERPRETATION GUIDE")
    print("=" * 70)
    print("""
  - If AUC delta is small (< ~0.03) AND suspect features rank low in
    importance: weak circularity concern, base features carry the signal.
  - If AUC delta is large OR a suspect feature dominates importance:
    the suspect feature is doing meaningful work the model couldn't get
    from socioeconomic data alone — treat as a real leakage risk and
    either drop it or explicitly justify its inclusion in the paper.
  - Report both the correlation check AND this ablation result together;
    a single correlation coefficient is not sufficient evidence on its own.
""")
else:
    print("\nNo suspect features available in this dataset — nothing to compare.")