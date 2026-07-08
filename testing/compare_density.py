"""
compare_density.py
============================
Runs the desert classifier under three DESERT_FEATURES configurations
and prints a side-by-side comparison of AUC, feature importance,
and predicted desert counts.

FIXES vs original:
  1. ZIP-agreement table now uses the SAME held-out test split (X_te/y_te)
     and the SAME threshold (0.50) as the classification_report above it,
     instead of refitting on the full dataset and scoring in-sample zips
     at a different threshold (0.35). The original 100% agreement number
     was comparing mostly-memorized training predictions, not genuine
     out-of-sample agreement — hence the contradiction with the recall
     numbers in the classification report.
  2. Added a direct circularity check: correlation between pop_density
     and nearest_supermarket_miles (and supermarkets_within_5mi), since
     the desert label is built from distance features. A strong
     correlation means pop_density's AUC gain may partly be re-deriving
     the label rather than adding independent signal.

Reads data/nj_zip_features_v5.csv (output of 03_features.py) and writes
data/density_comparison.csv.
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data" / "nj_zip_features_v5.csv"
DATA_DIR = ROOT / "data"
INPUT = DATA_DIR / "nj_zip_features_v5.csv"

RANDOM_SEED = 42
THRESHOLD = 0.50   # single threshold used EVERYWHERE below — report + agreement

df = pd.read_csv(INPUT, dtype={"zip": str})
print(f"Loaded: {df.shape[0]} zips × {df.shape[1]} columns")

# ── Circularity check (NEW) ────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  CIRCULARITY CHECK — does pop_density leak the label definition?")
print("=" * 70)
circ_cols = [c for c in ["pop_density", "population", "nearest_supermarket_miles",
                          "supermarkets_within_5mi"] if c in df.columns]
if len(circ_cols) >= 2:
    corr = df[circ_cols].corr()
    print(corr.round(3).to_string())
    if "pop_density" in corr.columns and "nearest_supermarket_miles" in corr.columns:
        r = corr.loc["pop_density", "nearest_supermarket_miles"]
        print(f"\n  pop_density vs nearest_supermarket_miles: r = {r:.3f}")
        if abs(r) > 0.5:
            print("  ⚠ Strong correlation — pop_density likely proxies the distance")
            print("    feature your desert label is built from. Treat Config C's AUC")
            print("    gain as partially circular, not purely predictive skill.")
        else:
            print("  Correlation is modest — circularity concern is weaker than assumed.")
else:
    print("  [SKIP] Required columns not found for circularity check.")

# ── Three feature sets ────────────────────────────────────────────────────────
BASE_SOCIOECONOMIC = [
    "median_income",
    "pct_poverty",
    "pct_snap",
    "pct_transit",
    "pct_no_vehicle",
    "pct_college",
    "pct_elderly",
    "income_stress",
    "need_burden",
    "economic_stress_score",
]

CONFIGS = {
    "A — Socioeconomic only\n    (no density, no population)": BASE_SOCIOECONOMIC,
    "B — + pop_density only\n    (density without raw population)": BASE_SOCIOECONOMIC + ["pop_density"],
    "C — Full original\n    (density + population — current 04_model.py)": BASE_SOCIOECONOMIC + ["pop_density", "population"],
}

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

results = {}

for config_name, features in CONFIGS.items():
    feats = [f for f in features if f in df.columns]
    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"\n  [WARN] Missing columns skipped: {missing}")

    # Keep the ORIGINAL df index (no reset_index) so test rows can be
    # mapped straight back to df via .loc — this is what broke last time.
    data = df[feats + ["is_food_desert"]].dropna()

    X = data[feats]
    y = data["is_food_desert"]

    # IMPORTANT: same split used for report AND for agreement table below.
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced",
        max_features="sqrt", random_state=RANDOM_SEED, n_jobs=-1,
    )

    cv_aucs = []
    for tr_idx, val_idx in cv5.split(X_tr, y_tr):
        m = clone(rf)
        m.fit(X_tr.iloc[tr_idx], y_tr.iloc[tr_idx])
        cv_aucs.append(roc_auc_score(y_tr.iloc[val_idx],
                                     m.predict_proba(X_tr.iloc[val_idx])[:, 1]))

    rf.fit(X_tr, y_tr)
    y_proba = rf.predict_proba(X_te)[:, 1]
    y_pred = (y_proba >= THRESHOLD).astype(int)
    test_auc = roc_auc_score(y_te, y_proba)

    perm = permutation_importance(
        rf, X_te, y_te,
        n_repeats=20, random_state=RANDOM_SEED, scoring="roc_auc", n_jobs=-1,
    )
    fi = pd.DataFrame({
        "feature":    feats,
        "importance": perm.importances_mean,
    }).sort_values("importance", ascending=False)

    results[config_name] = {
        "features":   feats,
        "n_features": len(feats),
        "cv_auc":     np.mean(cv_aucs),
        "cv_std":     np.std(cv_aucs),
        "test_auc":   test_auc,
        "fi":         fi,
        "zip_te":     df.loc[X_te.index, "zip"].tolist(),
        "y_te":       y_te,
        "y_pred":     y_pred,
        "y_proba":    y_proba,
        "n_desert_pred": int(y_pred.sum()),
    }

# ── Print comparison ──────────────────────────────────────────────────────────
print("\n\n" + "=" * 70)
print("  FEATURE SET COMPARISON — Desert Classifier (Random Forest)")
print("=" * 70)

for config_name, res in results.items():
    print(f"\n{'─' * 70}")
    print(f"  Config {config_name}")
    print(f"{'─' * 70}")
    print(f"  Features ({res['n_features']}): {', '.join(res['features'])}")
    print(f"\n  CV AUC  : {res['cv_auc']:.3f} ± {res['cv_std']:.3f}")
    print(f"  Test AUC: {res['test_auc']:.3f}")
    print(f"\n  Predicted deserts in test set @ {THRESHOLD} threshold : {res['n_desert_pred']} / {len(res['y_te'])}")
    print(f"\n  Classification report (threshold={THRESHOLD}):")
    print(classification_report(
        res["y_te"], res["y_pred"],
        target_names=["Has Access", "Desert"], digits=3
    ))
    print(f"  Top feature importances (permutation AUC drop):")
    for _, row in res["fi"].head(6).iterrows():
        bar = "█" * int(max(0, row["importance"]) * 200)
        print(f"    {row['feature']:<25} {row['importance']:+.4f}  {bar}")

# ── Agreement table — FIXED: same test split, same threshold ─────────────────
print("\n\n" + "=" * 70)
print("  ZIP-LEVEL AGREEMENT — held-out test zips only, same threshold as above")
print("=" * 70)

config_names = list(results.keys())

# All three configs must share the SAME test zips to compare fairly.
# Since each config drops different rows via dropna, intersect the test zips.
te_sets = [set(results[name]["zip_te"]) for name in config_names]
common_zips = set.intersection(*te_sets)
print(f"\n  Held-out zips common to all 3 configs' test splits: {len(common_zips)}")

rows = []
for z in sorted(common_zips):
    row = {"zip": z, "is_desert": df.loc[df["zip"] == z, "is_food_desert"].values[0]}
    for name in config_names:
        res = results[name]
        idx = list(res["zip_te"]).index(z)
        row[f"prob_{name[0]}"] = round(float(res["y_proba"][idx]), 3)
        row[f"flag_{name[0]}"] = int(res["y_pred"][idx])
    rows.append(row)

compare = pd.DataFrame(rows)
flag_cols = [f"flag_{n[0]}" for n in config_names]
compare["all_agree"] = compare[flag_cols].nunique(axis=1) == 1

print(f"\n  Total held-out ZIPs compared : {len(compare)}")
print(f"  All 3 configs agree         : {compare['all_agree'].sum()} "
      f"({compare['all_agree'].mean()*100:.1f}%)")

flipped = compare[~compare["all_agree"]]
if len(flipped) > 0:
    print(f"\n  ZIPs where configs disagree (@ {THRESHOLD} threshold, held-out only):")
    print(flipped.to_string(index=False))
else:
    print(f"\n  All configs agree on every held-out ZIP.")

out_path = INPUT.parent / "density_comparison_fixed.csv"
compare.to_csv(out_path, index=False)
print(f"\n  Saved: {out_path}")

print("\n" + "=" * 70)
print("  PROPOSAL TAKEAWAY (recomputed on apples-to-apples basis)")
print("=" * 70)
auc_A = results[config_names[0]]["test_auc"]
auc_C = results[config_names[2]]["test_auc"]
delta = auc_C - auc_A
agree_pct = compare["all_agree"].mean() * 100 if len(compare) else float("nan")

print(f"""
  AUC with density    : {auc_C:.3f}
  AUC without density : {auc_A:.3f}
  Delta               : {delta:+.3f}

  Held-out ZIP agreement : {agree_pct:.1f}% (n={len(compare)}) — compare this,
  not the old in-sample 100% number, against the decision rule below.

  If delta < 0.02 and agreement > 95%:
    → Argument to DROP density (subject to the circularity check above).
  If delta >= 0.02 or agreement < 90%:
    → Density adds real out-of-sample signal — but check the circularity
      result before attributing that to "rurality" rather than label leakage.
""")