"""
check_pop_density_proxy.py
==========================
Determines whether pop_density (dominant feature in the ablation test,
importance 0.302, delta AUC +0.058/+0.087) is a genuine independent
urbanicity signal, or whether it's functioning as a smooth proxy for
food-environment density -- which would mean Model 3's health
regressions ("income dominates, food environment barely matters")
could be partly an artifact of pop_density already absorbing some of
that food-environment variance before rfei/nj_swamp_score/etc. get a
turn.

Run this against nj_zip_features_v5.csv (or whatever your 04_model.py
INPUT currently points to).
"""

from pathlib import Path
import pandas as pd
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
INPUT = DATA_DIR / "nj_zip_features_v5.csv"

df = pd.read_csv(INPUT, dtype={"zip": str})
print(f"Loaded {len(df)} zips\n")

# ── Columns to check pop_density against ──────────────────────────────────
# Group 1: the two columns already excluded from DESERT_FEATURES as
#          leakage risks (these DEFINE is_desert_5mi, so high correlation
#          here is expected and not itself alarming for Model 1).
proximity_cols = ["nearest_supermarket_miles", "supermarkets_within_5mi"]

# Group 2: raw store counts that feed HEALTH_FEATURES and TYPOLOGY_FEATURES
# directly (rfei, mrfei, nj_swamp_score, dollar_store_ratio are all built
# from these). High correlation here is the actual question for Model 3.
store_cols = ["supermarket", "fast_food", "convenience", "grocery",
              "produce_market", "dollar_store", "restaurant"]
store_cols = [c for c in store_cols if c in df.columns]

# Group 3: the derived food-environment indexes themselves, which is the
# most direct test -- if pop_density correlates strongly with rfei/mrfei
# themselves, it's not just "near stores," it's actively redundant with
# the indexes HEALTH_FEATURES is supposed to be testing independently.
index_cols = ["rfei", "mrfei", "nj_swamp_score", "dollar_store_ratio",
              "swamp_method_count"]
index_cols = [c for c in index_cols if c in df.columns]

all_check_cols = proximity_cols + store_cols + index_cols
all_check_cols = [c for c in all_check_cols if c in df.columns]

corr = df[["pop_density"] + all_check_cols].corr()["pop_density"].drop("pop_density")
corr = corr.sort_values(key=abs, ascending=False)

print("── pop_density correlation with proximity features (Model 1 exclusions) ──")
print(corr[corr.index.isin(proximity_cols)].round(3).to_string())

print("\n── pop_density correlation with raw store counts (feed HEALTH_FEATURES indexes) ──")
print(corr[corr.index.isin(store_cols)].round(3).to_string())

print("\n── pop_density correlation with food-environment INDEXES directly ──")
print(corr[corr.index.isin(index_cols)].round(3).to_string())

# ── Verdict ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("VERDICT")
print("=" * 60)

strong_index_corr = corr[corr.index.isin(index_cols)].abs()
strong_store_corr = corr[corr.index.isin(store_cols)].abs()

if len(strong_index_corr) and strong_index_corr.max() > 0.6:
    worst = strong_index_corr.idxmax()
    print(f"pop_density correlates strongly with '{worst}' "
          f"(r = {corr[worst]:+.3f}).")
    print("This supports the proxy hypothesis: pop_density may be absorbing")
    print("food-environment signal in Model 3's regressions, making the")
    print("'income dominates, food environment barely matters' result partly")
    print("an artifact of feature ordering/collinearity rather than a clean")
    print("finding. Consider an ablation on Model 3 specifically (with vs.")
    print("without pop_density) before reporting those R^2/importance numbers.")
elif len(strong_store_corr) and strong_store_corr.max() > 0.6:
    worst = strong_store_corr.idxmax()
    print(f"pop_density correlates strongly with raw store count '{worst}' "
          f"(r = {corr[worst]:+.3f}), though not with the derived indexes "
          f"themselves.")
    print("Mixed picture: dense areas simply have more of every store type,")
    print("which is expected urbanicity, but check whether this still lets")
    print("pop_density stand in for what rfei/mrfei are supposed to measure.")
else:
    print("No strong correlation (|r| > 0.6) between pop_density and either")
    print("store counts or food-environment indexes.")
    print("This weakens the proxy hypothesis: pop_density's dominance in the")
    print("ablation test and Model 3's regressions is more likely a genuine")
    print("independent urbanicity signal, not food-environment variance")
    print("routed through population density. Still worth reporting as a")
    print("substantive finding (urbanicity > food access as a predictor),")
    print("rather than treating it as a leakage artifact to explain away.")