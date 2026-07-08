"""
check_rfei_outlier.py
======================
Quantifies the rfei.fillna(0) vs rfei_imputed question before deciding
whether the fix matters or is a non-issue.

Run this against nj_zip_features_v2_clean.csv (the INPUT to 03_features.py)
BEFORE re-running 03_features.py with the fix, so you can see exactly
which ZIPs are affected and by how much.
"""

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
INPUT = DATA_DIR / "nj_zip_features_v2_clean.csv"

df = pd.read_csv(INPUT, dtype={"zip": str})
print(f"Loaded {len(df)} zips")

def safe_div(num, denom, fill=np.nan):
    num   = pd.Series(num,   index=df.index) if not isinstance(num,   pd.Series) else num
    denom = pd.Series(denom, index=df.index) if not isinstance(denom, pd.Series) else denom
    return np.where((denom == 0) | denom.isna(), fill, num / denom)

def pct_rank_norm(series):
    return series.rank(pct=True)

# Recreate rfei exactly as 03_features.py does
df["rfei"] = safe_div(
    df["fast_food"] + df["convenience"],
    df["supermarket"] + df["grocery"] + df["produce_market"],
    fill=np.nan
).round(3)

df["rfei_no_data"] = df["rfei"].isna().astype(int)
df["rfei_imputed"] = df["rfei"].fillna(df["rfei"].max())

# ── Step 1: How many ZIPs are actually affected? ──────────────────────────
n_affected = df["rfei_no_data"].sum()
print(f"\nZIPs with rfei == NaN (zero food retailers of ANY kind): "
      f"{n_affected} / {len(df)}  ({n_affected/len(df)*100:.1f}%)")

if n_affected > 0:
    print("\nAffected ZIPs:")
    cols = ["zip", "supermarket", "grocery", "produce_market",
            "fast_food", "convenience", "pct_poverty", "pct_elderly",
            "pct_no_vehicle", "nearest_supermarket_miles"]
    cols = [c for c in cols if c in df.columns]
    print(df.loc[df["rfei_no_data"] == 1, cols].to_string(index=False))

# ── Step 2: Old (fillna(0)) vs new (rfei_imputed) score comparison ───────
for score_name, weight_no_vehicle, weight_dist, weight_transit_or_pov, w_rfei, transit_flag in [
    ("transportation_vuln_score", 0.35, 0.30, 0.20, 0.15, True),
    ("composite_vuln_index",      0.20, 0.25, 0.20, 0.25, False),
]:
    if score_name == "transportation_vuln_score":
        old = (
            pct_rank_norm(df["pct_no_vehicle"]) * 0.35
            + pct_rank_norm(df["nearest_supermarket_miles"]) * 0.30
            + (1 - pct_rank_norm(df["pct_transit"])) * 0.20
            + pct_rank_norm(df["rfei"].fillna(0)) * 0.15
        ) * 100
        new = (
            pct_rank_norm(df["pct_no_vehicle"]) * 0.35
            + pct_rank_norm(df["nearest_supermarket_miles"]) * 0.30
            + (1 - pct_rank_norm(df["pct_transit"])) * 0.20
            + pct_rank_norm(df["rfei_imputed"]) * 0.15
        ) * 100
    else:
        old = (
            pct_rank_norm(df["nearest_supermarket_miles"]) * 0.25
            + pct_rank_norm(df["rfei"].fillna(0)) * 0.25
            + pct_rank_norm(df["pct_poverty"]) * 0.20
            + pct_rank_norm(df["pct_no_vehicle"]) * 0.20
            + pct_rank_norm(df["pct_elderly"]) * 0.10
        ) * 100
        new = (
            pct_rank_norm(df["nearest_supermarket_miles"]) * 0.25
            + pct_rank_norm(df["rfei_imputed"]) * 0.25
            + pct_rank_norm(df["pct_poverty"]) * 0.20
            + pct_rank_norm(df["pct_no_vehicle"]) * 0.20
            + pct_rank_norm(df["pct_elderly"]) * 0.10
        ) * 100

    delta = new - old
    print(f"\n── {score_name} ──────────────────────────────────")
    print(f"Rows changed at all       : {(delta.abs() > 0.01).sum()}")
    print(f"Max increase (more vuln)  : {delta.max():.2f}")
    print(f"Mean |delta| among changed: {delta[delta.abs() > 0.01].abs().mean():.2f}")

    # Rank movement: did any affected ZIP cross into/out of top 20?
    old_rank = old.rank(ascending=False)
    new_rank = new.rank(ascending=False)
    rank_shift = (old_rank - new_rank)  # positive = moved UP (more vulnerable) in new

    moved_into_top20 = df[(new_rank <= 20) & (old_rank > 20)]
    moved_out_of_top20 = df[(old_rank <= 20) & (new_rank > 20)]

    print(f"ZIPs newly in top 20 most vulnerable : {len(moved_into_top20)}")
    if len(moved_into_top20):
        print(moved_into_top20[["zip"]].to_string(index=False))
    print(f"ZIPs dropped out of top 20           : {len(moved_out_of_top20)}")
    if len(moved_out_of_top20):
        print(moved_out_of_top20[["zip"]].to_string(index=False))

print("\n" + "="*60)
print("VERDICT")
print("="*60)
if n_affected == 0:
    print("No ZIPs have rfei == NaN — the fillna choice never fires. Non-issue.")
elif n_affected <= 3:
    print(f"Only {n_affected} ZIP(s) affected — check if they're degenerate "
          f"(near-zero population / PO-box-only) before imputing either way. "
          f"Consider excluding them instead of imputing.")
else:
    print(f"{n_affected} ZIPs affected — worth checking the top-20 rank-shift "
          f"results above to see if this changes any 'most vulnerable' list.")