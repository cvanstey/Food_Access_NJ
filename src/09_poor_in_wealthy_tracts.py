"""
09_poor_in_wealthy_tracts.py
==================
Research question: among ZIPs that gain "food desert" status only when the
poverty requirement is dropped (usda_la_1_10=1, usda_lila_1_10=0) — ZIPs
that look affluent at the ZIP-level average — where are the actual poor
residents? Two lenses:

  1. ABSOLUTE COUNT — rank gained-only ZIPs by raw population below poverty,
     not by rate. A ZIP with a low poverty RATE can still contain a large
     number of poor people if the ZIP is populous enough; ranking by rate
     alone erases them.

  2. TRACT-LEVEL POCKETS — a ZIP's poverty rate is a population-weighted
     average across the census tracts inside it. A "6% poverty" ZIP can
     contain one tract at 25% poverty sitting next to several tracts near
     0%. This pulls the underlying tracts for each gained-only ZIP (via the
     HUD ZIP<->tract crosswalk, same source used in 01_load_data.py) and
     flags any tract whose poverty rate sits well above the ZIP's own
     average — the pocket a ZIP-level number would hide.

This does NOT argue for removing food-desert status from anyone. It
identifies where poor residents are undercounted by ZIP-level aggregation,
which is the opposite direction of concern: are poor people being missed
because they live inside a ZIP that reads as wealthy on average?

Inputs (same sources as 01_load_data.py Section 10 — rebuilt here at
tract level since the pipeline only persists the ZIP-aggregated fara_agg.csv,
not the intermediate tract-level table):
    data/nj_zip_features_v5.csv
    data/FoodAccessResearchAtlasData2019.xlsx
    data/ZIP_TRACT_122025.xlsx

Run this AFTER merge_sources.py.
"""


from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_utils import normalize_zip

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

FEATURES_PATH = DATA_DIR / "nj_zip_features_v5.csv"
FARA_PATH     = DATA_DIR / "FoodAccessResearchAtlasData2019.xlsx"
HUD_PATH      = DATA_DIR / "ZIP_TRACT_122025.xlsx"

FLAG_WITH_POVERTY    = "usda_lila_1_10"
FLAG_WITHOUT_POVERTY = "usda_la_1_10"

# A tract is flagged as a "hidden poor pocket" if its poverty rate exceeds
# the ZIP's own average by at least this many percentage points. 10pp is a
# starting threshold, not a definitive cutoff -- tune based on what the
# tract-level distribution actually looks like once you see it.
POCKET_THRESHOLD_PP = 10.0


def section(title: str):
    print(f"\n{'─' * 72}\n{title}\n{'─' * 72}")


def load_gained_only_zips() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_PATH, dtype={"zip": str})
    gained = df[
        (df[FLAG_WITH_POVERTY] == 0) & (df[FLAG_WITHOUT_POVERTY] == 1)
    ].copy()
    return gained


def load_fara_tracts() -> pd.DataFrame:
    fara = pd.read_excel(
        FARA_PATH, sheet_name="Food Access Research Atlas",
        dtype={"CensusTract": str},
    )
    fara["CensusTract"] = fara["CensusTract"].str.zfill(11)
    fara_nj = fara[fara["CensusTract"].str.startswith("34")].copy()
    return fara_nj[["CensusTract", "PovertyRate", "Pop2010", "MedianFamilyIncome"]]


def load_zip_tract_crosswalk(valid_zips: set) -> pd.DataFrame:
    hud = pd.read_excel(HUD_PATH, dtype={"ZIP": str, "TRACT": str})
    hud.columns = hud.columns.str.strip().str.lower()
    hud["tract"] = hud["tract"].str.zfill(11)
    hud["zip"] = normalize_zip(hud["zip"])
    hud = hud[hud["tract"].str.startswith("34")]
    hud = hud[hud["zip"].isin(valid_zips)]
    return hud[["zip", "tract", "res_ratio"]].rename(columns={"tract": "CensusTract"})


def main():
    # ── 1. Absolute count ranking ────────────────────────────────────────────
    section("LENS 1 — Gained-only ZIPs ranked by absolute poor population")

    gained = load_gained_only_zips()
    print(f"  Gained-only ZIPs (loose def. only): {len(gained)}")

    count_col = "Population Below Poverty_acs"
    if count_col not in gained.columns:
        raise SystemExit(f"'{count_col}' not found — check merge_sources.py output columns.")

    ranked_by_count = gained.sort_values(count_col, ascending=False)
    display_cols = ["zip", "municipality", "county", "usda_median_income",
                     "pct_poverty", count_col, "Total Population_acs"]
    display_cols = [c for c in display_cols if c in ranked_by_count.columns]

    print(f"\n  Top 20 gained-only ZIPs by RAW poor population count "
          f"(not rate):\n")
    print(ranked_by_count[display_cols].head(20).to_string(index=False))

    # Contrast: does ranking by count vs. by rate surface a different set?
    top20_by_count = set(ranked_by_count.head(20)["zip"])
    top20_by_rate  = set(gained.sort_values("pct_poverty", ascending=False).head(20)["zip"])
    only_in_count  = top20_by_count - top20_by_rate

    print(f"\n  ZIPs in top-20-by-COUNT but NOT top-20-by-RATE: {len(only_in_count)}")
    print("  These are exactly the ZIPs a rate-only analysis would miss —")
    print("  low poverty rate, but a large enough population that the raw")
    print("  number of poor residents is still substantial.")
    if only_in_count:
        missed = ranked_by_count[ranked_by_count["zip"].isin(only_in_count)]
        print(missed[display_cols].to_string(index=False))

    out_path = DATA_DIR / "gained_only_ranked_by_poor_count.csv"
    ranked_by_count[display_cols].to_csv(out_path, index=False)
    print(f"\n  Saved -> {out_path}")

    # ── 2. Tract-level pockets ───────────────────────────────────────────────
    section("LENS 2 — Tract-level poverty pockets inside gained-only ZIPs")

    fara_tracts = load_fara_tracts()
    print(f"  NJ census tracts loaded: {len(fara_tracts)}")

    valid_zips = set(gained["zip"])
    crosswalk = load_zip_tract_crosswalk(valid_zips)
    print(f"  ZIP-tract crosswalk rows (gained-only ZIPs): {len(crosswalk)}")

    merged = crosswalk.merge(fara_tracts, on="CensusTract", how="left")
    merged = merged.merge(
        gained[["zip", "pct_poverty", "municipality", "county",
                "usda_median_income", "Total Population_acs"]],
        on="zip", how="left",
    )

    n_no_tract_data = merged["PovertyRate"].isna().sum()
    if n_no_tract_data:
        print(f"  [NOTE] {n_no_tract_data} zip-tract rows have no matching "
              f"FARA tract poverty data — excluded from pocket detection.")
    merged = merged.dropna(subset=["PovertyRate", "pct_poverty"])

    merged["poverty_gap_pp"] = merged["PovertyRate"] - merged["pct_poverty"]
    merged["est_poor_in_tract"] = (merged["PovertyRate"] / 100 * merged["Pop2010"]).round(0)

    pockets = merged[merged["poverty_gap_pp"] >= POCKET_THRESHOLD_PP].copy()
    pockets = pockets.sort_values("poverty_gap_pp", ascending=False)

    print(f"\n  Tracts flagged as hidden poverty pockets "
          f"(tract rate >= ZIP rate + {POCKET_THRESHOLD_PP:.0f}pp): {len(pockets)}")
    print(f"  Spanning {pockets['zip'].nunique()} distinct gained-only ZIPs\n")

    pocket_display = [
        "zip", "municipality", "county", "CensusTract",
        "pct_poverty", "PovertyRate", "poverty_gap_pp",
        "Pop2010", "est_poor_in_tract", "usda_median_income",
    ]
    print(pockets[pocket_display].head(25).to_string(index=False))

    pocket_out_path = DATA_DIR / "hidden_poverty_pockets.csv"
    pockets[pocket_display].to_csv(pocket_out_path, index=False)
    print(f"\n  Saved -> {pocket_out_path}")

    # ── 3. Summary: which ZIPs have the largest hidden pockets ──────────────
    section("Summary — gained-only ZIPs ranked by total hidden-pocket population")

    zip_pocket_summary = (
        pockets.groupby(["zip", "municipality", "county", "pct_poverty",
                          "usda_median_income"])
        .agg(
            n_pocket_tracts=("CensusTract", "count"),
            max_tract_poverty_rate=("PovertyRate", "max"),
            total_est_poor_in_pockets=("est_poor_in_tract", "sum"),
        )
        .reset_index()
        .sort_values("total_est_poor_in_pockets", ascending=False)
    )

    print(zip_pocket_summary.to_string(index=False))
    summary_out_path = DATA_DIR / "hidden_pocket_zip_summary.csv"
    zip_pocket_summary.to_csv(summary_out_path, index=False)
    print(f"\n  Saved -> {summary_out_path}")

    print(f"\n  These ZIPs are where the ZIP-level 'wealthy' label is most")
    print(f"  misleading: an affluent-reading average sitting on top of a")
    print(f"  tract with real, concentrated poverty inside it.")


if __name__ == "__main__":
    main()
