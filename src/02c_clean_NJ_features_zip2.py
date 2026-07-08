"""
clean_nj_zip_features.py
========================
DSSA 5810 — Progress Report 2
Data cleaning script for nj_zip_features_v2.csv

Cleaning steps performed:
  1.  Duplicate detection and deduplication (ZIP-level)
  2.  Missing value detection and documentation
  3.  Sentinel value detection and replacement (mRFEI_wic == -1)
  4.  Type conversion (county_fips, nj_mun_code, mcd_geoid → proper types)
  5.  Regularization of string columns (county, municipality, zip)
  6.  Out-of-range detection for percentage columns
  7.  Logical consistency checks (snap_supermarkets <= snap_stores)
  8.  Geolocation parsing — extract lat/lon from WKT POINT strings
  9.  Constant / near-zero-variance column flagging (liquor_store, bar)
  10. Final shape and missing-value summary

Output: data/nj_zip_features_v2_clean.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
ROOT_DIR  = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
INPUT  = DATA_DIR / "nj_zip_features_v2.csv"
OUTPUT = DATA_DIR / "nj_zip_features_v2_clean.csv"

DIVIDER = "\n" + "=" * 65


def section(title: str) -> None:
    print(f"{DIVIDER}\n  {title}\n" + "=" * 65)


def status(msg: str) -> None:
    print(f"  {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────
section("LOAD")

df = pd.read_csv(INPUT, dtype={"zip": str})
status(f"Loaded  : {INPUT}")
status(f"Shape   : {df.shape[0]} rows × {df.shape[1]} columns")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Duplicate detection and deduplication
# ─────────────────────────────────────────────────────────────────────────────
section("STEP 1 — Duplicate Detection & Deduplication")

# Full-row duplicates
full_dupes = df.duplicated().sum()
status(f"Exact full-row duplicates found : {full_dupes}")
if full_dupes > 0:
    df = df.drop_duplicates()
    status(f"Dropped {full_dupes} exact duplicate rows.")
else:
    status("No exact full-row duplicates — no rows dropped.")

# ZIP-level duplicates (key column)
zip_dupes_before = df.duplicated(subset="zip").sum()
status(f"Duplicate ZIP rows found        : {zip_dupes_before}")
if zip_dupes_before > 0:
    df = df.drop_duplicates(subset="zip", keep="first")
    status(f"Kept first occurrence of each ZIP. Rows after dedup: {len(df)}")
else:
    status("No duplicate ZIP values — no rows dropped.")

status(f"Shape after deduplication: {df.shape[0]} rows × {df.shape[1]} columns")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Missing value detection and documentation
# ─────────────────────────────────────────────────────────────────────────────
section("STEP 2 — Missing Value Detection")

missing_counts = df.isnull().sum()
missing_cols   = missing_counts[missing_counts > 0].sort_values(ascending=False)

if missing_cols.empty:
    status("No missing values detected in any column.")
else:
    status(f"{len(missing_cols)} columns contain missing values:\n")
    print(f"  {'Column':<45} {'Missing':>7}  {'% of rows':>10}")
    print("  " + "-" * 65)
    for col, n in missing_cols.items():
        pct = n / len(df) * 100
        print(f"  {col:<45} {n:>7}  {pct:>9.1f}%")

# Grouped summary by data source
status("\n  Missing by data-source group:")
groups = {
    "ACS demographics (B-series + pct_ cols)": [
        c for c in missing_cols.index
        if c.startswith("pct_") or c.startswith("Population")
        or c.startswith("Median") or c.startswith("Households")
        or c.startswith("Total") or c.startswith("Civilian")
        or c.startswith("Workers") or c.startswith("Owner")
        or c.startswith("Single") or c.startswith("Vacant")
    ],
    "USDA / FARA": [c for c in missing_cols.index if c.startswith("usda_")],
    "Crosswalk (municipality overlap)": [
        c for c in missing_cols.index
        if "municipality" in c or "zip_in" in c
    ],
}
for grp, cols in groups.items():
    grp_cols = [c for c in cols if c in missing_cols.index]
    if grp_cols:
        status(f"    {grp}: {len(grp_cols)} columns affected")

# Policy: missing values are RETAINED (no rows removed per project requirement).
# NaN is meaningful in ACS/FARA columns — small ZIPs fall below Census
# disclosure thresholds. Downstream models must handle NaN via imputation
# or complete-case analysis. Missing crosswalk overlap pcts reflect ZIPs
# that span multiple municipalities and were not resolved.
status("\n  Policy: missing values are retained as NaN (no rows removed).")
status("  Downstream models must handle NaN via imputation or complete-case analysis.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Sentinel value detection and replacement
# ─────────────────────────────────────────────────────────────────────────────
section("STEP 3 — Sentinel Value Detection & Replacement")

# mRFEI_wic uses -1 as a sentinel meaning "no denominator (no food retailers)"
# It is NOT a valid ratio value and must be replaced with NaN before modeling.
sentinel_col   = "mRFEI_wic"
sentinel_value = -1

if sentinel_col in df.columns:
    n_sentinel = (df[sentinel_col] == sentinel_value).sum()
    status(f"'{sentinel_col}' sentinel (-1) occurrences : {n_sentinel} / {len(df)} rows")
    df[sentinel_col] = df[sentinel_col].replace(sentinel_value, np.nan)
    n_after = (df[sentinel_col] == sentinel_value).sum()
    status(f"Replaced with NaN. Remaining -1 values    : {n_after}")
    status(f"'{sentinel_col}' valid range after replacement: "
           f"[{df[sentinel_col].min():.2f}, {df[sentinel_col].max():.2f}]")
else:
    status(f"Column '{sentinel_col}' not found — skipping.")

# ACS Census uses -666666666 as a suppression sentinel; should have been
# caught in 01_load_data.py but we verify here.
acs_sentinel = -666666666
num_cols = df.select_dtypes(include=[np.number]).columns
acs_hits  = {col: (df[col] == acs_sentinel).sum()
             for col in num_cols if (df[col] == acs_sentinel).sum() > 0}
if acs_hits:
    for col, n in acs_hits.items():
        df[col] = df[col].replace(acs_sentinel, np.nan)
        status(f"ACS sentinel found and replaced in '{col}': {n} values → NaN")
else:
    status("No ACS suppression sentinels (-666666666) remaining in numeric columns.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Type conversion
# ─────────────────────────────────────────────────────────────────────────────
section("STEP 4 — Type Conversion")

# county_fips: stored as float (e.g. 3.0) — should be zero-padded 3-char string
if "county_fips" in df.columns:
    before_dtype = df["county_fips"].dtype
    df["county_fips"] = (
        df["county_fips"]
        .dropna()
        .astype(int)
        .astype(str)
        .str.zfill(3)
        .reindex(df.index)
    )
    # Re-apply NaN for rows that were NaN before (reindex fills with NaN)
    status(f"county_fips: {before_dtype} → str (zero-padded to 3 digits, e.g. '003')")
    status(f"  Sample values: {df['county_fips'].dropna().head(5).tolist()}")

# nj_mun_code: stored as float (e.g. 1225.0) — should be integer string
if "nj_mun_code" in df.columns:
    before_dtype = df["nj_mun_code"].dtype
    df["nj_mun_code"] = (
        df["nj_mun_code"]
        .apply(lambda x: str(int(x)) if pd.notna(x) else np.nan)
    )
    status(f"nj_mun_code: {before_dtype} → str (integer, e.g. '1225')")
    status(f"  Sample values: {df['nj_mun_code'].dropna().head(5).tolist()}")

# mcd_geoid: stored as float (e.g. 3402382000.0) — should be string GEOID
if "mcd_geoid" in df.columns:
    before_dtype = df["mcd_geoid"].dtype
    df["mcd_geoid"] = (
        df["mcd_geoid"]
        .apply(lambda x: str(int(x)) if pd.notna(x) else np.nan)
    )
    status(f"mcd_geoid: {before_dtype} → str (e.g. '3402382000')")
    status(f"  Sample values: {df['mcd_geoid'].dropna().head(5).tolist()}")

# state_county_fips: may also be float — convert to string
if "state_county_fips" in df.columns:
    before_dtype = df["state_county_fips"].dtype
    if before_dtype != object:
        df["state_county_fips"] = (
            df["state_county_fips"]
            .apply(lambda x: str(int(x)) if pd.notna(x) else np.nan)
        )
        status(f"state_county_fips: {before_dtype} → str")
    else:
        status(f"state_county_fips: already str — no change")

# USDA binary flag columns: ensure integer (0/1), not float
usda_binary_cols = [
    "usda_lila_1_10", "usda_lila_half_10", "usda_lila_1_20",
    "usda_la_1_10", "usda_la_half_10", "usda_urban",
]
for col in usda_binary_cols:
    if col in df.columns:
        before = df[col].dtype
        df[col] = df[col].astype("Int64")   # nullable integer (handles NaN)
        status(f"{col}: {before} → Int64 (nullable integer flag)")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Regularization of string columns
# ─────────────────────────────────────────────────────────────────────────────
section("STEP 5 — String Regularization")

# zip: already 5-digit zero-padded strings — verify and confirm
non_5digit = df["zip"].str.len() != 5
non_numeric = ~df["zip"].str.match(r"^\d{5}$")
status(f"zip — non-5-digit values : {non_5digit.sum()}")
status(f"zip — non-numeric values : {non_numeric.sum()}")
if non_5digit.sum() == 0 and non_numeric.sum() == 0:
    status("zip format is clean (all 5-digit numeric strings).")

# county: strip leading/trailing whitespace and title-case
if "county" in df.columns:
    before_unique = df["county"].dropna().nunique()
    df["county"] = df["county"].str.strip().str.title()
    after_unique  = df["county"].dropna().nunique()
    status(f"county: stripped whitespace, applied title case.")
    status(f"  Unique values before: {before_unique}  after: {after_unique}")
    status(f"  Values: {sorted(df['county'].dropna().unique().tolist())}")

# municipality: strip whitespace
if "municipality" in df.columns:
    before_unique = df["municipality"].dropna().nunique()
    df["municipality"] = df["municipality"].str.strip()
    after_unique  = df["municipality"].dropna().nunique()
    status(f"municipality: stripped whitespace.")
    status(f"  Unique values before: {before_unique}  after: {after_unique}")

# Geolocation: already WKT POINT strings — validate format
if "Geolocation" in df.columns:
    bad_geo = ~df["Geolocation"].str.match(r"^POINT \(-?\d+\.\d+ -?\d+\.\d+\)$", na=False)
    n_bad   = bad_geo.sum()
    n_null  = df["Geolocation"].isnull().sum()
    status(f"Geolocation: {n_null} nulls, {n_bad - n_null} malformed POINT strings.")
    if n_bad == 0:
        status("Geolocation format is clean.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Out-of-range detection for percentage columns
# ─────────────────────────────────────────────────────────────────────────────
section("STEP 6 — Out-of-Range Detection (Percentage Columns)")

# All pct_ columns and PLACES % columns must be in [0, 100].
# Exclude ACS raw count columns whose names contain "%" as part of the
# income-threshold label (e.g. "Households Rent 40-49% of Income") — those
# are counts in absolute households, not percentage rates.
ACS_COUNT_COLS_WITH_PCT_IN_NAME = {
    "Households Rent 40-49% of Income",
    "Households Rent 50%+ of Income",
}

pct_cols = (
    [c for c in df.columns if c.startswith("pct_")]
    + [c for c in df.columns
       if "%" in c and c not in ACS_COUNT_COLS_WITH_PCT_IN_NAME]
)

any_oor = False
print(f"  {'Column':<50} {'<0':>5}  {'>100':>5}  {'Min':>8}  {'Max':>8}")
print("  " + "-" * 80)
for col in pct_cols:
    if col not in df.columns:
        continue
    series  = pd.to_numeric(df[col], errors="coerce")
    n_low   = (series < 0).sum()
    n_high  = (series > 100).sum()
    col_min = series.min()
    col_max = series.max()
    flag    = "  ← OUT OF RANGE" if (n_low > 0 or n_high > 0) else ""
    print(f"  {col:<50} {n_low:>5}  {n_high:>5}  {col_min:>8.2f}  {col_max:>8.2f}{flag}")
    if n_low > 0 or n_high > 0:
        any_oor = True

if not any_oor:
    status("\n  All percentage columns are within [0, 100]. No out-of-range values.")
else:
    status("\n  Out-of-range values found (see above). These should be reviewed before modeling.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Logical consistency checks
# ─────────────────────────────────────────────────────────────────────────────
section("STEP 7 — Logical Consistency Checks")

checks = [
    # (description, boolean mask of VIOLATIONS)
    (
        "snap_supermarkets <= snap_stores",
        df["snap_supermarkets"] > df["snap_stores"]
    ),
    (
        "pct_no_vehicle in [0,100] (non-null)",
        df["pct_no_vehicle"].dropna().gt(100) | df["pct_no_vehicle"].dropna().lt(0)
    ),
    (
        "nearest_supermarket_miles >= 0",
        df["nearest_supermarket_miles"] < 0
    ),
    (
        "nearest_fastfood_miles >= 0",
        df["nearest_fastfood_miles"] < 0
    ),
    (
        "nearest_convenience_miles >= 0",
        df["nearest_convenience_miles"] < 0
    ),
    (
        "area_sqmi > 0",
        df["area_sqmi"] <= 0
    ),
    (
        "pop_density >= 0 (non-null)",
        df["pop_density"].dropna().lt(0)
    ),
    (
        "Total Population_acs >= 0 (non-null)",
        df["Total Population_acs"].dropna().lt(0)
    ),
    (
        "Owner-Occupied Housing Units <= Total Occupied Housing Units (non-null)",
        (
            df["Owner-Occupied Housing Units"].notna()
            & df["Total Occupied Housing Units"].notna()
            & (df["Owner-Occupied Housing Units"] > df["Total Occupied Housing Units"])
        )
    ),
]

all_passed = True
for desc, mask in checks:
    n_fail = int(mask.sum())
    result = "PASS" if n_fail == 0 else f"FAIL — {n_fail} violation(s)"
    status(f"  {desc:<60} {result}")
    if n_fail > 0:
        all_passed = False

if all_passed:
    status("\n  All logical consistency checks passed.")
else:
    status("\n  Some checks failed — review violations before modeling.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Geolocation parsing: extract lat/lon from WKT POINT strings
# ─────────────────────────────────────────────────────────────────────────────
section("STEP 8 — Geolocation Parsing (WKT → latitude / longitude)")

if "Geolocation" in df.columns:
    extracted = df["Geolocation"].str.extract(
        r"POINT \((?P<longitude>-?\d+\.\d+) (?P<latitude>-?\d+\.\d+)\)"
    ).astype(float)

    df = pd.concat([df, extracted[["longitude", "latitude"]]], axis=1)

    n_parsed = df["latitude"].notna().sum()
    n_fail   = df["Geolocation"].notna().sum() - n_parsed
    status(f"Parsed {n_parsed} lat/lon pairs from 'Geolocation' column.")
    status(f"Failed to parse : {n_fail}")
    status(f"latitude  range : [{df['latitude'].min():.4f}, {df['latitude'].max():.4f}]")
    status(f"longitude range : [{df['longitude'].min():.4f}, {df['longitude'].max():.4f}]")

    # Sanity check: all points should be within NJ bounding box
    nj_lat = (38.8, 41.4)
    nj_lon = (-75.6, -73.8)
    out_of_nj = (
        df["latitude"].lt(nj_lat[0])  | df["latitude"].gt(nj_lat[1])
        | df["longitude"].lt(nj_lon[0]) | df["longitude"].gt(nj_lon[1])
    ).sum()
    status(f"Points outside NJ bounding box: {out_of_nj}")
    if out_of_nj == 0:
        status("All coordinates fall within NJ bounding box.")
else:
    status("'Geolocation' column not found — skipping lat/lon extraction.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — Constant / near-zero-variance column flagging
# ─────────────────────────────────────────────────────────────────────────────
section("STEP 9 — Constant & Near-Zero-Variance Column Flagging")

status("Columns are NOT dropped — they are flagged for modeling decisions.")
status("Constant columns carry zero predictive information.\n")

num_df = df.select_dtypes(include=[np.number])
constant_cols     = []
low_variance_cols = []

for col in num_df.columns:
    unique_vals = num_df[col].dropna().nunique()
    if unique_vals <= 1:
        constant_cols.append((col, unique_vals, num_df[col].dropna().unique().tolist()))
    elif unique_vals <= 3:
        low_variance_cols.append((col, unique_vals, num_df[col].dropna().unique().tolist()))

if constant_cols:
    status("  CONSTANT columns (1 unique value — zero variance):")
    for col, n, vals in constant_cols:
        status(f"    {col:<35} unique={n}  values={vals}")
else:
    status("  No fully constant columns found.")

if low_variance_cols:
    status("\n  LOW-VARIANCE columns (2–3 unique values):")
    for col, n, vals in low_variance_cols:
        status(f"    {col:<35} unique={n}  values={vals}")
else:
    status("  No low-variance columns found.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 10 — Final shape and missing-value summary
# ─────────────────────────────────────────────────────────────────────────────
section("STEP 10 — Final Summary")

status(f"Final shape  : {df.shape[0]} rows × {df.shape[1]} columns")
status(f"Unique ZIPs  : {df['zip'].nunique()}")

final_missing = df.isnull().sum()
final_missing = final_missing[final_missing > 0]
status(f"Columns with remaining NaN: {len(final_missing)} (all intentional — see Step 2)")

status("\n  Retained NaN summary:")
print(f"  {'Column':<45} {'NaN count':>10}  {'Reason'}")
print("  " + "-" * 85)

nan_reasons = {
    "Median Contract Rent":            "ACS suppression (small ZIP / disclosure threshold)",
    "Median Household Income_acs":     "ACS suppression",
    "Median Home Value":               "ACS suppression",
    "pct_transit":                     "ACS suppression (zero commuter universe)",
    "pct_snap":                        "ACS suppression",
    "pct_no_vehicle":                  "ACS suppression",
    "pct_poverty":                     "ACS suppression",
    "pct_elderly":                     "ACS suppression",
    "Median Age_acs":                  "ACS suppression",
    "usda_median_income":              "No FARA tract mapped to ZIP",
    "usda_poverty_rate":               "No FARA tract mapped to ZIP",
    "pct_zip_in_municipality":         "ZIP spans multiple municipalities (not resolved)",
    "pct_municipality_in_zip":         "ZIP spans multiple municipalities (not resolved)",
    "mRFEI_wic":                       "No food retailers in ZIP (was sentinel -1, now NaN)",
}

for col, n in final_missing.sort_values(ascending=False).items():
    reason = nan_reasons.get(col, "ACS suppression or multi-source join gap")
    print(f"  {col:<45} {n:>10}  {reason}")

status("\n  Column dtype summary after cleaning:")
print(f"  {df.dtypes.value_counts().to_string()}")


# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
section("SAVE")

df.to_csv(OUTPUT, index=False)
status(f"Saved → {OUTPUT}")
status(f"Shape  : {df.shape}")

print(f"\n{'=' * 65}")
print("  Cleaning complete.")
print(f"{'=' * 65}\n")