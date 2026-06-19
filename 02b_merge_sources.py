"""
merge_sources.py
============
Merges all cleaned data sources into a single zip-level feature table.
"""

from pathlib import Path
import geopandas as gpd
import pandas as pd
import pdfplumber
import warnings
from rename_columns import apply_renames
import json
from collections import Counter

warnings.filterwarnings("ignore")

BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR / "data"
OUTPUT     = DATA_DIR / "nj_zip_features_v2.csv"
OSM_PATH   = DATA_DIR / "osm_data.json"
ZIP_PATH   = DATA_DIR / "nj_zip_master.csv"

TIGER_BASE = "https://www2.census.gov/geo/tiger"
TIGER_YEAR = 2021
STATE_FIPS = "34"

NJ_LAT = (38.8, 41.4)
NJ_LON = (-75.6, -73.8)


def log(msg):
    print(f"  {msg}")


def normalize_zip(series):
    return series.astype(str).str.strip().str.zfill(5)


OSM_TYPE_MAP = {
    ("shop", "supermarket"):    "Supermarket",
    ("shop", "grocery"):        "Grocery Store",
    ("shop", "convenience"):    "Convenience Store",
    ("amenity", "fast_food"):   "Fast Food",
    ("amenity", "restaurant"):  "Restaurant",
    ("shop", "variety_store"):  "Variety Store",
    ("shop", "greengrocer"):    "Produce Market",
    ("shop", "butcher"):        "Meat / Seafood",
    ("shop", "seafood"):        "Meat / Seafood",
    ("shop", "bakery"):         "Bakery",
    ("shop", "deli"):           "Deli",
    ("shop", "health_food"):    "Health Food Store",
    ("shop", "organic"):        "Health Food Store",
    ("shop", "farm"):           "Farm Stand",
    ("amenity", "marketplace"): "Farmers Market",
    ("shop", "chemist"):        "Pharmacy",
    ("shop", "drugstore"):      "Pharmacy",
    ("shop", "wholesale"):      "Wholesale Club",
}


# ── 0. Load OSM once ──────────────────────────────────────────────────────────
print("\n── 0. Loading OSM data")
if not OSM_PATH.exists():
    raise FileNotFoundError(f"OSM data not found at {OSM_PATH} — run ingestion script first")

with open(OSM_PATH, "r", encoding="utf-8") as f:
    osm_data = json.load(f)

elements = osm_data.get("elements", [])
log(f"OSM elements loaded: {len(elements):,}")

# ── 1. Base zip spine ─────────────────────────────────────────────────────────
print("\n── 1. Base zip spine")
zcta     = gpd.read_file(DATA_DIR / "zcta_nj.gpkg")
base     = pd.DataFrame({"zip": zcta["zip"].astype(str).str.zfill(5).unique()})
base     = base.sort_values("zip").reset_index(drop=True)
base = base.drop_duplicates(subset="zip").reset_index(drop=True)

osm_counts = pd.read_csv(DATA_DIR / "osm_counts.csv", dtype={"zip": str})
osm_counts["zip"] = osm_counts["zip"].str.zfill(5)
base     = base.merge(osm_counts, on="zip", how="left")

for col in osm_counts.columns.drop("zip"):
    base[col] = base[col].fillna(0).astype(int)

log(f"Base table: {len(base):,} zips")

# ── 2. ACS ────────────────────────────────────────────────────────────────────
print("\n── 2. ACS demographics")
acs_df = pd.read_csv(DATA_DIR / "acs_df.csv")
acs_df["zip"] = normalize_zip(acs_df["zip"])
acs_df = acs_df.loc[:, ~acs_df.columns.str.contains("^Unnamed")]

# ── 2b. Derive pct_ columns from raw ACS counts ───────────────────────────
elderly_cols = [
    "B01001_020E","B01001_021E","B01001_022E",
    "B01001_023E","B01001_024E","B01001_025E",
    "B01001_044E","B01001_045E","B01001_046E",
    "B01001_047E","B01001_048E","B01001_049E",
]

# Only compute if the raw columns exist
if all(c in acs_df.columns for c in elderly_cols):
    acs_df["pct_elderly"] = (
        acs_df[elderly_cols].sum(axis=1) / acs_df["B01003_001E"] * 100
    )
else:
    log("⚠ B01001 age columns missing — pct_elderly will be NaN")
    acs_df["pct_elderly"] = float("nan")

acs_df["pct_transit"] = (
    acs_df["B08301_010E"] / acs_df["B08301_001E"] * 100
)

# Standardize names so 03_features.py can find them
acs_df = acs_df.rename(columns={
    "poverty_rate":    "pct_poverty",
    "no_vehicle_rate": "pct_no_vehicle",
    "snap_rate":       "pct_snap",
})

# ── 3. CDC PLACES ─────────────────────────────────────────────────────────────
print("\n── 3. CDC PLACES")
places_df = pd.read_csv(DATA_DIR / "places_df.csv")

places_df["zip"] = normalize_zip(places_df["zip"])

places_df = places_df.loc[
    :, ~places_df.columns.str.contains("^Unnamed")
]


# ── 4. Crosswalk ──────────────────────────────────────────────────────────────
print("\n── 4. Crosswalk")
tax_df = pd.read_csv(DATA_DIR / "nj_zip_complete.csv",
                     dtype={"zip_code": str, "county_fips": str})
tax_df = tax_df.rename(columns={"zip_code": "zip"})
tax_df["zip"] = normalize_zip(tax_df["zip"])
tax_df["county_fips"] = tax_df["county_fips"].astype(str).str.zfill(3)

# ── Filter to NJ counties only ────────────────────────────────────────────────
nj_county_fips = {str(i).zfill(3) for i in range(1, 42, 2)}
pre = len(tax_df)
tax_df = tax_df[tax_df["county_fips"].isin(nj_county_fips)]
print(f"NJ crosswalk filter: {pre} → {len(tax_df)} rows ({pre - len(tax_df)} dropped)")


tax_df = (tax_df.sort_values("pct_zip_in_municipality", ascending=False)
    .drop_duplicates(subset="zip", keep="first")
)

tax_df = tax_df[[
    "zip", "county", "county_fips", "municipality",
    "nj_mun_code", "mcd_geoid",
    "pct_zip_in_municipality", "pct_municipality_in_zip"
]]


# ── 5. SNAP ───────────────────────────────────────────────────────────────────
print("\n── 5. SNAP retailers")
snap_df = pd.read_csv(DATA_DIR / "snap_retailer_location_data.csv", encoding="utf-8-sig")
snap_nj = snap_df[snap_df["State"].str.strip() == "NJ"].copy()
snap_nj["zip"] = normalize_zip(snap_nj["Zip_Code"])
snap_counts = snap_nj.groupby("zip").size().rename("snap_stores").reset_index()

type_col = next((c for c in snap_nj.columns if "type" in c.lower()), None)
if type_col:
    snap_super = (
        snap_nj[snap_nj[type_col].str.contains("Supermarket|Grocery", na=False)]
        .groupby("zip").size().rename("snap_supermarkets").reset_index()
    )
    snap_counts = snap_counts.merge(snap_super, on="zip", how="left")
    snap_counts["snap_supermarkets"] = snap_counts["snap_supermarkets"].fillna(0)


# ── 6. WIC ────────────────────────────────────────────────────────────────────
print("\n── 6. WIC vendors")
wic_rows = []
with pdfplumber.open(DATA_DIR / "vendorlocations.pdf") as pdf:
    for page in pdf.pages:
        for table in page.extract_tables():
            for row in table:
                if row and any(row):
                    wic_rows.append(row)

wic = pd.DataFrame(wic_rows)
wic.columns = range(len(wic.columns))
wic.columns = wic.iloc[1].astype(str)
wic = wic.iloc[2:].copy()
wic = wic[wic["eWIC Status"] == "eWIC Certified"]
wic = wic.rename(columns={"Zip": "zip"})
wic["zip"] = normalize_zip(wic["zip"])
wic_counts = wic.groupby("zip").size().rename("wic_stores").reset_index()


# ── 7. OSM Produce Markets ────────────────────────────────────────────────────
print("\n── 7. OSM produce markets")

produce_records = []
for el in elements:
    tags = el.get("tags", {})
    if tags.get("shop") != "greengrocer":
        continue
    lat = el.get("lat")
    lon = el.get("lon")
    if lat is None and "center" in el:
        lat = el["center"]["lat"]
        lon = el["center"]["lon"]
    if lat is None or lon is None:
        continue
    if not (NJ_LAT[0] <= lat <= NJ_LAT[1] and NJ_LON[0] <= lon <= NJ_LON[1]):
        continue
    produce_records.append({"lat": lat, "lon": lon})

produce_df = pd.DataFrame(produce_records)
log(f"Produce market nodes found: {len(produce_df)}")

if len(produce_df) > 0:
    zcta = gpd.read_file(DATA_DIR / "zcta_nj.gpkg")  # no prefix filter
    zcta = zcta.to_crs(epsg=4326)

    produce_gdf = gpd.GeoDataFrame(
        produce_df,
        geometry=gpd.points_from_xy(produce_df["lon"], produce_df["lat"]),
        crs="EPSG:4326"
    )

    joined = gpd.sjoin(produce_gdf, zcta, how="left", predicate="within")
    produce_counts = (
        joined.groupby("zip").size()
        .rename("produce_market")
        .reset_index()
    )
    produce_counts["zip"] = normalize_zip(produce_counts["zip"])
    log(f"ZIPs with produce markets : {produce_counts['zip'].nunique()}")
    log(f"Total produce market count: {produce_counts['produce_market'].sum()}")
else:
    log("No produce market nodes found — column will be 0")


nearest_df = pd.read_csv(DATA_DIR / "nj_zip_nearest.csv", dtype={"zip": str})
nearest_df["zip"] = normalize_zip(nearest_df["zip"])
nearest_df = nearest_df[["zip", "nearest_fastfood_miles",
                          "nearest_convenience_miles", "nearest_supermarket_miles"]]

# ── 8. USDA FARA ──────────────────────────────────────────────────────────────
print("\n── 8. USDA FARA (Food Access Research Atlas)")
FARA_PATH = DATA_DIR / "fara_agg.csv"
if not FARA_PATH.exists():
    raise FileNotFoundError(
        f"fara_agg.csv not found at {FARA_PATH} — run progress_report.py first"
    )

fara_df = pd.read_csv(FARA_PATH, dtype={"zip": str})
fara_df["zip"] = normalize_zip(fara_df["zip"])
log(f"FARA rows loaded: {len(fara_df):,}  columns: {len(fara_df.columns)}")

# Binary flag columns — fill missing with 0 (no LILA tract mapped to that ZIP)
FARA_BINARY_COLS = [
    "usda_lila_1_10", "usda_lila_half_10", "usda_lila_1_20",
    "usda_la_1_10",   "usda_la_half_10",   "usda_urban",
]
# Continuous columns — fill missing with NaN (unknown, not zero)
FARA_CONTINUOUS_COLS = [
    "usda_lapop_1_10", "usda_lapop_half_10",
    "usda_poverty_rate", "usda_median_income", "usda_pop2010",
]


# ── 9. MERGE ALL ──────────────────────────────────────────────────────────────
print("\n── 9. Merging")

out = (
    base
    .merge(acs_df,          on="zip", how="left")
    .merge(places_df,       on="zip", how="left")
    .merge(tax_df,          on="zip", how="left")
    .merge(snap_counts,     on="zip", how="left")
    .merge(wic_counts,      on="zip", how="left")
    .merge(fara_df,         on="zip", how="left")
    .merge(nearest_df, on="zip", how="left")
)

# Integer store counts
out["snap_stores"]    = out["snap_stores"].fillna(0).astype(int)
out["wic_stores"]     = out["wic_stores"].fillna(0).astype(int)
out["produce_market"] = out["produce_market"].fillna(0).astype(int)

if "county_fips" in out.columns:
    out["state_county_fips"] = "34" + out["county_fips"].astype(str).str.zfill(3)

if "snap_supermarkets" in out.columns:
    out["snap_supermarkets"] = out["snap_supermarkets"].fillna(0).astype(int)

# FARA binary flags: 0 = no LILA tract in ZIP (not unknown)
for col in FARA_BINARY_COLS:
    if col in out.columns:
        out[col] = out[col].fillna(0).astype(int)

# FARA population counts: 0 = no low-access population mapped to ZIP
for col in ["usda_lapop_1_10", "usda_lapop_half_10", "usda_pop2010"]:
    if col in out.columns:
        out[col] = out[col].fillna(0)

# FARA rate/income averages: leave as NaN where no tract data exists
# (usda_poverty_rate, usda_median_income — NaN is meaningful here)

out["mRFEI_wic"] = (
    out["wic_stores"] /
    (out["wic_stores"] + out["fast_food"] + out["convenience"] + out["dollar_store"])
    * 100
).round(2).fillna(-1)

out = apply_renames(out)
out = out.drop(columns=[c for c in out.columns if "95% CI" in c], errors="ignore")

# ── 9b. Compute pop_density from ZCTA area ────────────────────────────────
zcta_area = gpd.read_file(DATA_DIR / "zcta_nj.gpkg").to_crs(epsg=32618)
zcta_area["area_sqmi"] = zcta_area.geometry.area / 2_589_988
zcta_area["zip"] = zcta_area["zip"].astype(str).str.zfill(5)
zcta_area = (
    zcta_area[["zip", "area_sqmi"]]
    .drop_duplicates(subset="zip", keep="first")  # ← add this
)

out = out.merge(zcta_area, on="zip", how="left")
out["pop_density"] = out["Total Population_acs"] / out["area_sqmi"]


# ── 10. Diagnostics ───────────────────────────────────────────────────────────
print("\n── 10. Diagnostics")

key_cols = {
    "ACS":     "pct_poverty",
    "PLACES":  "Diabetes % (Adults)",             # post-rename name
    "TAX":     "municipality",                    # not renamed, survives
    "SNAP":    "snap_stores",                     # not renamed, survives
    "WIC":     "wic_stores",                      # not renamed, survives
    "PRODUCE": "produce_market",                  # not renamed, survives
    "FARA":    "usda_lila_1_10",                  # not renamed, survives
}

for src, col in key_cols.items():
    if col in out.columns:
        n_data    = out[col].notna().sum()
        n_zero    = (out[col] == 0).sum()
        n_missing = out[col].isna().sum()
        log(f"{src:<8} [{col}]  data={n_data:>4}  zero={n_zero:>4}  missing={n_missing:>4}  / {len(out)}")
    else:
        log(f"{src:<8}  !! column '{col}' not found in output")


# ── OSM type counts ───────────────────────────────────────────────────────────
type_counts = Counter()
for el in elements:
    tags = el.get("tags", {})
    if "shop" in tags:
        key, value = "shop", tags["shop"]
    elif "amenity" in tags:
        key, value = "amenity", tags["amenity"]
    else:
        continue
    label = OSM_TYPE_MAP.get((key, value), f"{key}:{value}")
    type_counts[label] += 1

print("\n" + "=" * 60)
print("OSM FOOD ENVIRONMENT TYPE COUNTS")
print("=" * 60)
for k, v in type_counts.most_common():
    print(f"{k:<25} {v:>6}")
print(f"\nTotal OSM elements: {len(elements):,}")


# ── 11. Save ──────────────────────────────────────────────────────────────────
print("\n── 11. Saving")
pre = len(out)
out = out.drop_duplicates(subset="zip", keep="first")
if len(out) < pre:
    log(f"Dropped {pre - len(out)} duplicate ZIP rows before save")
out.to_csv(OUTPUT, index=False)
log(f"Saved → {OUTPUT}")
log(f"Final shape: {out.shape}")

# Column inventory for FARA columns
fara_cols_present = [c for c in out.columns if c.startswith("usda_")]
log(f"FARA columns in output ({len(fara_cols_present)}): {fara_cols_present}")

df = pd.read_csv("data/nj_zip_features_v2.csv", dtype={"zip": str})
print(f"Total columns: {df.shape[1]}")
print("\nAll columns:")
for i, c in enumerate(df.columns):
    print(f"  {i:3d}  {c}")
