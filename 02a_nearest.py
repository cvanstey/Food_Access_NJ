"""
02a_nearest.py

This creates distance calculations for supermarkets, convenience stores, etc.
Run this before merge_sources.py
==================
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent
DATA_DIR  = BASE_DIR / "data"
OSM_FILE  = DATA_DIR / "osm_data.json"
ZIPS_FILE = DATA_DIR / "nj_zip_crosswalk.csv"
OUTPUT    = DATA_DIR / "nj_zip_nearest.csv"

NJ_LAT = (38.8, 41.4)
NJ_LON = (-75.6, -73.8)
MILES_PER_DEGREE = 69.0


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Parse OSM, classify each node
# ─────────────────────────────────────────────────────────────────────────────
def classify_node(tags: dict) -> str | None:
    amenity = tags.get("amenity", "")
    shop    = tags.get("shop", "")
    if amenity == "fast_food":
        return "fast_food"
    if shop == "convenience":
        return "convenience"
    if shop == "supermarket":
        return "supermarket"
    return None


with open(OSM_FILE) as f:
    osm = json.load(f)

records    = []
skipped_pa = 0

for el in osm["elements"]:
    lat = el.get("lat")
    lon = el.get("lon")

    if lat is None and "center" in el:
        lat = el["center"]["lat"]
        lon = el["center"]["lon"]

    if lat is None or lon is None:
        continue

    if not (NJ_LAT[0] <= lat <= NJ_LAT[1] and NJ_LON[0] <= lon <= NJ_LON[1]):
        skipped_pa += 1
        continue

    tags     = el.get("tags", {})
    category = classify_node(tags)
    if category is None:
        continue

    records.append({
        "category": category,
        "lat":      lat,
        "lon":      lon,
        "name":     tags.get("name", ""),
        "brand":    tags.get("brand", ""),
    })

osm_df = pd.DataFrame(records)
print(f"Skipped {skipped_pa} out-of-state nodes")
print(f"Kept:   {osm_df['category'].value_counts().to_dict()}")

print("\n── Top convenience brands")
print(osm_df[osm_df["category"] == "convenience"]["brand"].value_counts().head(15))

print("\n── Top fast food brands")
print(osm_df[osm_df["category"] == "fast_food"]["brand"].value_counts().head(15))


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Build KD-trees
# ─────────────────────────────────────────────────────────────────────────────
def build_tree(df: pd.DataFrame, category: str):
    pts = df[df["category"] == category][["lat", "lon"]].values
    if len(pts) == 0:
        print(f"  [WARNING] No points found for category: {category}")
        return None, None
    print(f"  KD-tree: {len(pts)} points for '{category}'")
    return cKDTree(pts), pts


ff_tree,   _ = build_tree(osm_df, "fast_food")
conv_tree, _ = build_tree(osm_df, "convenience")
super_tree, _ = build_tree(osm_df, "supermarket")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Load ZIP file and join Census ZCTA centroids
# ─────────────────────────────────────────────────────────────────────────────
CENTROIDS_FILE = DATA_DIR / "2024_gaz_zcta_national.txt"  # tab-delimited

zips = pd.read_csv(ZIPS_FILE, dtype={"zip_code": str})
zips = zips.rename(columns={"zip_code": "zip"})

centroids = pd.read_csv(
    CENTROIDS_FILE,
    sep="\t",
    dtype={"GEOID": str},
).rename(columns=lambda c: c.strip())  # <-- this fixes the trailing whitespace

centroids = (
    centroids
    [["GEOID", "INTPTLAT", "INTPTLONG"]]
    .rename(columns={"GEOID": "zip", "INTPTLAT": "lat", "INTPTLONG": "lon"})
)
centroids["lat"] = centroids["lat"].astype(float)
centroids["lon"] = centroids["lon"].astype(float)

zips = zips.merge(centroids, on="zip", how="left")

missing = zips[["lat", "lon"]].isna().any(axis=1).sum()
if missing:
    print(f"  [WARNING] {missing} ZIPs missing centroids after join — they'll be dropped")
    zips = zips.dropna(subset=["lat", "lon"])

print(f"\nParsed {len(zips)} ZIP centroids")
print(zips[["zip", "lat", "lon"]].head(5))

assert zips["lat"].between(38.8, 41.4).all(), "Unexpected lat values — check join"
assert zips["lon"].between(-75.6, -73.8).all(), "Unexpected lon values — check join"
print("  Coordinate sanity check passed")

# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Query nearest distances for each ZIP centroid
# ─────────────────────────────────────────────────────────────────────────────
zip_coords = zips[["lat", "lon"]].values


def nearest_miles(tree, coords: np.ndarray) -> np.ndarray:
    if tree is None:
        return np.full(len(coords), np.nan)
    dist_deg, _ = tree.query(coords, k=1)
    return (dist_deg * MILES_PER_DEGREE).round(4)


zips["nearest_fastfood_miles"]    = nearest_miles(ff_tree,   zip_coords)
zips["nearest_convenience_miles"] = nearest_miles(conv_tree, zip_coords)
zips["nearest_supermarket_miles"] = nearest_miles(super_tree, zip_coords)


print("\n── Distance summary (miles)")
print(zips[["nearest_fastfood_miles", "nearest_convenience_miles",
            "nearest_supermarket_miles"]].describe().round(3))

print("\n── Sample output")
print(zips[["zip", "nearest_fastfood_miles",
            "nearest_convenience_miles", "nearest_supermarket_miles"]].head(10))


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Save
# ─────────────────────────────────────────────────────────────────────────────
zips.to_csv(OUTPUT, index=False)
print(f"\nSaved → {OUTPUT}  shape={zips.shape}")

df = pd.read_csv("data/nj_zip_nearest.csv", dtype={"zip": str})
print(f"Total columns: {df.shape[1]}")
print("\nAll columns:")
for i, c in enumerate(df.columns):
    print(f"  {i:3d}  {c}")
