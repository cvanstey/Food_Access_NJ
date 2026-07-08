"""
01_zipcrosswalk.py
==============================
Produces a many-to-many ZIP <-> municipality crosswalk so that every
municipality in NJ appears in the output, even when it is small relative
to the ZIP(s) that overlap it.

Run this before running load data as it.
"""

import argparse
import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.validation import make_valid

warnings.filterwarnings("ignore")


import json

BASE_DIR     = Path(__file__).resolve().parent
DATA_DIR     = BASE_DIR / "data"
CROSSWALK_CACHE = DATA_DIR / "crosswalk_data.json"

MUNICIPALITIES_URL = (
    "https://services2.arcgis.com/XVOqAjTOJ5P6ngMu/arcgis/rest/services/"
    "NJ_Municipalities_3857/FeatureServer/0/query"
    "?where=1%3D1&outFields=*&f=geojson&outSR=4326"
)
ZCTA_URL = (
    "https://www2.census.gov/geo/tiger/TIGER2023/ZCTA520/"
    "tl_2023_us_zcta520.zip"
)
TRACT_URL = (
    "https://www2.census.gov/geo/tiger/TIGER2023/TRACT/"
    "tl_2023_34_tract.zip"
)

NJ_BBOX = (-75.6, 38.8, -73.8, 41.4)
NJ_CRS  = "EPSG:3424"

NJ_COUNTY_FIPS = {
    "001": "Atlantic",    "003": "Bergen",      "005": "Burlington",
    "007": "Camden",      "009": "Cape May",     "011": "Cumberland",
    "013": "Essex",       "015": "Gloucester",   "017": "Hudson",
    "019": "Hunterdon",   "021": "Mercer",       "023": "Middlesex",
    "025": "Monmouth",    "027": "Morris",       "029": "Ocean",
    "031": "Passaic",     "033": "Salem",        "035": "Somerset",
    "037": "Sussex",      "039": "Union",        "041": "Warren",
}


def load_municipalities() -> gpd.GeoDataFrame:
    print("  Loading NJ municipal boundaries...")
    gdf = gpd.read_file(MUNICIPALITIES_URL)
    gdf = gdf[["NAME", "MUN_CODE", "CENSUS2020", "geometry"]].copy()
    gdf["geometry"] = gdf["geometry"].apply(make_valid)
    gdf = gdf.to_crs("EPSG:4326")

    gdf["CENSUS2020"] = gdf["CENSUS2020"].astype(str).str.zfill(10)
    gdf["county_fips"] = gdf["CENSUS2020"].str[2:5]
    gdf["county"]      = gdf["county_fips"].map(NJ_COUNTY_FIPS)

    unresolved = gdf["county"].isna().sum()
    if unresolved:
        print(f"  [WARNING] {unresolved} municipalities with unrecognised county FIPS:")
        print(gdf[gdf["county"].isna()][["NAME", "CENSUS2020", "county_fips"]].to_string())

    print(f"  OK {len(gdf)} municipalities across {gdf['county'].nunique()} counties")
    return gdf


def load_zctas() -> gpd.GeoDataFrame:
    print("  Loading ZCTA boundaries (national file, clipping to NJ)...")
    gdf = gpd.read_file(ZCTA_URL)
    gdf = gdf[["ZCTA5CE20", "geometry"]].copy()
    gdf["geometry"] = gdf["geometry"].apply(make_valid)
    gdf = gdf.to_crs("EPSG:4326")
    gdf = gdf.cx[NJ_BBOX[0]:NJ_BBOX[2], NJ_BBOX[1]:NJ_BBOX[3]].copy()
    print(f"  OK {len(gdf)} ZCTAs in/near NJ")
    return gdf


def load_tracts() -> gpd.GeoDataFrame:
    print("  Loading NJ census tracts...")
    gdf = gpd.read_file(TRACT_URL)
    gdf = gdf[["GEOID", "geometry"]].rename(columns={"GEOID": "tract_geoid"}).copy()
    gdf["geometry"] = gdf["geometry"].apply(make_valid)
    gdf = gdf.to_crs("EPSG:4326")
    print(f"  OK {len(gdf)} census tracts")
    return gdf


def intersect_zip_municipality(
    zcta_m: gpd.GeoDataFrame,
    mun_m: gpd.GeoDataFrame,
    min_overlap: float,
) -> pd.DataFrame:
    joined = gpd.overlay(
        zcta_m[["ZCTA5CE20", "zip_area_m2", "geometry"]],
        mun_m[["NAME", "county", "county_fips", "MUN_CODE",
               "CENSUS2020", "mun_area_m2", "geometry"]],
        how="intersection",
        keep_geom_type=False,
    )
    joined["intersection_area_m2"] = joined.geometry.area

    joined["pct_zip_in_municipality"] = (
        joined["intersection_area_m2"] / joined["zip_area_m2"]
    )
    joined["pct_municipality_in_zip"] = (
        joined["intersection_area_m2"] / joined["mun_area_m2"]
    )

    joined = joined[joined["pct_zip_in_municipality"] > min_overlap].copy()

    idx_dom = (
        joined
        .sort_values("pct_zip_in_municipality", ascending=False)
        .drop_duplicates("ZCTA5CE20")
        .index
    )
    joined["is_dominant_mun"] = joined.index.isin(idx_dom)

    return joined.drop(columns="geometry")


def dominant_tract_per_zip(
    zcta_m: gpd.GeoDataFrame,
    tract_m: gpd.GeoDataFrame,
    min_overlap: float,
) -> pd.DataFrame:
    joined = gpd.overlay(
        zcta_m[["ZCTA5CE20", "zip_area_m2", "geometry"]],
        tract_m[["tract_geoid", "tract_area_m2", "geometry"]],
        how="intersection",
        keep_geom_type=False,
    )
    joined["intersection_area_m2"] = joined.geometry.area
    joined["pct_zip_in_tract"] = (
        joined["intersection_area_m2"] / joined["zip_area_m2"]
    )
    joined["pct_tract_in_zip"] = (
        joined["intersection_area_m2"] / joined["tract_area_m2"]
    )
    joined = joined.copy()

    dominant = (
        joined
        .sort_values("pct_zip_in_tract", ascending=False)
        .drop_duplicates("ZCTA5CE20")
        [["ZCTA5CE20", "tract_geoid", "tract_area_m2",
          "pct_zip_in_tract", "pct_tract_in_zip"]]
    )
    return dominant


def build_spatial_crosswalk(
    mun: gpd.GeoDataFrame,
    zcta: gpd.GeoDataFrame,
    tracts: gpd.GeoDataFrame,
    min_overlap: float = 0.001,
) -> pd.DataFrame:

    print("Projecting layers to NJ State Plane...")
    mun_m   = mun.to_crs(NJ_CRS).copy()
    zcta_m  = zcta.to_crs(NJ_CRS).copy()
    tract_m = tracts.to_crs(NJ_CRS).copy()

    zcta_m["zip_area_m2"]    = zcta_m.geometry.area
    mun_m["mun_area_m2"]     = mun_m.geometry.area
    tract_m["tract_area_m2"] = tract_m.geometry.area

    print("Intersecting ZIP x Municipality (many-to-many)...")
    zip_mun = intersect_zip_municipality(zcta_m, mun_m, min_overlap)

    print("Finding dominant tract per ZIP...")
    zip_tract = dominant_tract_per_zip(zcta_m, tract_m, min_overlap)

    crosswalk = zip_mun.merge(zip_tract, on="ZCTA5CE20", how="left")

    n_zips = crosswalk["ZCTA5CE20"].nunique()
    n_muns = crosswalk["MUN_CODE"].nunique()
    print(f"  OK {len(crosswalk):,} ZIP-municipality pairs "
          f"({n_zips} unique ZIPs, {n_muns} unique municipalities)")

    all_mun_names  = set(mun["MUN_CODE"])
    seen_mun_names = set(crosswalk["MUN_CODE"])
    missing = all_mun_names - seen_mun_names
    if missing:
        print(f"\n  [INFO] {len(missing)} municipalities have no ZCTA overlap "
              f"(they may lie entirely within water or be very small):")
        miss_df = (
            mun[mun["MUN_CODE"].isin(missing)][["NAME", "county", "MUN_CODE"]]
            .sort_values(["county", "NAME"])
        )
        print(miss_df.to_string(index=False))

    return crosswalk


def build_output(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(columns={
        "ZCTA5CE20":  "zip_code",
        "NAME":       "municipality",
        "MUN_CODE":   "nj_mun_code",
        "CENSUS2020": "mcd_geoid",
        "tract_geoid":"census_tract_geoid",
    }).copy()

    pct_cols = [
        "pct_zip_in_municipality", "pct_municipality_in_zip",
        "pct_zip_in_tract",        "pct_tract_in_zip",
    ]
    for col in pct_cols:
        if col in out.columns:
            out[col] = out[col].round(6)

    col_order = [
        "zip_code", "county", "county_fips", "municipality",
        "nj_mun_code", "mcd_geoid", "census_tract_geoid",
        "is_dominant_mun", "is_dominant_tract",
        "intersection_area_m2", "zip_area_m2", "mun_area_m2", "tract_area_m2",
        "pct_zip_in_municipality", "pct_municipality_in_zip",
        "pct_zip_in_tract",        "pct_tract_in_zip",
    ]
    out = out[[c for c in col_order if c in out.columns]]
    return (
        out
        .sort_values(["county", "municipality", "zip_code"])
        .reset_index(drop=True)
    )


def main():
    parser = argparse.ArgumentParser(
        description="Build a ZIP/ZCTA -> municipality + census tract "
                    "spatial crosswalk for NJ (many-to-many)."
    )
    parser.add_argument(
        "--out", default=str(DATA_DIR / "nj_zip_crosswalk.csv"),
        help="Output CSV path  [default: data/nj_zip_crosswalk.csv]",
    )
    parser.add_argument(
        "--min-overlap", type=float, default=0.001,
        help="Minimum fraction of a ZIP's area that must fall inside a "
             "feature for it to be included.  [default: 0.001]",
    )
    args = parser.parse_args()

    print("\n=== 1/3  Geometry layers ===")
    mun = load_municipalities()
    zcta = load_zctas()

    # ── Hard NJ-prefix filter ────────────────────────────────────────────
    # NJ is the only state ever assigned ZIP prefixes 07x/08x. Filtering
    # here, BEFORE the nj_union overlay below, matters: the overlay clips
    # each ZCTA polygon down to only its NJ-overlapping fragment, and
    # zip_area_m2 gets computed on that already-clipped fragment later in
    # build_spatial_crosswalk(). That means pct_zip_in_municipality is
    # measured against a denominator that was already restricted to NJ —
    # so ANY nonzero border sliver (a real edge overlap, or just TIGER
    # shapefile snapping imprecision) will look like ~100% overlap and
    # trivially clear min_overlap, regardless of how small the true
    # overlap was. This is almost certainly how out-of-state ZIPs like
    # 10977 (Spring Valley, NY) and 19007 (Bristol, PA) end up inside
    # nj_zip_crosswalk.csv despite being real, non-NJ ZCTAs.
    # Filtering by prefix identity here removes them before that area
    # math ever runs, instead of trying to out-threshold a ratio that's
    # structurally biased toward showing 100%.
    before_prefix = len(zcta)
    zcta = zcta[zcta["ZCTA5CE20"].str.startswith(("07", "08"))].copy()
    dropped_prefix = before_prefix - len(zcta)
    print(f"  NJ-prefix filter: {before_prefix} -> {len(zcta)} ZCTAs "
          f"({dropped_prefix} dropped as non-NJ-prefix border/adjacent ZCTAs)")

    nj_union = mun.unary_union
    zcta = gpd.overlay(
        zcta,
        gpd.GeoDataFrame(geometry=[nj_union], crs=zcta.crs),
        how="intersection"
    )
    print(f"  OK {len(zcta)} ZCTAs in NJ")

    tracts = load_tracts()

    keyport = mun[mun["MUN_CODE"] == "1324"]

    print("\nKeyport geometry:")
    print(keyport[["NAME", "county", "MUN_CODE"]])

    hits = zcta[zcta.intersects(keyport.geometry.iloc[0])]

    print("\nZCTAs intersecting Keyport:")
    print(len(hits))

    if len(hits):
        print(hits["ZCTA5CE20"].tolist())

    print("\n=== 2/3  Spatial crosswalk ===")
    crosswalk = build_spatial_crosswalk(mun, zcta, tracts,
                                        min_overlap=args.min_overlap)

    print("\n=== 3/3  Save ===")
    output = build_output(crosswalk)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_path, index=False)

    dom = output[output["is_dominant_mun"]]
    print(f"\n  OK {len(output):,} rows (ZIP x municipality pairs) -> {out_path}")
    print(f"    Unique ZIPs                  : {output['zip_code'].nunique()}")
    print(f"    Unique municipalities (total): {output['nj_mun_code'].nunique()}")
    print(f"    Unique counties              : {output['county'].nunique()}")
    print(f"    ZIPs with dominant tract     : "
        f"{output.loc[output['census_tract_geoid'].notna(), 'zip_code'].nunique()}"
    )
    print(f"\n  Dominant-municipality rows (1 per ZIP): {len(dom)}")
    print()

    missing = set(mun["MUN_CODE"]) - set(crosswalk["MUN_CODE"])

    print("\nZIPs missing tract assignment:")

    print(
        output.loc[
            output["census_tract_geoid"].isna(),
            ["zip_code", "municipality", "county"]
        ].drop_duplicates()
    )

    print(len(missing))

    print(
        mun.loc[
            mun["MUN_CODE"].isin(missing),
            ["NAME", "county", "MUN_CODE"]
        ]
    )
    preview_cols = [
        "zip_code", "county", "municipality", "is_dominant_mun",
        "pct_zip_in_municipality", "pct_municipality_in_zip",
    ]
    print(dom[preview_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()