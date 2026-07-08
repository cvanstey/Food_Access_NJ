"""
00b_enrich_crosswalk.py
========================
Stage: enrich the ZIP <-> municipality crosswalk down to one row per ZIP.

Input:
    data/nj_zip_crosswalk.csv       (produced by 00a_build_crosswalk.py)
    data/2024_Gaz_zcta_national.txt (Census Gazetteer ZCTA file)
    data/ZIP_TRACT_122025.xlsx      (HUD ZIP -> tract crosswalk)
    data/TRACT_ZIP_122025.xlsx      (HUD tract -> ZIP crosswalk)

Output:
    data/nj_zip_complete.csv        (one row per ZIP)

Known gaps (see conversation notes — do not silently "fix" these by guessing):
    - No `lat` / `lon` columns are produced here. The historical version of
      nj_zip_complete.csv had both `lat`/`lon` AND `gaz_lat`/`gaz_lon`, but
      only the Gazetteer source (gaz_lat/gaz_lon) is reproducible from known
      inputs. If a second coordinate source is confirmed, add it as its own
      merge step rather than duplicating gaz_lat/gaz_lon under a new name.
    - No `_base`-suffixed columns (zip_area_m2_base, pct_zip_in_municipality_base,
      etc.) are produced here. Nothing in the known pipeline explains a second
      area/pct computation that would justify a `_base` suffix collision.
      Do not fabricate these — if they're needed, find or rebuild the step
      that produced them first.
"""

import argparse
from pathlib import Path

import pandas as pd

from pipeline_utils import section, normalize_zip

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

CROSSWALK_PATH   = DATA_DIR / "nj_zip_crosswalk.csv"
GAZETTEER_PATH   = DATA_DIR / "2024_Gaz_zcta_national.txt"
HUD_ZIP_TO_TRACT = DATA_DIR / "ZIP_TRACT_122025.xlsx"
HUD_TRACT_TO_ZIP = DATA_DIR / "TRACT_ZIP_122025.xlsx"
OUTPUT_PATH      = DATA_DIR / "nj_zip_complete.csv"

# NJ county FIPS codes are the odd numbers 001-041 (21 counties)
NJ_COUNTY_FIPS = {str(i).zfill(3) for i in range(1, 42, 2)}

def load_crosswalk(path: Path) -> pd.DataFrame:
    """Load the many-to-many ZIP x municipality crosswalk.

    Renames zip_code -> zip and normalizes the ZIP and census tract
    GEOID columns to consistent, zero-padded strings.
    """
    df = pd.read_csv(
        path,
        dtype={
            "zip_code": str,
            "county_fips": str,
            "nj_mun_code": str,
            "mcd_geoid": str,
            "census_tract_geoid": str,
        },
    )
    df = df.rename(columns={"zip_code": "zip"})
    df["zip"] = normalize_zip(df["zip"])
    df["census_tract_geoid"] = clean_tract(df["census_tract_geoid"])
    return df


def clean_tract(series: pd.Series) -> pd.Series:
    """Normalize a census tract GEOID column to zero-padded 11-digit strings.

    Handles values that arrived as floats (e.g. "34001000100.0") by
    truncating at the decimal point before padding.
    """
    return series.astype(str).str.split(".").str[0].str.zfill(11)


def load_gazetteer(path: Path) -> pd.DataFrame:
    """Load the Census Gazetteer ZCTA file and keep only the columns
    this pipeline needs: land/water area and the ZCTA internal-point
    centroid coordinates.
    """
    gaz = pd.read_csv(path, sep="\t", dtype={"GEOID": str}, skipinitialspace=True)
    gaz.columns = gaz.columns.str.strip()
    gaz = gaz.rename(columns={"GEOID": "zip"})
    gaz["zip"] = normalize_zip(gaz["zip"])
    return gaz[["zip", "ALAND", "AWATER", "ALAND_SQMI", "AWATER_SQMI", "INTPTLAT", "INTPTLONG"]]


def load_hud_zip_to_tract(path: Path, valid_zips: set) -> pd.DataFrame:
    """Load the HUD ZIP->tract crosswalk, restricted to NJ tracts and
    to ZIPs already known from the base crosswalk.
    """
    df = pd.read_excel(path, dtype={"ZIP": str, "TRACT": str})
    df.columns = df.columns.str.strip().str.lower()
    df["tract"] = df["tract"].str.zfill(11)
    df["zip"] = normalize_zip(df["zip"])
    df = df[df["tract"].str.startswith("34")]
    df = df[df["zip"].isin(valid_zips)]
    return df


def load_hud_tract_to_zip(path: Path, valid_zips: set) -> pd.DataFrame:
    """Load the HUD tract->ZIP crosswalk, restricted to NJ tracts and
    to ZIPs already known from the base crosswalk.
    """
    df = pd.read_excel(path, dtype={"TRACT": str, "ZIP": str})
    df.columns = df.columns.str.strip().str.lower()
    df["tract"] = df["tract"].str.zfill(11)
    df["zip"] = normalize_zip(df["zip"])
    df = df[df["tract"].str.startswith("34")]
    df = df[df["zip"].isin(valid_zips)]
    return df


def summarize_hud_zip_to_tract(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse ZIP->tract rows to one summary row per ZIP: how many
    tracts it touches, which tract dominates by residential ratio,
    and that ratio's value.
    """
    return (
        df.sort_values("res_ratio", ascending=False)
        .groupby("zip")
        .agg(
            hud_tract_count=("tract", "count"),
            hud_dominant_tract=("tract", "first"),
            hud_max_res_ratio=("res_ratio", "max"),
        )
        .reset_index()
    )


def summarize_hud_tract_to_zip(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse tract->ZIP rows to one summary row per tract: how many
    ZIPs touch it, which ZIP dominates by residential ratio, and that
    ratio's value. Renamed to census_tract_geoid for the join back to
    the ZIP-level table.
    """
    return (
        df.sort_values("res_ratio", ascending=False)
        .groupby("tract")
        .agg(
            hud_zip_count=("zip", "count"),
            hud_dominant_zip=("zip", "first"),
            hud_max_res_ratio_t2z=("res_ratio", "max"),
        )
        .reset_index()
        .rename(columns={"tract": "census_tract_geoid"})
    )


def dedupe_to_dominant_municipality(crosswalk: pd.DataFrame) -> pd.DataFrame:
    """Collapse the many-to-many ZIP x municipality crosswalk down to
    one row per ZIP: the row where is_dominant_mun is True.

    NOTE: earlier drafts of this step used
        df.sort_values("zip").drop_duplicates(subset="zip", keep="first")
    which is a bug — sort_values("zip") only orders *between* ZIPs; for
    rows sharing a ZIP it preserves whatever order they arrived in
    (alphabetical by municipality name from 00a's output), not the
    dominant municipality. Sorting on is_dominant_mun explicitly fixes
    this.
    """
    deduped = (
        crosswalk
        .sort_values(["zip", "is_dominant_mun"], ascending=[True, False])
        .drop_duplicates(subset="zip", keep="first")
        .copy()
    )
    # These two columns only make sense in the many-to-many table;
    # drop them now that we've used is_dominant_mun to pick a winner.
    return deduped.drop(columns=["is_dominant_mun", "intersection_area_m2"])


def enrich_crosswalk(
    crosswalk_path: Path,
    gazetteer_path: Path,
    hud_zip_to_tract_path: Path,
    hud_tract_to_zip_path: Path,
) -> pd.DataFrame:
    """Build the one-row-per-ZIP enriched crosswalk.

    Starts from the deduplicated (dominant-municipality) crosswalk and
    merges in Gazetteer land/water area + centroid, plus HUD ZIP<->tract
    coverage summaries. Filters to NJ counties only at the end.
    """
    section("Loading inputs")
    crosswalk = load_crosswalk(crosswalk_path)
    print(f"  Crosswalk rows (many-to-many): {len(crosswalk):,}")

    gazetteer = load_gazetteer(gazetteer_path)
    print(f"  Gazetteer ZCTAs: {len(gazetteer):,}")

    valid_zips = set(crosswalk["zip"])
    hud_zip_to_tract = load_hud_zip_to_tract(hud_zip_to_tract_path, valid_zips)
    hud_tract_to_zip = load_hud_tract_to_zip(hud_tract_to_zip_path, valid_zips)
    print(f"  HUD zip->tract rows (NJ, known ZIPs): {len(hud_zip_to_tract):,}")
    print(f"  HUD tract->zip rows (NJ, known ZIPs): {len(hud_tract_to_zip):,}")

    section("Deduplicating to one row per ZIP")
    base = dedupe_to_dominant_municipality(crosswalk)
    print(f"  Unique ZIPs after dedup: {len(base):,}")

    section("Merging enrichment sources")
    merged = base.merge(gazetteer, on="zip", how="left")
    merged = merged.rename(columns={"INTPTLAT": "gaz_lat", "INTPTLONG": "gaz_lon"})
    print(f"  Gazetteer match rate: {merged['ALAND'].notna().mean():.1%}")

    hud_zip_summary = summarize_hud_zip_to_tract(hud_zip_to_tract)
    merged = merged.merge(hud_zip_summary, on="zip", how="left")

    hud_tract_summary = summarize_hud_tract_to_zip(hud_tract_to_zip)
    merged = merged.merge(hud_tract_summary, on="census_tract_geoid", how="left")
    print(f"  HUD tract-summary match rate: {merged['hud_tract_count'].notna().mean():.1%}")

    section("Filtering to NJ counties")
    merged["county_fips"] = merged["county_fips"].str.strip().str.zfill(3)
    pre = len(merged)
    merged = merged[merged["county_fips"].isin(NJ_COUNTY_FIPS)].copy()
    print(f"  NJ county filter: {pre} -> {len(merged)} rows ({pre - len(merged)} dropped)")

    merged["county_fips_full"] = "34" + merged["county_fips"]

    return merged.sort_values("zip").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(
        description="Enrich the NJ ZIP crosswalk to one row per ZIP "
                     "with Gazetteer and HUD data."
    )
    parser.add_argument(
        "--out", default=str(OUTPUT_PATH),
        help=f"Output CSV path  [default: {OUTPUT_PATH}]",
    )
    args = parser.parse_args()

    if not CROSSWALK_PATH.exists():
        raise FileNotFoundError(
            f"\n  Crosswalk not found at: {CROSSWALK_PATH}\n"
            f"  Run 00a_build_crosswalk.py first to generate it."
        )

    merged = enrich_crosswalk(
        CROSSWALK_PATH, GAZETTEER_PATH, HUD_ZIP_TO_TRACT, HUD_TRACT_TO_ZIP
    )

    section("Sanity checks")
    print(f"  Rows: {len(merged)}")
    print(f"  Unique ZIPs: {merged['zip'].nunique()}")
    print(f"  ZIP nulls: {merged['zip'].isna().sum()}")
    print(f"  Columns: {merged.columns.tolist()}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"\n  OK {len(merged):,} rows -> {out_path}")


if __name__ == "__main__":
    main()