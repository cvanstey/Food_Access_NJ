from pathlib import Path
import geopandas as gpd
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

ZCTA_PATH = r"C:\Users\crook\PROJECTS\GTFS\NJ_GTFS_Transit\data\arcgis_inputs\nj_zcta.shp"
FDC_PATH  = r"zip://C:\Users\crook\Downloads\Proposed_New_Jersey_Food_Deserts.zip"

# ── 1. Load both layers ─────────────────────────────────────────────
fdc  = gpd.read_file(FDC_PATH)
zcta = gpd.read_file(ZCTA_PATH)

# ── 2. Reproject ZCTA to match FDC's projected CRS (NJ State Plane, ft) ──
zcta = zcta.to_crs(fdc.crs)

zcta = zcta[["ZCTA5CE20", "geometry"]].rename(columns={"ZCTA5CE20": "zip"})
fdc_slim = fdc[["DESERTNAME", "COUNTY", "POPULATION", "FD_RANK", "FD_SCR", "geometry"]]

# ── 3. Overlay: intersect ZCTAs with FDC polygons ──────────────────────
overlay = gpd.overlay(zcta, fdc_slim, how="intersection")
overlay["overlap_area"] = overlay.geometry.area

zcta_total_area = zcta.set_index("zip").geometry.area
overlay["zcta_area"] = overlay["zip"].map(zcta_total_area)
overlay["overlap_pct"] = overlay["overlap_area"] / overlay["zcta_area"]

print(f"\nTotal ZCTA-FDC intersection rows: {len(overlay)}")
print(f"Unique ZIPs touching any FDC: {overlay['zip'].nunique()}")

# ── 4. Keep dominant FDC per ZIP (largest area overlap) ────────────────
best = (
    overlay.sort_values("overlap_pct", ascending=False)
    .drop_duplicates(subset="zip", keep="first")
    [["zip", "DESERTNAME", "COUNTY", "FD_RANK", "FD_SCR", "overlap_pct"]]
    .rename(columns={
        "DESERTNAME": "njeda_fdc_name",
        "COUNTY": "njeda_fdc_county",
        "FD_RANK": "njeda_fdc_rank",
        "FD_SCR": "njeda_fdc_score",
        "overlap_pct": "njeda_overlap_pct",
    })
)

# Flag using >10% area overlap threshold (adjustable)
best["is_desert_njeda"] = (best["njeda_overlap_pct"] > 0.10).astype(int)

print(f"\nZIPs with ANY overlap: {len(best)}")
print(f"ZIPs with >10% area overlap (flagged): {best['is_desert_njeda'].sum()}")
print(f"\nTop 20 by overlap:")
print(best.sort_values("njeda_overlap_pct", ascending=False).head(20).to_string(index=False))

# ── 5. Save crosswalk ───────────────────────────────────────────────
out_path = DATA_DIR / "njeda_fdc_zip_crosswalk.csv"
best.to_csv(out_path, index=False)
print(f"\nSaved → {out_path}")