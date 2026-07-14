"""
01_load_data.py
==================
Data Acquisition
Food Environment Pipeline — New Jersey

This script:
  1. Downloads all raw datasets directly from their public sources
  2. Decompresses any zipped files (handled transparently by geopandas for shapefiles)
  3. Reads each dataset into memory and prints a confirmation summary

Authentication / API Keys
--------------------------
Census API (ACS data):
  - A free API key removes rate limits and is strongly recommended.
  - Sign up at: https://api.census.gov/data/key_signup.html
  - Paste your key into CENSUS_API_KEY below.
  - The anonymous endpoint (no key) is used as a fallback if left blank,
    but may produce an error with repeated requests.

OpenStreetMap Overpass API:
  - No authentication required. Free public endpoint.
  - URL: https://overpass-api.de/api/interpreter

Census TIGER Shapefiles:
  - No authentication required. Free public endpoint.
  - URL: https://www2.census.gov/geo/tiger

CDC PLACES:
  - No authentication required. Accessed via Socrata Open Data API (SODA).
  - URL: https://data.cdc.gov/api/views/kee5-23sr/rows.csv?accessType=DOWNLOAD

Dependencies
------------
  pip install requests geopandas pandas pdfplumber numpy openpyxl
"""

import io
import re
import time
import warnings
from pathlib import Path
import gdown
import geopandas as gpd
import numpy as np
import pandas as pd
import pdfplumber
import requests
import json
import os

from pipeline_utils import section, normalize_zip
from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
DATA_DIR = ROOT_DIR / "data"

DATA_DIR.mkdir(exist_ok=True)


CENSUS_API_KEY = os.getenv("CENSUS_API_KEY")

if CENSUS_API_KEY:
    print("Using Census API key.")
else:
    print("No Census API key found. Using the public API (may be slower).")

STATE_NAME = "New Jersey"
STATE_FIPS = "34"
STATE_ABBR = "NJ"

ACS_YEAR   = 2022
TIGER_YEAR = 2021

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
TIGER_BASE   = "https://www2.census.gov/geo/tiger"
PLACES_URL   = "https://data.cdc.gov/api/views/kee5-23sr/rows.csv?accessType=DOWNLOAD"
WIC_PDF_PATH  = "https://www.nj.gov/health/fhs/wic/data/vendorlocations.pdf"


# Local data files
CROSSWALK_PATH = DATA_DIR / "nj_zip_complete.csv"
OSM_CACHE      = DATA_DIR / "osm_data.json"
SNAP_CSV = DATA_DIR / "snap_retailer_location_data.csv"
NJEDA_PDF_PATH = DATA_DIR / "food-security-product-deck.-march-2024.pdf"

with pdfplumber.open(NJEDA_PDF_PATH) as pdf:
    print(len(pdf.pages))


NJ_BBOX = "38.9,-75.6,41.4,-73.9"

OSM_TAGS = [
    ("shop", "supermarket"),
    ("shop", "grocery"),
    ("shop", "convenience"),
    ("amenity", "fast_food"),
    ("amenity", "restaurant"),
    ("shop", "variety_store"),
    ("shop", "greengrocer"),
    ("shop", "butcher"),
    ("shop", "seafood"),
    ("shop", "bakery"),
    ("shop", "deli"),
    ("shop", "health_food"),
    ("shop", "organic"),
    ("shop", "farm"),
    ("amenity", "marketplace"),
    ("shop", "chemist"),
    ("shop", "drugstore"),
    ("shop", "wholesale"),
    ("shop", "alcohol"),
    ("amenity", "bar"),
]

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
    ("shop", "alcohol"):        "Liquor Store",
    ("amenity", "bar"):         "Bar",
}

ACS_VARS = [
    "B19013_001E", "B01003_001E", "B01002_001E", "B17001_002E", "B17001_001E",
    "B22003_002E", "B22003_001E", "B08201_002E", "B08201_001E", "B08301_001E",
    "B08301_010E", "B25003_001E", "B25003_002E", "B15003_017E", "B15003_022E",
    "B15003_023E", "B15003_024E", "B15003_025E", "B15003_001E", "B02001_002E",
    "B02001_003E", "B02001_005E", "B03002_003E", "B03002_004E",
    "B03002_006E", "B03002_012E", "B03002_001E", "B25058_001E", "B25077_001E",
    "B11012_010E", "B11012_001E", "B19057_002E", "B19057_001E", "B28002_013E",
    "B28002_001E", "B25070_010E", "B25070_011E", "B25070_001E", "B25002_003E",
    "B25002_001E", "B23025_005E", "B23025_001E", "B01001_020E", "B01001_021E",
    "B01001_022E", "B01001_023E", "B01001_024E", "B01001_025E", "B01001_044E",
    "B01001_045E", "B01001_046E", "B01001_047E", "B01001_048E", "B01001_049E",
]

NJ_COUNTIES = [
    "Atlantic", "Bergen", "Burlington", "Camden", "Cape May",
    "Cumberland", "Essex", "Gloucester", "Hudson", "Hunterdon",
    "Mercer", "Middlesex", "Monmouth", "Morris", "Ocean",
    "Passaic", "Salem", "Somerset", "Sussex", "Union", "Warren",
]

OSM_COL_MAP = {
    "Supermarket":       "supermarket",
    "Grocery Store":     "grocery",
    "Convenience Store": "convenience",
    "Fast Food":         "fast_food",
    "Restaurant":        "restaurant",
    "Variety Store":     "dollar_store",
    "Produce Market":    "produce_market",
    "Meat / Seafood":    "meat_seafood",
    "Bakery":            "bakery",
    "Deli":              "deli",
    "Health Food Store": "health_food",
    "Farm Stand":        "farm_stand",
    "Farmers Market":    "farmers_market",
    "Pharmacy":          "pharmacy",
    "Wholesale Club":    "wholesale",
    "Liquor Store":      "liquor_store",
    "Bar":               "bar",
}

# ── Helpers ───────────────────────────────────────────────────────────────────
# section() and normalize_zip() now live in pipeline_utils.py (imported above)

FOLDER_URL = "https://drive.google.com/drive/folders/1RMIZwLmmeUC1CMEF0X8oBuy4m7azTiOo"

# Download only if the files aren't already present
if not (DATA_DIR / "nj_zip_features_v5.csv").exists():
    print("Downloading shared project data from Google Drive...")
    gdown.download_folder(
        url=FOLDER_URL,
        output=str(DATA_DIR),
        quiet=False,
    )
    print("Download complete.")


def http_get(url: str, *, timeout: int = 300, post_data: dict = None,
             stream: bool = False, params: dict = None) -> requests.Response:
    """GET or POST with basic retry on timeout / server errors."""
    for attempt in range(1, 4):
        try:
            if post_data:
                resp = requests.post(url, data=post_data, timeout=timeout, stream=stream)
            else:
                resp = requests.get(url, params=params, timeout=timeout, stream=stream)
            resp.raise_for_status()
            return resp
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            print(f"  [attempt {attempt}/3] {e}")
            if attempt < 3:
                wait = 30 * attempt
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else "?"
            print(f"  [attempt {attempt}/3] HTTP {status}: {e}")
            if attempt < 3 and status in (429, 500, 503):
                wait = 60 * attempt
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


def fetch_pdf_bytes(url: str) -> bytes:
    """Download a PDF from a URL and return its raw bytes."""
    print(f"  Downloading PDF: {url}")
    resp = http_get(url, timeout=120)
    return resp.content


# ── 1. OpenStreetMap Food Locations ───────────────────────────────────────────

section("1 of 9 — OpenStreetMap Food Locations")

if OSM_CACHE.exists():
    print("  Loading cached OSM data...")
    with open(OSM_CACHE, "r", encoding="utf-8") as f:
        osm_data = json.load(f)
else:
    print(f"  Querying Overpass API for {STATE_NAME} food locations...")
    print("  (May take 30–120 seconds)")

    OVERPASS_ENDPOINTS = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    ]

    def build_query(k, v):
        return f"""
[out:json][timeout:180];
(
  node["{k}"="{v}"]({NJ_BBOX});
  way["{k}"="{v}"]({NJ_BBOX});
);
out center;
"""

    all_elements = []

    for k, v in OSM_TAGS:
        print(f"  → Fetching {k}={v} ...")
        query   = build_query(k, v)
        success = False

        for endpoint in OVERPASS_ENDPOINTS:
            try:
                resp = requests.post(
                    endpoint,
                    data={"data": query},
                    headers={
                        "User-Agent": "NJFoodDesertResearch/1.0 (academic project)",
                        "Accept":     "application/json",
                    },
                    timeout=180,
                )
                resp.raise_for_status()
                data     = resp.json()
                elements = data.get("elements", [])
                all_elements.extend(elements)
                print(f"     ✔ {endpoint.split('/')[2]} ({len(elements)} elements)")
                success = True
                break
            except Exception as e:
                print(f"     ✖ {endpoint.split('/')[2]} failed: {e}")

        if not success:
            print(f"     ⚠ Skipping {k}={v} (all endpoints failed)")

    osm_data = {"elements": all_elements}

    with open(OSM_CACHE, "w", encoding="utf-8") as f:
        json.dump(osm_data, f)
    print(f"  ✔ Cached OSM data saved ({len(all_elements):,} elements)")

print(f"  OSM elements loaded: {len(osm_data.get('elements', [])):,}")

# ── 2. Census TIGER — County Boundaries ──────────────────────────────────────

section("2 of 9 — Census TIGER County Boundaries")

county_url  = f"{TIGER_BASE}/TIGER{TIGER_YEAR}/COUNTY/tl_{TIGER_YEAR}_us_county.zip"
print(f"  Downloading US county shapefile...")
print(f"  URL: {county_url}")

counties_all = gpd.read_file(county_url)
counties_nj  = counties_all[counties_all["STATEFP"] == STATE_FIPS].copy()
counties_nj  = counties_nj.to_crs(epsg=4326)

print(f"  ✔ County boundaries loaded: {len(counties_nj)} NJ counties")
print(counties_nj[["NAME", "COUNTYFP"]].to_string(index=False))

# ── 3. Census TIGER — ZCTA Boundaries ────────────────────────────────────────

section("3 of 9 — Census TIGER ZCTA Boundaries")

zcta_nj = gpd.read_file(DATA_DIR / "zcta_nj.gpkg")
zcta_nj = zcta_nj.to_crs(epsg=4326)

zcta_nj = gpd.sjoin(
    zcta_nj,
    counties_nj[["geometry"]],
    how="inner",
    predicate="intersects",
).drop_duplicates(subset="zip").reset_index(drop=True)

# Filter to known NJ ZIPs from crosswalk — removes border ZIPs that merely
# touch NJ county boundaries (e.g. 19153 Philadelphia, 10977 Spring Valley NY)
nj_crosswalk = pd.read_csv(DATA_DIR / "nj_zip_crosswalk.csv", dtype={"zip_code": str})
nj_crosswalk["zip_code"] = normalize_zip(nj_crosswalk["zip_code"])
valid_nj_zips = set(nj_crosswalk["zip_code"])

pre     = len(zcta_nj)
zcta_nj = zcta_nj[normalize_zip(zcta_nj["zip"]).isin(valid_nj_zips)]
print(f"  Border ZIP filter: {pre} → {len(zcta_nj)} ZCTAs ({pre - len(zcta_nj)} dropped)")
print(f"  ✔ ZCTA boundaries loaded: {len(zcta_nj):,} ZCTAs")

print("\nFirst 10 ZIPs:")
print(zcta_nj[["zip"]].head(10).to_string(index=False))

print(f"\nCRS: {zcta_nj.crs}")

# ── 4. Census ACS 5-Year Estimates ───────────────────────────────────────────

section("4 of 9 — Census ACS 5-Year Demographic Estimates")

acs_base = f"https://api.census.gov/data/{ACS_YEAR}/acs/acs5"
geo_str  = "&for=zip%20code%20tabulation%20area:*"
key_str  = f"&key={CENSUS_API_KEY}" if CENSUS_API_KEY else ""

print(f"  Fetching ACS {ACS_YEAR} 5-year estimates ({len(ACS_VARS)} variables)...")
if not CENSUS_API_KEY:
    print("  Note: No API key — using anonymous endpoint (may be rate-limited).")

chunk_size = 40
var_chunks = [ACS_VARS[i:i + chunk_size] for i in range(0, len(ACS_VARS), chunk_size)]
chunk_dfs  = []

for idx, chunk in enumerate(var_chunks):
    vars_str = ",".join(chunk)
    acs_url  = f"{acs_base}?get={vars_str}{geo_str}{key_str}"
    acs_resp = http_get(acs_url, timeout=60)

    print("\n" + "=" * 70)
    print(f"Chunk {idx + 1}/{len(var_chunks)}")
    print(f"Status Code: {acs_resp.status_code}")
    print(f"URL: {acs_url}")
    print("Response (first 1000 characters):")
    print(acs_resp.text[:1000])
    print("=" * 70)

    # Stop here so we can inspect the response
    raise SystemExit

    chunk_df = pd.DataFrame(acs_raw[1:], columns=acs_raw[0])
    chunk_df = chunk_df.rename(columns={"zip code tabulation area": "zip"})
    chunk_dfs.append(chunk_df)
    time.sleep(1)

acs_df = chunk_dfs[0]
for next_df in chunk_dfs[1:]:
    acs_df = pd.merge(acs_df, next_df, on="zip", how="outer")

nj_zips = set(normalize_zip(zcta_nj["zip"]).unique())
pre    = len(acs_df)
acs_df = acs_df[normalize_zip(acs_df["zip"]).isin(nj_zips)]
print(f"  ACS NJ filter: {pre:,} → {len(acs_df):,} ZCTAs ({pre - len(acs_df):,} dropped)")

for col in ACS_VARS:
    if col in acs_df.columns:
        acs_df[col] = pd.to_numeric(acs_df[col], errors="coerce")

sentinel = -666666666
acs_df.replace(sentinel, np.nan, inplace=True)

# Derived rates (with zero-division guard)
acs_df["poverty_rate"] = np.where(
    acs_df["B17001_001E"] > 0,
    acs_df["B17001_002E"] / acs_df["B17001_001E"] * 100,
    np.nan,
)
acs_df["snap_rate"] = np.where(
    acs_df["B22003_001E"] > 0,
    acs_df["B22003_002E"] / acs_df["B22003_001E"] * 100,
    np.nan,
)
acs_df["no_vehicle_rate"] = np.where(
    acs_df["B08201_001E"] > 0,
    acs_df["B08201_002E"] / acs_df["B08201_001E"] * 100,
    np.nan,
)

elderly_cols = [
    "B01001_020E", "B01001_021E", "B01001_022E",
    "B01001_023E", "B01001_024E", "B01001_025E",
    "B01001_044E", "B01001_045E", "B01001_046E",
    "B01001_047E", "B01001_048E", "B01001_049E",
]
acs_df["pct_elderly"] = np.where(
    acs_df["B01003_001E"] > 0,
    acs_df[elderly_cols].sum(axis=1) / acs_df["B01003_001E"] * 100,
    np.nan,
)

acs_df.to_csv(DATA_DIR / "acs_df.csv", index=False)

print("\nACS Derived Rates")
for col in ["poverty_rate", "snap_rate", "no_vehicle_rate"]:
    print(f"\n{col}")
    print(acs_df[col].describe().round(2))

print(f"  ✔ ACS loaded: {len(acs_df):,} ZCTAs × {len(acs_df.columns)} columns")
print(acs_df.head(3).to_string(index=False))

print(f"\nChunk {idx + 1}/{len(var_chunks)}")
print(f"URL: {acs_url}")
print(f"Status: {acs_resp.status_code}")
print("Response preview:")
print(acs_resp.text[:1000])

try:
    acs_raw = acs_resp.json()
except ValueError:
    raise RuntimeError("Census API did not return valid JSON.")

# ── 5. CDC PLACES — ZIP-level Health Outcomes ────────────────────────────────

section("5 of 9 — CDC PLACES Health Outcomes")

try:
    print("  Downloading CDC PLACES data...")
    places_resp = http_get(PLACES_URL, timeout=60, stream=True)
    raw_bytes   = b"".join(places_resp.iter_content(chunk_size=1024 * 256))
    places_df   = pd.read_csv(io.BytesIO(raw_bytes))
    print(f"  ✔ CDC PLACES loaded: {len(places_df):,} rows × {len(places_df.columns)} columns")

except Exception as e:
    print(f"  ⚠ CDC PLACES unavailable: {e}")
    cached_file = DATA_DIR / "places_df.csv"
    if cached_file.exists():
        print("  Loading cached CDC PLACES file...")
        places_df = pd.read_csv(cached_file)
        print(f"  ✔ Cached file loaded ({len(places_df):,} rows)")
    else:
        print("  ⚠ No cached file found. Using empty DataFrame.")
        places_df = pd.DataFrame()

# Normalise ZIP column regardless of which source was used
if not places_df.empty:
    print("  CDC PLACES columns:", places_df.columns.tolist())
    zip_col_candidates = ["ZCTA5", "ZIP Code", "ZipCode", "zip_code", "zip", "ZIP", "Zip", "LocationID"]
    zip_col = next((c for c in zip_col_candidates if c in places_df.columns), None)
    if zip_col is None:
        print("  ⚠ Could not find ZIP column in CDC PLACES. Available columns:")
        print("   ", places_df.columns.tolist())
        places_df["zip"] = None
    else:
        print(f"  Using column '{zip_col}' as ZIP identifier")
        places_df = places_df.rename(columns={zip_col: "zip"})
        places_df["zip"] = normalize_zip(places_df["zip"])

# ── 6. NJ ZIP → Municipality Crosswalk ───────────────────────────────────────

section("6 of 9 — NJ ZIP → Municipality Crosswalk")

if not CROSSWALK_PATH.exists():
    raise FileNotFoundError(
        f"\n  Crosswalk not found at: {CROSSWALK_PATH}\n"
        f"  Run nj_zip_crosswalk.py first to generate it."
    )

print(f"  Loading crosswalk from: {CROSSWALK_PATH}")

crosswalk_df = pd.read_csv(
    CROSSWALK_PATH,
    dtype={
        "zip":                str,
        "county_fips":        str,
        "census_tract_geoid": str,
        "nj_mun_code":        str,
        "mcd_geoid":          str,
    },
)
crosswalk_df["zip"] = normalize_zip(crosswalk_df["zip"])

print(f"  ✔ Crosswalk loaded: {len(crosswalk_df):,} rows × {len(crosswalk_df.columns)} columns")
print(f"    Unique ZIPs          : {crosswalk_df['zip'].nunique()}")
print(f"    Unique municipalities: {crosswalk_df['municipality'].nunique()}")
print(f"    Unique counties      : {crosswalk_df['county'].nunique()}")
print(f"    ZIPs with tract      : {crosswalk_df['census_tract_geoid'].notna().sum()}")
print(crosswalk_df.head(3).to_string(index=False))

# ── 7. NJ WIC Authorized Retailers (PDF) ─────────────────────────────────────

section("7 of 9 — NJ WIC Authorized Retailers (PDF)")

print(f"  Extracting WIC vendor table from: {WIC_PDF_PATH}")

wic_pdf_bytes = fetch_pdf_bytes(WIC_PDF_PATH)
wic_rows_raw  = []

with pdfplumber.open(io.BytesIO(wic_pdf_bytes)) as pdf:
    print(f"  Pages: {len(pdf.pages)}")
    for page in pdf.pages:
        for table in page.extract_tables():
            for row in table:
                if row and any(row):
                    wic_rows_raw.append(row)

wic_df_raw         = pd.DataFrame(wic_rows_raw)
wic_df_raw.columns = range(len(wic_df_raw.columns))
header_row         = wic_df_raw.iloc[1].tolist()
wic_df_raw.columns = [str(h).strip() for h in header_row]
wic_df_raw         = wic_df_raw.iloc[2:].reset_index(drop=True)
wic_df_raw         = wic_df_raw[wic_df_raw["eWIC Status"] != "eWIC Status"].copy()
wic_df_raw         = wic_df_raw.rename(columns={
    "eWIC Status": "ewic_status",
    "Name":        "store_name",
    "Address":     "address",
    "City":        "city",
    "State":       "state",
    "Zip":         "zip",
})
wic_nj = wic_df_raw[wic_df_raw["state"].str.strip() == "NJ"].copy()

print(f"  ✔ WIC vendors cleaned: {len(wic_nj):,} NJ rows")
print(wic_nj.head(3).to_string(index=False))

# ── 8. USDA SNAP-Authorized Retailers ────────────────────────────────────────

section("8 of 9 — USDA SNAP-Authorized Retailers")


snap_df = pd.read_csv(SNAP_CSV, low_memory=False)

snap_df.columns = [
    c.strip().upper().replace(" ", "_")
    for c in snap_df.columns
]
state_col = next(
    (c for c in snap_df.columns if c in ("STATE", "STATE_CODE", "ST")), None
)
print(f"  State column detected: {state_col}")

snap_nj = snap_df[snap_df[state_col].str.strip().str.upper() == "NJ"].copy()
print(f"  ✔ NJ SNAP retailers: {len(snap_nj):,} stores")
print(snap_nj.head(3).to_string(index=False))

# # ── 9. NJEDA Designated Food Desert Communities ───────────────────────────────
#
# section("9 of 9 — NJEDA Designated Food Desert Communities")
#
# print(f"  Extracting NJEDA food desert table from: {NJEDA_PDF_PATH}")
#
# njeda_rows      = []
#
# with pdfplumber.open(NJEDA_PDF_PATH) as pdf:
#     print(f"  Total pages: {len(pdf.pages)}")
#     for page_num in [10, 11]:
#         page = pdf.pages[page_num]
#         for table in page.extract_tables():
#             for row in table:
#                 if row and any(row):
#                     njeda_rows.append(row)
#
# njeda_df = pd.DataFrame(njeda_rows)
# print(f"  Raw rows extracted: {len(njeda_df)}")
# print(njeda_df.head(6).to_string(index=False))
#
#
# def clean_njeda(df: pd.DataFrame) -> pd.DataFrame:
#     raw_text     = df.iloc[0, 0]
#     county_names = "|".join(NJ_COUNTIES)
#     pattern      = (
#         r"(\d{1,2})\s+([\w ,/\*\-]+?)\s+"
#         rf"({county_names})\s+([\d.]+)\s+([\d,]+)"
#     )
#     rows = []
#     for m in re.findall(pattern, raw_text):
#         rank, name, county, score, pop = m
#         rows.append({
#             "rank":       int(rank),
#             "name":       name.strip().rstrip("*"),
#             "asterisk":   "*" in name,
#             "county":     county.strip(),
#             "score":      float(score),
#             "population": int(pop.replace(",", "")),
#         })
#     return pd.DataFrame(rows).sort_values("rank").reset_index(drop=True)
#
#
# njeda_clean = clean_njeda(njeda_df)
# print(njeda_clean.to_string(index=False))
# print(f"\n  ✔ {len(njeda_clean)} food desert communities extracted")
# njeda_clean.to_csv(DATA_DIR / "njeda_communities.csv", index=False)
# print("  Saved → data/njeda_communities.csv")

# ── 10. USDA FARA Food Access Research Atlas ─────────────────────────────────

section("10 of 10 — USDA FARA Food Access Research Atlas")

fara = pd.read_excel(
    DATA_DIR / "FoodAccessResearchAtlasData2019.xlsx",
    sheet_name="Food Access Research Atlas",
    dtype={"CensusTract": str},
)
fara["CensusTract"] = fara["CensusTract"].str.zfill(11)
fara_nj = fara[fara["CensusTract"].str.startswith("34")].copy()

FARA_COLS = [
    "CensusTract",
    "LILATracts_1And10",
    "LILATracts_halfAnd10",
    "LILATracts_1And20",
    "LOWINCOMETracts",
    "LA1and10",
    "LAhalfand10",
    "LAPOP1_10",
    "LAPOP05_10",
    "PovertyRate",
    "MedianFamilyIncome",
    "Urban",
    "Pop2010",
]
fara_nj = fara_nj[[c for c in FARA_COLS if c in fara_nj.columns]]
print(f"  NJ tracts loaded: {len(fara_nj)}")
print(f"  Columns kept: {fara_nj.columns.tolist()}")

# ── Tract-level validation against FARA ground truth ─────────────────────────

print("\n── Tract-level FARA validation")

hud_zip_to_tract = pd.read_excel(
    DATA_DIR / "ZIP_TRACT_122025.xlsx",
    dtype={"ZIP": str, "TRACT": str},
)
hud_zip_to_tract.columns = hud_zip_to_tract.columns.str.strip().str.lower()
hud_zip_to_tract["tract"] = hud_zip_to_tract["tract"].str.zfill(11)
hud_zip_to_tract["zip"]   = normalize_zip(hud_zip_to_tract["zip"])
hud_zip_to_tract = hud_zip_to_tract[hud_zip_to_tract["tract"].str.startswith("34")]

tract_zip = hud_zip_to_tract[["zip", "tract", "res_ratio"]].rename(
    columns={"tract": "CensusTract"}
)

fara_zip = tract_zip.merge(fara_nj, on="CensusTract", how="left")

print(f"  FARA NJ tracts total       : {len(fara_nj)}")
print(f"  FARA LILA tracts (1/10)    : {int(fara_nj['LILATracts_1And10'].sum())}")
print(f"  FARA LILA tracts (half/10) : {int(fara_nj['LILATracts_halfAnd10'].sum())}")

matched = fara_zip[fara_zip["LILATracts_1And10"].notna()].copy()
print(f"  Tracts matched via crosswalk: {matched['CensusTract'].nunique()}")
print(f"  Matched LILA tracts (1/10)  : {int(matched['LILATracts_1And10'].sum())}")
print(f"  Match rate                  : {matched['CensusTract'].nunique() / len(fara_nj) * 100:.1f}%")

your_tracts = set(tract_zip["CensusTract"].unique())
fara_tracts = set(fara_nj["CensusTract"].unique())
lila_tracts = set(fara_nj[fara_nj["LILATracts_1And10"] == 1]["CensusTract"].unique())
missed_lila = lila_tracts - your_tracts
print(f"\n  LILA tracts missing from crosswalk: {len(missed_lila)}")
if missed_lila:
    print(f"  Sample missing: {list(missed_lila)[:5]}")

print(f"\n  Tract overlap: {tract_zip['CensusTract'].isin(fara_nj['CensusTract']).sum()} / {len(tract_zip)} rows matched")
print(f"  Unique tracts in tract_zip : {tract_zip['CensusTract'].nunique()}")
print(f"  Unique tracts in fara_nj   : {fara_nj['CensusTract'].nunique()}")
print(f"  Sample tract_zip GEOIDs    : {tract_zip['CensusTract'].head(3).tolist()}")
print(f"  Sample fara_nj GEOIDs      : {fara_nj['CensusTract'].head(3).tolist()}")

# ── Aggregate to ZIP via crosswalk ────────────────────────────────────────────

fara_agg = fara_zip.groupby("zip").agg(
    usda_lila_1_10     = ("LILATracts_1And10",    "max"),
    usda_lila_half_10  = ("LILATracts_halfAnd10", "max"),
    usda_lila_1_20     = ("LILATracts_1And20",    "max"),
    usda_la_1_10       = ("LA1and10",              "max"),
    usda_la_half_10    = ("LAhalfand10",           "max"),
    usda_lapop_1_10    = ("LAPOP1_10",             "sum"),
    usda_lapop_half_10 = ("LAPOP05_10",            "sum"),
    usda_poverty_rate  = ("PovertyRate",           "mean"),
    usda_median_income = ("MedianFamilyIncome",    "mean"),
    usda_urban         = ("Urban",                 "max"),
    usda_pop2010       = ("Pop2010",               "sum"),
).reset_index()

print(f"  ✔ FARA aggregated to ZIP: {len(fara_agg)} zips")
print(f"  LILA zips (1/10 def):     {fara_agg['usda_lila_1_10'].sum():.0f}")
print(f"  LILA zips (half/10 def):  {fara_agg['usda_lila_half_10'].sum():.0f}")
print(fara_zip["LILATracts_1And10"].value_counts(dropna=False))
print(f"  Matched tracts: {fara_zip['CensusTract'].notna().sum()} / {len(fara_zip)}")

# ── ZIP-level agreement with FARA ─────────────────────────────────────────────

print("\n── ZIP-level agreement")
fara_desert_zips = set(fara_agg[fara_agg["usda_lila_1_10"] == 1]["zip"].unique())
print(f"  ZIPs containing a LILA tract (1/10): {len(fara_desert_zips)}")
print(f"  Sample: {sorted(fara_desert_zips)[:10]}")

# ── OSM store counts by ZIP ───────────────────────────────────────────────────

section("OSM Store Counts by ZIP")

osm_records = []
for el in osm_data.get("elements", []):
    tags = el.get("tags", {})
    lat  = el.get("lat") or (el["center"]["lat"] if "center" in el else None)
    lon  = el.get("lon") or (el["center"]["lon"] if "center" in el else None)
    if lat is None or lon is None:
        continue
    key = "shop" if "shop" in tags else "amenity" if "amenity" in tags else None
    if key is None:
        continue
    label = OSM_TYPE_MAP.get((key, tags[key]))
    if label is None:
        continue
    osm_records.append({"lat": lat, "lon": lon, "type": label})

osm_gdf = gpd.GeoDataFrame(
    osm_records,
    geometry=gpd.points_from_xy(
        [r["lon"] for r in osm_records],
        [r["lat"] for r in osm_records],
    ),
    crs="EPSG:4326",
)

osm_joined = gpd.sjoin(osm_gdf, zcta_nj[["zip", "geometry"]], how="left", predicate="within")
osm_joined = osm_joined.dropna(subset=["zip"])

osm_counts = (
    osm_joined.groupby(["zip", "type"])
    .size()
    .unstack(fill_value=0)
    .rename(columns=OSM_COL_MAP)
    .reset_index()
)

for col in OSM_COL_MAP.values():
    if col not in osm_counts.columns:
        osm_counts[col] = 0

osm_counts["zip"] = normalize_zip(osm_counts["zip"])

# ── Save & Summary ────────────────────────────────────────────────────────────

section("Acquisition Complete — All Datasets in Memory")

DATA_DIR.mkdir(parents=True, exist_ok=True)

places_df.to_csv(DATA_DIR / "places_df.csv", index=False)
crosswalk_df.to_csv(DATA_DIR / "crosswalk_df.csv", index=False)
wic_nj.to_csv(DATA_DIR / "wic_df.csv", index=False)

# Normalise SNAP zip column name before saving
zip_col_snap = next(
    (c for c in snap_nj.columns if c in ("ZIP", "ZIP_CODE", "ZIPCODE")), None
)
if zip_col_snap and zip_col_snap != "zip":
    snap_nj = snap_nj.rename(columns={zip_col_snap: "zip"})

snap_nj.to_csv(DATA_DIR / "snap_df.csv", index=False)
fara_agg.to_csv(DATA_DIR / "fara_agg.csv", index=False)
osm_counts.to_csv(DATA_DIR / "osm_counts.csv", index=False)

osm_total = len(osm_data.get("elements", []))

# ── Diagnostic: trace out-of-state ZIPs ──────────────────────────────────────
# Left here so you can see at least one of the challenges with ZIP-code
# boundary handling — ZIPs on a state border can bleed into adjacent states.

watch = ["19153", "10977"]
snap_zip_series = snap_nj["zip"] if "zip" in snap_nj.columns else pd.Series(dtype=str)

print("\n── OUT-OF-STATE ZIP TRACE ──────────────────────────────────────────")
print(f"zcta_nj       : {[z for z in watch if z in set(zcta_nj['zip'].astype(str).str.zfill(5))]}")
print(f"acs_df        : {[z for z in watch if z in set(acs_df['zip'].astype(str).str.zfill(5))]}")
print(f"places_df     : {[z for z in watch if z in set(places_df['zip'].astype(str).str.zfill(5))]}")
print(f"crosswalk_df  : {[z for z in watch if z in set(crosswalk_df['zip'].astype(str).str.zfill(5))]}")
print(f"fara_agg      : {[z for z in watch if z in set(fara_agg['zip'].astype(str).str.zfill(5))]}")
print(f"osm_counts    : {[z for z in watch if z in set(osm_counts['zip'].astype(str).str.zfill(5))]}")
print(f"snap_nj       : {[z for z in watch if z in set(snap_zip_series.astype(str).str.zfill(5))]}")

print(f"""
  Dataset                      Shape
  ──────────────────────────────────────────────────────
  OSM zips                     {len(osm_counts)} ZIPs × {len(osm_counts.columns)} cols
  OSM food locations           {osm_total:>7,} elements
  ZCTA boundaries (NJ)         {zcta_nj.shape[0]:>7,} rows × {zcta_nj.shape[1]} cols
  County boundaries (NJ)       {counties_nj.shape[0]:>7,} rows × {counties_nj.shape[1]} cols
  Census ACS estimates         {acs_df.shape[0]:>7,} rows × {acs_df.shape[1]} cols
  CDC PLACES health data       {places_df.shape[0]:>7,} rows × {places_df.shape[1]} cols
  ZIP → Municipality crosswalk {crosswalk_df.shape[0]:>7,} rows × {crosswalk_df.shape[1]} cols
  WIC authorized retailers     {len(wic_nj):>7,} rows (NJ only)
  SNAP authorized retailers    {len(snap_nj):>7,} rows (NJ only)
  NJEDA food desert communities{len(njeda_clean):>7,} communities
  USDA FARA (tract-level)      {fara_nj.shape[0]:>7,} rows × {fara_nj.shape[1]} cols
  USDA FARA (zip-aggregated)   {fara_agg.shape[0]:>7,} rows × {fara_agg.shape[1]} cols
""")