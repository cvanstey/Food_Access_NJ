# Food Access NJ

This project analyzes food landscapes across New Jersey ZIP codes by comparing USDA Food Access Research Atlas methodologies against complementary spatial, demographic, and retailer datasets to evaluate food access, food deserts, and food swamps at the ZIP and tract level.

## Hypothesis

**Working hypothesis:** New Jersey ZIP codes with higher concentrations of older adults experience significantly poorer food access, characterized by greater supermarket distance and lower availability of SNAP and WIC food retailers.

**Null hypothesis (H₀):** Food access characteristics do not differ based on the concentration of older adults within New Jersey ZIP codes.

## Methodology Reference

It was important to compare methods of food access measurement implemented by federal and state levels. 

| Method / Tool | Focus Area | Core Approach / Metric | Formula | Primary Source |
|---|---|---|---|---|
| USDA Research Atlas | Food Deserts | Poverty and distance thresholds | Distance ≥ 1 mi (urban) or ≥ 10 mi (rural) + Poverty Rate ≥ 20% | USDA ERS Food Access Research Atlas |
usda_desert_flag/is_desert_usda recreates the poverty≥20% + distance threshold rule directly. is_desert_fara instead pulls the official FARA LILA tract flag (usda_lila_1_10) from upstream data — this means there is a self-computed and an authoritative version, compared against each other in the print block. |

| RFEI | Food Swamps | Ratio of unhealthy to healthy retailers | (Fast Food + Convenience Stores) / (Supermarkets + Produce Markets) | Public Health Research Standard |
03 features.py Section 1 — rfei = (fast_food + convenience) / (supermarket + grocery + produce_market). rfei_full extends it with dollar stores. |

| mRFEI | Food Swamps | % of healthy retailers in tract | (Healthy Food Retailers / Total Food Retailers) × 100 | CDC |
03 features.py Section 2 — mrfei = (healthy / total) × 100 |

| NJ Food Swamp Score | Food Swamps | Distance to swamp vs. supermarket | (Shortest dist. to swamp outlet / Shortest dist. to supermarket), scaled 0–100 | NJ DCA / NJEDA Approved Food Deserts Map |
03 features.py Section 4 — nj_swamp_score = nearest fast food / nearest supermarket distance, scaled 0–100. |

| GIS Network Analysis | Both | Street routes and travel times | Network distance (actual routing) vs. Euclidean distance (straight line) | Urban Planning GIS Frameworks |
nearest_supermarket_miles / nearest_fastfood_miles are used as inputs, however, supermarkets_within_5mi is explicitly commented # placeholder using ZIP count until spatial buffer computed — the point-in-polygon buffer described in the table isn't implemented yet. |


| mrfei_wic (WIC-specific mRFEI variant) | dollar store ratio/dominance/desert flags |
| rule-based access_typology classifier | (True Desert / Food Swamp / Food Mirage / Dollar Store Desert) |
| 4-method swamp consensus vote | (is_swamp_consensus, swamp_method_count) |

|Food Environment Index | composite_vuln_index, novehicle_vuln_score, elderly_vuln_score blend proximity + poverty/access into a weighted score — but use percentile-rank weighting, not the County Health Rankings & Roadmaps CHR&R 0–10 ranked-average method used.|

| Machine Learning | Food Deserts | Predicts access using census variables | Predictive modeling (Hot-Spot Analysis, regression) | Advanced Spatial Data Science |
| Food Environment Index | Combined | Merges proximity with food insecurity | Weighted average rank (0–10) of limited access + food insecurity | County Health Rankings & Roadmaps |

| Spatial Buffering | Food Swamps | Counts fast food within walking zones | Point-in-polygon aggregation within 0.25/0.5 mi buffers | GIS Spatial Analysis |

| Composite Factor Score | Combined | Weights 24 neighborhood indicators | Iterated principal factor analysis, orthogonal varimax rotation | NJ Food Desert Designation Methodology | I used a different method here - Same composite_vuln_index is the closest analog, but it's a hand-weighted percentile composite — not the iterated principal factor analysis with varimax rotation specified by the NJEDA. I would need actual factor analysis, not weighted ranks to achieve this.|

| Structural Determinants Model | Food Swamps & Transit | Intersects transport barriers with retail environments | Multi-variable intersectional analysis of retail density vs. transit equity | IJERPH / MDPI Study | TBD IN THE ACRGIS PIECE |

| FDC Programmatic Allocation Framework | Program Deployment | Maps statistical scores to capital investments, tax credits, grants | Rank-ordered classification (1–50) across 1,015 block groups | NJEDA Food Security Products Deck (March 2024) |

**Related references:** - I was thinking about highlighting where the state is encouraging disepensaries in economically disadvantaged areas because I believe there is a correlation between food quality and vice retail. This is not currently part of the analysis. 
- NJ Economically Disadvantaged Areas: https://www.nj.gov/cannabis/businesses/priority-applications/
- NJEDA Priority Applications: https://www.nj.gov/cannabis/businesses/priority-applications/eda/

## Pipeline Overview

This project runs as a sequence of 9 scripts, from data acquisition through targeted analysis and ZIP-level lookup. `01_load_data.py` is the data acquisition stage — it downloads/reads all source datasets, validates and cleans them, and writes intermediate files back into `data/` for use by later steps. See **Running the project** below for the full execution order and a description of each stage.

## Setup

### 1. Clone the repository

```bash
git clone [repo-url]
cd [repo-name]
```

### 2. Install dependencies

```bash
pip install requests geopandas pandas pdfplumber numpy openpyxl
```

### 3. Set up a Census API key (optional, recommended)

A free Census API key removes rate limits on ACS data requests.

- Sign up: https://api.census.gov/data/key_signup.html
- Set it as an environment variable — do **not** hardcode it in the script:

```bash
export CENSUS_API_KEY="your-key-here"
```

### 4. Download the local data files

The data files required to run this project are too large to host on GitHub. Download them from Google Drive instead:

**Download link:** [https://drive.google.com/drive/folders/14_zmFgw-F0yEetg64rVz1eCSIC9dVMWp?usp=drive_link]

Unzip the contents into a `data/` folder at the root of the project so it matches the structure below.

#### Required files checklist

Place these inside `data/`:

- [ ] `nj_zip_complete.csv` — full ZIP → municipality/census tract crosswalk
- [ ] `nj_zip_crosswalk.csv` — valid NJ ZIP list, used to filter out-of-state border ZCTAs
- [ ] `zcta_nj.gpkg` — NJ ZCTA boundary geometries
- [ ] `FoodAccessResearchAtlasData2019.xlsx` — USDA FARA, tract-level
- [ ] `ZIP_TRACT_122025.xlsx` — HUD ZIP-to-tract crosswalk
- [ ] `snap_retailer_location_data.csv` — USDA SNAP-authorized retailers
- [ ] `food-security-product-deck.-march-2024.pdf` — NJEDA deck (source for food desert community rankings)

These 7 files are confirmed sufficient — everything else the pipeline needs is pulled live from public APIs at runtime (see below).

> Note: if `nj_zip_complete.csv` is ever missing, `01_load_data.py` will raise a `FileNotFoundError` instructing you to run `nj_zip_crosswalk.py` first to regenerate it. That script isn't part of this checklist since the generated file is already included in the Drive download — but keep it in mind if you ever need to rebuild the crosswalk from scratch.

### Expected project structure

```
[project-root]/
├── 01_load_data.py
├── data/
│   ├── nj_zip_complete.csv
│   ├── nj_zip_crosswalk.csv
│   ├── zcta_nj.gpkg
│   ├── FoodAccessResearchAtlasData2019.xlsx
│   ├── ZIP_TRACT_122025.xlsx
│   ├── snap_retailer_location_data.csv
│   └── food-security-product-deck.-march-2024.pdf
└── README.md
```

## External / Live Data Sources

Pulled automatically by `01_load_data.py` at runtime — no manual download needed, just network access:

| Source | Dataset | Auth Required |
|---|---|---|
| OSM Overpass API | Food-related point locations (supermarkets, convenience stores, fast food, etc.) | No |
| Census TIGER | NJ county boundary shapefiles | No |
| Census ACS 5-Year API | ZCTA-level demographic estimates | Optional (recommended — see Setup step 3) |
| CDC PLACES (Socrata) | ZIP-level health outcome data | No |
| NJ DOH | WIC-authorized vendor PDF | No |

OSM results are cached locally to `data/osm_data.json` after the first run to avoid re-querying Overpass on subsequent runs.

## Running the project
### TBD - run_pipeline.py orchestrator
Run the pipeline in order:

```bash
python 01_load_data.py          # Data acquisition — downloads/reads all source datasets
python 02a_nearest.py           # This creates distance calculations for supermarkets, convenience stores, etc.Run this before merge_sources.py
python 02b_merge_sources.py      # Merges all cleaned data sources into a single zip-level feature table.
python 03_features.py           # Builds derived features/metrics for analysis
python 04_model.py              # Statistical / ML modeling
python 05_reports.py            # Generates report outputs
python 06_analysis.py           # Core analysis
python 07_targeted_analysis.py  # Targeted/sub-population analysis (e.g. elderly ZIPs, per hypothesis)
python 08_zip_lookup.py         # ZIP-level lookup tool/utility
```


**`01_load_data.py`** performs data acquisition only: it downloads/reads all source datasets, prints a confirmation summary for each (10 sections total), and writes cleaned intermediate files into `data/` — `acs_df.csv`, `places_df.csv`, `crosswalk_df.csv`, `wic_df.csv`, `snap_df.csv`, `fara_agg.csv`, `osm_counts.csv`, `njeda_communities.csv` — for use by later pipeline steps.

### Expected output

A successful run loads/produces the following datasets:

| Dataset | Shape |
|---|---|
| OSM zips | 535 ZIPs × 18 cols |
| OSM food locations | 37,818 elements |
| ZCTA boundaries (NJ) | 598 rows × 3 cols |
| County boundaries (NJ) | 21 rows × 18 cols |
| Census ACS estimates | 598 rows × 59 cols |
| CDC PLACES health data | 32,520 rows × 84 cols |
| ZIP → Municipality crosswalk | 598 rows × 36 cols |
| WIC authorized retailers | 889 rows (NJ only) |
| SNAP authorized retailers | 5,447 rows (NJ only) |
| NJEDA food desert communities | 25 communities |
| USDA FARA (tract-level) | 2,002 rows × 12 cols |
| USDA FARA (zip-aggregated) | 691 rows × 12 cols |

Exit code `0` indicates a clean run with no validation errors.

The script ends with an **out-of-state ZIP trace** — a diagnostic check confirming that border ZCTAs which geometrically touch NJ counties (e.g. `19153` in Philadelphia, `10977` in Spring Valley, NY) are correctly filtered out before reaching the final ZIP-level datasets. Seeing these ZIPs in early-stage debug output is expected; seeing them in final aggregated files (`fara_agg`, `osm_counts`, etc.) would indicate a bug.

## Data sources

| File | Description | Source |
|---|---|---|
| `nj_zip_complete.csv` | Full ZIP → municipality/census tract crosswalk (generated by `nj_zip_crosswalk.py`) | Self-generated |
| `nj_zip_crosswalk.csv` | Valid NJ ZIP list used to filter out-of-state border ZCTAs | Self-generated |
| `zcta_nj.gpkg` | NJ ZCTA boundary geometries | Census TIGER |
| `FoodAccessResearchAtlasData2019.xlsx` | USDA Food Access Research Atlas, 2019 (tract-level) | USDA ERS |
| `ZIP_TRACT_122025.xlsx` | ZIP-to-census-tract crosswalk, Dec 2025 | HUD USPS Crosswalk |
| `snap_retailer_location_data.csv` | SNAP-authorized retailer locations | USDA FNS |
| `food-security-product-deck.-march-2024.pdf` | NJEDA Food Security product/reference deck, March 2024 (source for food desert community rankings) | NJEDA |

## Notes

- Make sure the `data/` folder is excluded from version control (see `.gitignore`) — both the downloaded source files and the intermediate files `01_load_data.py` generates (`*.csv`, `osm_data.json`) shouldn't go to GitHub.
- Never commit API keys. Use environment variables and a `.env` file excluded via `.gitignore` instead of hardcoding credentials in scripts.
- The Overpass API call can take 30–120 seconds on a cold run (no cache); subsequent runs use the local `osm_data.json` cache and are much faster.
