# Food Access NJ

Analyzes food landscapes across New Jersey ZIP codes by comparing USDA Food Access Research Atlas methodologies against complementary spatial, demographic, and retailer datasets — evaluating food access, food deserts, and food swamps at the ZIP and tract level.

---

## Features

- Integrates USDA, Census, CDC PLACES, NJEDA, SNAP, WIC, and OpenStreetMap datasets
- Calculates nearest supermarket and food retailer distances
- Evaluates food access patterns aggregated to ZIP/ZCTA reporting units while preserving tract-level USDA comparisons where available
- Reconciles three food desert definitions (USDA LILA, NJEDA Food Desert Community, and a trained classifier) into a consensus flag
- Reconciles three food swamp metrics (RFEI, mRFEI, WIC-specific mRFEI) into a consensus flag
- Generates separate ZIP-level vulnerability indices for elderly and no-vehicle populations, rather than one blended score
- Produces reports and statistical analyses
- Interactive ZIP code lookup utility

## Hypothesis

**Working hypothesis:** New Jersey ZIP codes with higher concentrations of older adults experience significantly poorer food access, characterized by greater supermarket distance and lower availability of SNAP and WIC food retailers.

**Null hypothesis (H₀):** Food access characteristics do not differ based on the concentration of older adults within New Jersey ZIP codes.

---

## Setup

### 1. Clone the repository

```bash
git clone [repo-url]
cd [repo-name]
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up a Census API key (optional, recommended)

A free Census API key removes rate limits on ACS data requests.

- Sign up: https://api.census.gov/data/key_signup.html
- Set it as an environment variable — do **not** hardcode it in the script:

```bash
export CENSUS_API_KEY="your-key-here"
```

### 4. Download project data

The datasets required to reproduce this analysis are too large to store in GitHub. Download the data package from Google Drive:

**Download data package:**  
[Google Drive Data Folder](https://drive.google.com/drive/folders/14_zmFgw-F0yEetg64rVz1eCSIC9dVMWp?usp=drive_link)

After downloading:

1. Extract the files.
2. Create a `data/` directory in the project root if it does not already exist.
3. Place all downloaded files directly inside the `data/` folder.

The expected project structure is:

```text
NJ_Food_Access/
│
├── data/
│   ├── nj_zip_complete.csv
│   ├── nj_zip_crosswalk.csv
│   ├── zcta_nj.gpkg
│   ├── FoodAccessResearchAtlasData2019.xlsx
│   ├── ZIP_TRACT_122025.xlsx
│   ├── snap_retailer_location_data.csv
│   └── food-security-product-deck.-march-2024.pdf
│
├── src/
├── plots/
├── reports/
└── README.md
```

> If `nj_zip_complete.csv` is missing, `01_load_data.py` will raise a `FileNotFoundError` and prompt you to run `nj_zip_crosswalk.py` to regenerate it. That script isn't part of this checklist since the generated file is included in the Drive download — but keep it in mind if you need to rebuild the crosswalk from scratch.

### Expected project structure

```
NJ_Food_Access/
│
├── data/
├── pipeline_logs/
├── plots/
├── reports/
├── src/
│   ├── run_pipeline.py
│   ├── pipeline_utils.py
│   ├── 00a_build_crosswalk.py
│   ├── 00b_enrich_crosswalk.py
│   ├── 01_load_data.py
│   ├── 02a_nearest.py
│   ├── 02b_merge_sources.py
│   ├── 02c_clean_NJ_features_zip2.py
│   ├── 03_features.py
│   ├── 04_model.py
│   ├── 05_reports.py
│   ├── 06_analytics.py
│   ├── 07_targeted_analysis.py
│   ├── 08_zip_lookup.py
│   ├── 09_poor_in_wealthy_tracts.py
│   ├── mcnemar_poverty_requirement.py
│   ├── rename_columns.py
│   └── njzipfilter.py
│
├── testing/
│   └── compare_density.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Google Colab

A Google Colab notebook is provided for reproducible execution:

[Open in Google Colab](https://colab.research.google.com/drive/1MB6RLrqhqrNn8QonWsdPujrutfzU-swH?usp=sharing)

The notebook:
1. Clones this repository
2. Installs dependencies
3. Mounts the Google Drive data package
4. Runs the complete pipeline

---
## Running the Pipeline

Run the full pipeline with the orchestrator:

```bash
python src/run_pipeline.py
```

This runs every stage below in order and stops immediately if one fails, logging full output to `pipeline_logs/`. Resume from a failed stage with `python run_pipeline.py --from <stage_id>`, or run a single stage with `python run_pipeline.py --only <stage_id>`.

### The Pipeline

The orchestrator's `STAGES` list runs **12 scripts** in order:

```text
01_load_data.py
02a_nearest.py
02b_merge_sources.py
02c_clean_NJ_features_zip2.py
03_features.py
04_model.py
05_reports.py
06_analytics.py
../testing/compare_density.py
07_targeted_analysis.py
mcnemar_poverty_requirement.py
09_poor_in_wealthy_tracts.py
```

`08_zip_lookup.py` is **not** in this list — it's commented out in `run_pipeline.py` on purpose, since it's an interactive lookup tool rather than a batch stage. Run it manually and separately (see below); `python run_pipeline.py` will never invoke it.

> **Check this before relying on the orchestrator:** `run_pipeline.py`'s `STAGES` list must contain the exact filename `run_stage()` looks for on disk. If the file was recently renamed to `09_poor_in_wealthy_tracts.py`, confirm the `STAGES` list was updated to match — otherwise the orchestrator will silently `[SKIP]` this stage (file-not-found) rather than erroring.

Two stages still don't follow the numbered `0X_name.py` convention:

- **`compare_density.py`** (in `../testing/`, run between stages 06 and 07) — the density-ablation / circularity check comparing desert-classifier performance with and without `pop_density` as a feature (Configs A/B/C).
- **`mcnemar_poverty_requirement.py`** (run after `07_targeted_analysis.py`) — an exact McNemar's test on whether dropping USDA's poverty requirement from the desert definition changes classification for a non-random share of ZIPs, plus a comparison of the "gained-only" ZIPs against core LILA ZIPs.

`09_poor_in_wealthy_tracts.py` (run last) now follows the numbered convention. It picks up where `mcnemar_poverty_requirement.py` leaves off: among the "gained-only" ZIPs identified there (ZIPs that would only qualify as food deserts if USDA's poverty requirement were dropped — i.e. `usda_la_1_10=1, usda_lila_1_10=0`), it asks where the actual poor residents are hiding, via two lenses — ranking those ZIPs by *raw count* of residents below poverty (not rate, since a low-rate ZIP can still contain a large poor population), and flagging census tracts inside those ZIPs whose poverty rate sits 10+ points above the ZIP's own average. Its docstring is explicit that this does **not** argue for removing food-desert status from anyone — it's the opposite concern: are poor residents being undercounted because they live inside a ZIP that reads as wealthy on average. Outputs: `gained_only_ranked_by_poor_count.csv`, `hidden_poverty_pockets.csv`, `hidden_pocket_zip_summary.csv`.

Because `stage_id()` derives `--from`/`--only` ids by splitting each filename on its first underscore, the clean ids (`01`, `02a`, `04`, `09`, etc.) work for every numbered script, including `09_poor_in_wealthy_tracts.py` now. `mcnemar_poverty_requirement.py`'s id is still `mcnemar` (no numeric prefix).

Stage 4 (modeling) is by far the longest single stage, typically taking up to 6 minutes. It compares Logistic Regression, Random Forest, and Gradient Boosting classifiers for consensus desert-status prediction (Gradient Boosting is selected on cross-validated AUC), fits 20 individual Random Forest health-outcome regressors plus companion OLS models, and validates thoroughly: a leave-one-county-out spatial cross-validation (14 county-level fits — 7 of NJ's 21 counties are skipped for having zero desert cases), a 2,000-resample bootstrap for confidence intervals, and permutation importance with 20 repeats per feature. If a run appears to hang here, it's very likely still working — this stage does more computation than the rest of the pipeline combined.

Note that stage output does not stream live when run through the orchestrator — each stage runs as a subprocess with `stdout` redirected straight to its log file in `pipeline_logs/`, so the notebook/terminal will show nothing between stage headers no matter how long a stage takes. To check that a long-running stage (especially stage 4) is still progressing, tail the current log file in a separate cell/terminal:

```bash
tail -f pipeline_logs/run_<timestamp>.log
```

**Manual / individual stages**, if you need to run one by hand:

```bash
If nj_zip_complete.csv is missing, regenerate it by running:

python 00a_build_crosswalk.py
python 00b_enrich_crosswalk.py
python 01_load_data.py                    # Data acquisition — downloads/reads all source datasets
python 02a_nearest.py                     # Distance calculations (supermarkets, convenience stores, etc.)
python 02b_merge_sources.py               # Merges all cleaned sources into a single ZIP-level feature table
python 02c_clean_NJ_features_zip2.py      # Cleans nj_zip_features_v2.csv (dedup, sentinel values, type fixes) → nj_zip_features_v2_clean.csv
python 03_features.py                     # Builds derived features and metrics
python 04_model.py                        # Statistical / ML modeling
python 05_reports.py                      # Generates report outputs
python 06_analytics.py                    # Core analysis
python ../testing/compare_density.py      # Density-ablation / circularity check (not run by orchestrator until stage 06 completes)
python 07_targeted_analysis.py            # Sub-population analysis (elderly ZIPs, per hypothesis)
python mcnemar_poverty_requirement.py     # McNemar's test: effect of dropping USDA's poverty requirement
python 09_poor_in_wealthy_tracts.py       # Hidden poverty pocket / tract-masking analysis (run after 02b_merge_sources.py at minimum)
python 08_zip_lookup.py                   # Interactive ZIP-level lookup tool — NOT run by the orchestrator; run manually
```

`rename_columns.py` is not a standalone stage — it's imported directly by `02b_merge_sources.py` to rename ACS/PLACES columns before saving. `pipeline_utils.py` is likewise a shared module, not a stage.

*Note: `clean_NJ_features_zip2.py` isn't yet renamed to match the pipeline's numbered convention (e.g. `02c_clean_features.py`) — rename it and update `run_pipeline.py`'s `STAGES` list once you do. The same applies to `mcnemar_poverty_requirement.py`, which could become e.g. `10_mcnemar_poverty.py`.*

`01_load_data.py` performs data acquisition only — it downloads/reads all source datasets, prints a confirmation summary for each of its 10 sections, and writes cleaned intermediate files into `data/` for use by later steps: `acs_df.csv`, `places_df.csv`, `crosswalk_df.csv`, `wic_df.csv`, `snap_df.csv`, `fara_agg.csv`, `osm_counts.csv`, `njeda_communities.csv`.

### Expected output from `01_load_data.py`

| Dataset | Shape |
|---|---|
| OSM ZIPs | 535 ZIPs × 18 cols |
| OSM food locations | 37,818 elements (2,208 supermarkets, 8,814 fast-food outlets, 5,230 convenience stores after filtering) |
| ZCTA boundaries (NJ) | 598 rows × 3 cols |
| County boundaries (NJ) | 21 rows × 18 cols |
| Census ACS estimates | 598 rows × 59 cols |
| CDC PLACES health data | 32,520 rows × 84 cols |
| ZIP → Municipality crosswalk | 598 rows × 36 cols |
| WIC authorized retailers | 890 rows (NJ only) |
| SNAP authorized retailers | 5,447 rows (NJ only) |
| NJEDA food desert communities | 25 communities |
| USDA FARA (tract-level) | 2,002 rows × 12 cols |
| USDA FARA (ZIP-aggregated) | 691 rows × 12 cols |

The script ends with an **out-of-state ZIP trace** — a diagnostic check confirming that border ZCTAs which geometrically touch NJ counties (e.g. `19153` in Philadelphia, `10977` in Spring Valley, NY) are correctly filtered out before reaching the final ZIP-level datasets. Seeing these ZIPs in early-stage debug output is expected; seeing them in final aggregated files would indicate a bug.

---
## Statistical Modeling

The project evaluates food access vulnerability using:

- Exploratory data analysis and correlation analysis
- Logistic regression, Random Forest, and Gradient Boosting classification (compared for consensus desert-status prediction; Gradient Boosting selected on cross-validated AUC)
- Random Forest regressors for 20 CDC PLACES health outcomes, plus OLS models controlling for a principal-component deprivation index
- Leave-one-county-out spatial cross-validation
- Bootstrap confidence intervals for model evaluation

Model features include demographic, socioeconomic, transportation, and environmental variables while excluding proximity features used to define the target outcome to reduce leakage.

### Food Desert Methods

Three independently constructed desert definitions are reconciled into a consensus flag:

- **USDA Food Access Research Atlas (FARA) / LILA** — the federal Low Income Low Access measure, evaluated at both the 1-mile/10-mile and ½-mile/10-mile thresholds. Official USDA FARA flags are imported for direct comparison.
- **NJEDA Food Desert Community (FDC) designation** — the state's sub-municipal block-group-based classification.
- **A trained classifier** — a Gradient Boosting model fit on seven socioeconomic features (median income, poverty rate, SNAP rate, transit rate, no-vehicle rate, college-education rate, elderly rate) to predict desert status.

A ZIP is flagged as a **consensus food desert** when at least 2 of these 3 methods agree (45 consensus deserts statewide). Consensus deserts are further split into two subtypes that respond to different underlying causes of isolation: **socioeconomic deserts** (isolated and poor) and **structural deserts** (isolated but not poor, disproportionately elderly).

### Food Swamp Methods

**RFEI (Retail Food Environment Index)**
Ratio of unhealthy to healthy retailers, following Cooksey-Stowers (2017): `(fast_food + convenience) / (supermarket + grocery + produce_market)`. An extended variant (`rfei_full`) adds dollar stores to the numerator.

**mRFEI (Modified RFEI)**
CDC method measuring the percentage of healthy retailers among all food retailers: `(healthy / total) × 100`.

**WIC-specific mRFEI**
Substitutes WIC-certified vendors as the healthy retailer count.

**3-Method Consensus Vote**
These three swamp methods are combined into a consensus flag (`is_swamp_consensus`) requiring agreement from at least 2 of 3 methods, with a continuous score (`swamp_score_continuous`) and method count (`swamp_method_count`) for transparency. This consensus flags 369 ZIPs (61.7% of the state) as food swamps — a high enough share that swamp status alone does not separate at-risk ZIPs from healthy ones as cleanly as desert status does.

### Composite & Vulnerability Scores

A core finding of the accompanying analysis is that blending distinct sources of need into a single composite score can conceal which residents within a ZIP actually bear the burden — a structurally isolated, elderly-heavy ZIP and a poor, transportation-poor ZIP can post nearly identical composite scores while needing entirely different interventions. For that reason:

- **Poverty is treated as a control/stratification variable**, not folded into a labeled vulnerability score.
- **`elderly_vuln_score`** and **`novehicle_vuln_score`** are computed as separate percentile-rank-weighted indices within those sub-populations, and are the recommended indicators for targeting age-friendly transit vs. income/transit interventions respectively.
- A **`composite_vuln_index`**, blending supermarket distance, RFEI, poverty rate, vehicle access, and elderly concentration via percentile-rank weighting, is still computed and used for exploratory/reference purposes — but should not substitute for the separate sub-population scores in funding or policy decisions. Note: this differs from the County Health Rankings & Roadmaps Food Environment Index, which uses a ranked-average method on a 0–10 scale, and from the NJEDA composite factor score, which uses iterated principal factor analysis with orthogonal varimax rotation across 24 neighborhood indicators — a full factor analysis implementation is a planned improvement.

**Rule-based Access Typology** classifies each ZIP into one of six categories: True Desert, Food Swamp, Food Mirage, Well Served, Dollar Store Desert, or Desert-Swamp Overlap.

### Implementation Notes

- `supermarkets_within_5mi` is currently a placeholder using ZIP-level store count; point-in-polygon spatial buffering is not yet implemented.
- GIS network analysis fields (`nearest_supermarket_miles`, `nearest_fastfood_miles`) use Euclidean distance, not routed network distance. Full network routing is planned for a future ArcGIS component.
- The Structural Determinants Model (multi-variable intersectional analysis of retail density vs. transit equity) is also deferred to the ArcGIS component.
- The FDC Programmatic Allocation Framework (NJEDA rank-ordered classification across 1,015 block groups) is referenced as a comparison target but not yet implemented.

### On OSM Classifications

OSM uses its own tagging taxonomy that does not map cleanly onto food access research definitions. Wawa is tagged `shop=convenience` in OSM (inflating the RFEI numerator), and small grocers are often tagged the same way (deflating the denominator). RFEI and mRFEI store counts therefore use SNAP/WIC data where available; OSM data is used primarily for spatial features (nearest-distance calculations) where it is more reliable.

### Known Issues

- **`05_reports.py`'s statewide typology breakdown uses stale category names.** The `nj_statewide_summary.csv` output currently prints `Transit Desert` and `Adequate Access` (both 0%) instead of `Well Served` and `Desert-Swamp Overlap`, and the reported percentages sum to ~69% rather than 100%. The 6-category typology (`True Desert`, `Food Swamp`, `Food Mirage`, `Well Served`, `Dollar Store Desert`, `Desert-Swamp Overlap`) is correct everywhere else in the pipeline (e.g. `access_typology_profiles.csv`) — only this one summary block needs its category list updated.
- **Two different "swamp" figures appear side by side without explanation.** `pct_food_swamp_consensus` in the statewide summary reports 71.6% (428 ZIPs — any ZIP meeting the 2-of-3 `is_swamp_consensus` flag), while the access typology table in the same report shows `Food Swamp` at 61.7% (369 ZIPs — the subset whose *typology label* is purely "Food Swamp," after ZIPs that are also Desert-Swamp Overlap, Food Mirage, etc. are broken out). Both numbers are correct, but the report doesn't currently explain the difference; consider a one-line note in the generated report or a renamed column.
- **`mrfei` and `mrfei_wic` flag identical ZIPs (442, 73.9%) in `swamp_method_comparison.csv`.** The premise for using three independent swamp methods is that they shouldn't share a common blind spot. Worth checking how `mrfei_wic` is computed in `02b_merge_sources.py`/`03_features.py` to confirm it's actually using WIC-vendor data as the healthy-retailer count rather than falling back to the same value as plain `mrfei`.

---

## External / Live Data Sources

Pulled automatically by `01_load_data.py` at runtime — no manual download needed:

| Source | Dataset | Auth Required |
|---|---|---|
| OSM Overpass API | Food-related point locations (supermarkets, convenience stores, fast food, etc.) | No |
| Census TIGER | NJ county boundary shapefiles | No |
| Census ACS 5-Year API | ZCTA-level demographic estimates | Optional (recommended) |
| CDC PLACES (Socrata) | ZIP-level health outcome data | No |
| NJ DOH | WIC-authorized vendor PDF | No |

OSM results are cached locally to `data/osm_data.json` after the first run to avoid re-querying Overpass on subsequent runs. The initial Overpass call can take 30–120 seconds; subsequent runs using the cache are much faster.

---

## Data Sources

| File | Description | Source |
|---|---|---|
| `nj_zip_complete.csv` | ZIP → municipality/census tract crosswalk | Self-generated |
| `nj_zip_crosswalk.csv` | Valid NJ ZIP list for filtering border ZCTAs | Self-generated |
| `zcta_nj.gpkg` | NJ ZCTA boundary geometries | Census TIGER |
| `FoodAccessResearchAtlasData2019.xlsx` | USDA Food Access Research Atlas, 2019 (tract-level) | USDA ERS |
| `ZIP_TRACT_122025.xlsx` | ZIP-to-census-tract crosswalk, Dec 2025 | HUD USPS Crosswalk |
| `snap_retailer_location_data.csv` | SNAP-authorized retailer locations | USDA FNS |
| `food-security-product-deck.-march-2024.pdf` | NJEDA Food Security reference deck, March 2024 | NJEDA |

---
## Outputs

The pipeline generates:

| Output | Description |
|---|---|
| nj_zip_features_v5.csv | Final ZIP-level feature matrix |
| nj_zip_scores.csv | Model predictions and vulnerability scores |
| county_summary.csv | County-level food access summaries |
| municipality_summary.csv | Municipality-level summaries |
| access_typology_profiles.csv | Food access classification profiles |
| plots/ | EDA and model visualizations |
| reports/ | Analytical summaries |

---
