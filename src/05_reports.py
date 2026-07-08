from pathlib import Path
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# ═════════════════════════════════════════════════════════════════════════════
# PATHS
# ═════════════════════════════════════════════════════════════════════════════

ROOT_DIR  = Path(__file__).resolve().parent.parent
DATA     = ROOT_DIR / "data"
REPORTS  = ROOT_DIR / "reports"
REPORTS.mkdir(exist_ok=True)

FEATURES_FILE  = DATA / "nj_zip_features_v5.csv"   # full feature set  ← primary

SCORES_FILE    = DATA / "nj_zip_scores_1.csv"       # model predictions ← merged in
FI_FILE        = DATA / "model_feature_importance.csv"
META_FILE      = DATA / "pipeline_metadata.json"
BOOT_FILE      = DATA / "bootstrap_metrics.csv"
THRESH_FILE    = DATA / "threshold_tuning.csv"
SCV_FILE       = DATA / "spatial_cv_results.csv"

# ═════════════════════════════════════════════════════════════════════════════
# LOAD  — base features from v5, model predictions from scores_1
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("  NJ FOOD ACCESS REPORT GENERATOR")
print("═" * 60)

FEATURES_FILE = DATA / "nj_zip_features_v5.csv"
SCORES_FILE   = DATA / "nj_zip_scores_1.csv"
MODEL_FEATURE_IMPORTANCE = DATA / "model_feature_importance.csv"

# Load fully-engineered feature file (has county, municipality, all flags)
df = pd.read_csv(FEATURES_FILE, dtype={"zip": str})
df["zip"] = df["zip"].astype(str).str.zfill(5)
print(f"\n  Features : {len(df):,} rows × {len(df.columns):,} columns  ({FEATURES_FILE.name})")

# Load model scores and pull in only the predicted/model columns
if SCORES_FILE.exists():
    scores = pd.read_csv(SCORES_FILE, dtype={"zip": str})
    scores["zip"] = scores["zip"].astype(str).str.zfill(5)

    pred_cols = [c for c in scores.columns if c.startswith("predicted_") or c.startswith("typo_prob_")]
    model_meta_cols = ["desert_probability", "predicted_desert", "predicted_desert_tuned", "predicted_typology"]
    model_meta_cols = [c for c in model_meta_cols if c in scores.columns]

    cols_to_merge = ["zip"] + model_meta_cols + pred_cols
    cols_to_merge = [c for c in cols_to_merge if c in scores.columns]

    # Only bring in columns that don't already exist in df
    new_cols = [c for c in cols_to_merge if c not in df.columns or c == "zip"]
    df = df.merge(scores[new_cols], on="zip", how="left")

    print(f"  Scores   : merged {len(new_cols)-1} model columns from {SCORES_FILE.name}")
else:
    print(f"  Scores   : {SCORES_FILE.name} not found — model prediction sections will be skipped")

print(f"  Combined : {len(df):,} rows × {len(df.columns):,} columns")

# ═════════════════════════════════════════════════════════════════════════════
# GUARDS — fail fast with a clear message if key columns are missing
# ═════════════════════════════════════════════════════════════════════════════
REQUIRED = [
    "swamp_score_continuous", "composite_vuln_index",
    "county", "municipality",
    "supermarket", "fast_food", "convenience", "dollar_store",
    "mrfei", "rfei", "access_typology",
    "pct_poverty", "pct_no_vehicle", "pct_elderly",
    "nearest_supermarket_miles",
]
missing = [c for c in REQUIRED if c not in df.columns]
if missing:
    raise ValueError(
        f"\n  ✗ Input file is missing required columns:\n    {missing}"
        f"\n\n  Make sure INPUT_FILE points to the fully-processed v5 feature file."
    )

print("  Columns : all required columns present ✓")

# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════
STORE_TYPES = ["supermarket", "fast_food", "convenience", "dollar_store",
               "restaurant", "produce_market", "snap_stores", "wic_stores"]
STORE_TYPES = [c for c in STORE_TYPES if c in df.columns]

HEALTH_COLS = [c for c in df.columns if "%" in c and "predicted" not in c.lower()]

TYPOLOGY_ORDER = [
    "True Desert", "Dollar Store Desert", "Food Swamp",
    "Food Mirage", "Transit Desert", "Adequate Access",
]

def pct(series):
    """Return percentage of True/1 values as a rounded float."""
    return round(series.mean() * 100, 1)

def section(title):
    print(f"\n{'─' * 60}")
    print(f"  {title.upper()}")
    print(f"{'─' * 60}")

def save(frame, name):
    path = REPORTS / name
    frame.to_csv(path, index=False)
    print(f"  → Saved: reports/{name}  ({len(frame):,} rows)")


# ═════════════════════════════════════════════════════════════════════════════
# 1. NJ STATEWIDE SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
section("1. NJ Statewide Summary")

total_zips = len(df)
total_pop  = df["population"].sum() if "population" in df.columns else None

statewide = {
    "total_zips"                   : total_zips,
    "total_population"             : f"{int(total_pop):,}" if total_pop else "N/A",
    "pct_food_desert_usda"         : f"{pct(df['usda_desert_flag'])}%" if "usda_desert_flag" in df.columns else "N/A",
    "pct_food_swamp_consensus"     : f"{pct(df['is_swamp_consensus'])}%",
    "pct_swamp_rfei"               : f"{pct(df['swamp_rfei_flag'])}%" if "swamp_rfei_flag" in df.columns else "N/A",
    "pct_swamp_nj_method"          : f"{pct(df['swamp_nj_flag'])}%" if "swamp_nj_flag" in df.columns else "N/A",
    "pct_dollar_store_desert"      : f"{pct(df['dollar_store_desert'])}%" if "dollar_store_desert" in df.columns else "N/A",
    "pct_dollar_store_dominance"   : f"{pct(df['dollar_store_dominance'])}%" if "dollar_store_dominance" in df.columns else "N/A",
    "median_rfei"                  : round(df["rfei"].median(), 2),
    "median_mrfei"                 : round(df["mrfei"].median(), 1),
    "median_nearest_supermarket_mi": round(df["nearest_supermarket_miles"].median(), 2),
    "pct_poverty_avg"              : f"{round(df['pct_poverty'].mean(), 1)}%",
    "pct_no_vehicle_avg"           : f"{round(df['pct_no_vehicle'].mean(), 1)}%",
    "pct_elderly_avg"              : f"{round(df['pct_elderly'].mean(), 1)}%",
    "avg_composite_vuln_index"     : round(df["composite_vuln_index"].mean(), 1),
}

print()
for k, v in statewide.items():
    print(f"  {k:<40} {v}")

# Access typology breakdown
print("\n  Access Typology Distribution (% of ZIPs):")
typology_counts = df["access_typology"].value_counts()
for t in TYPOLOGY_ORDER:
    n = typology_counts.get(t, 0)
    print(f"    {t:<25} {n:>4} ZIPs  ({n/total_zips*100:.1f}%)")

statewide_df = pd.DataFrame(list(statewide.items()), columns=["metric", "value"])
save(statewide_df, "nj_statewide_summary.csv")


# ═════════════════════════════════════════════════════════════════════════════
# 2. COUNTY SUMMARIES
# ═════════════════════════════════════════════════════════════════════════════
section("2. County Summaries")

county_agg = (
    df.groupby("county", dropna=False)
    .agg(
        zip_count                   = ("zip",                        "nunique"),
        population                  = ("population",                 "sum")    if "population" in df.columns else ("zip", "count"),
        pct_food_swamp              = ("is_swamp_consensus",         "mean"),
        pct_usda_desert             = ("usda_desert_flag",           "mean")   if "usda_desert_flag" in df.columns else ("zip", "count"),
        pct_dollar_store_desert     = ("dollar_store_desert",        "mean")   if "dollar_store_desert" in df.columns else ("zip", "count"),
        pct_dollar_store_dominance  = ("dollar_store_dominance",     "mean")   if "dollar_store_dominance" in df.columns else ("zip", "count"),
        avg_rfei                    = ("rfei",                       "mean"),
        avg_mrfei                   = ("mrfei",                      "mean"),
        avg_nj_swamp_score          = ("nj_swamp_score",             "mean")   if "nj_swamp_score" in df.columns else ("zip", "count"),
        avg_nearest_supermarket_mi  = ("nearest_supermarket_miles",  "mean"),
        avg_supermarkets            = ("supermarket",                "mean"),
        avg_fast_food               = ("fast_food",                  "mean"),
        avg_convenience             = ("convenience",                "mean"),
        avg_dollar_store            = ("dollar_store",               "mean"),
        avg_pct_poverty             = ("pct_poverty",                "mean"),
        avg_pct_no_vehicle          = ("pct_no_vehicle",             "mean"),
        avg_pct_elderly             = ("pct_elderly",                "mean"),
        avg_composite_vuln          = ("composite_vuln_index",       "mean"),
        avg_novehicle_vuln          = ("novehicle_vuln_score",       "mean")   if "novehicle_vuln_score" in df.columns else ("zip", "count"),
        avg_elderly_vuln            = ("elderly_vuln_score",         "mean")   if "elderly_vuln_score" in df.columns else ("zip", "count"),
    )
    .reset_index()
)

# Typology breakdown per county
typo_county = (
    df.groupby(["county", "access_typology"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)
typo_county.columns = [
    f"typology_{c.lower().replace(' ', '_')}" if c != "county" else "county"
    for c in typo_county.columns
]
county_agg = county_agg.merge(typo_county, on="county", how="left")

# Round percentages
pct_cols = [c for c in county_agg.columns if c.startswith("pct_")]
county_agg[pct_cols] = (county_agg[pct_cols] * 100).round(1)

county_agg = county_agg.sort_values("avg_composite_vuln", ascending=False)

print(f"\n  {len(county_agg)} counties ranked by composite vulnerability:\n")
preview_cols = ["county", "zip_count", "avg_composite_vuln", "pct_food_swamp",
                "pct_usda_desert", "avg_mrfei", "avg_nearest_supermarket_mi"]
preview_cols = [c for c in preview_cols if c in county_agg.columns]
print(county_agg[preview_cols].to_string(index=False))
save(county_agg, "county_summary.csv")


# ═════════════════════════════════════════════════════════════════════════════
# 3. MUNICIPALITY SUMMARIES
# ═════════════════════════════════════════════════════════════════════════════
section("3. Municipality Summaries")

muni_agg = (
    df.groupby(["municipality", "county"], dropna=False)
    .agg(
        zip_count                  = ("zip",                       "nunique"),
        pct_food_swamp             = ("is_swamp_consensus",        "mean"),
        pct_usda_desert            = ("usda_desert_flag",          "mean")  if "usda_desert_flag" in df.columns else ("zip", "count"),
        pct_dollar_store_desert    = ("dollar_store_desert",       "mean")  if "dollar_store_desert" in df.columns else ("zip", "count"),
        avg_rfei                   = ("rfei",                      "mean"),
        avg_mrfei                  = ("mrfei",                     "mean"),
        avg_nearest_supermarket_mi = ("nearest_supermarket_miles", "mean"),
        avg_supermarkets           = ("supermarket",               "mean"),
        avg_fast_food              = ("fast_food",                 "mean"),
        avg_convenience            = ("convenience",               "mean"),
        avg_dollar_store           = ("dollar_store",              "mean"),
        avg_pct_poverty            = ("pct_poverty",               "mean"),
        avg_pct_no_vehicle         = ("pct_no_vehicle",            "mean"),
        avg_composite_vuln         = ("composite_vuln_index",      "mean"),
    )
    .reset_index()
)

pct_cols = [c for c in muni_agg.columns if c.startswith("pct_")]
muni_agg[pct_cols] = (muni_agg[pct_cols] * 100).round(1)
muni_agg = muni_agg.sort_values("avg_composite_vuln", ascending=False)

print(f"\n  {len(muni_agg)} municipalities — top 20 by vulnerability:\n")
print(muni_agg.head(20)[["municipality", "county", "avg_composite_vuln",
                           "pct_food_swamp", "avg_mrfei"]].to_string(index=False))
save(muni_agg, "municipality_summary.csv")


# ═════════════════════════════════════════════════════════════════════════════
# 4. ZIP CODE SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
section("4. ZIP Code Summary")

zip_cols = [
    "zip", "county", "municipality",
    "population", "pop_density",
    "access_typology",
    "composite_vuln_index", "novehicle_vuln_score", "elderly_vuln_score",
    "swamp_score_continuous", "swamp_method_count",
    "rfei", "mrfei",
    "nj_swamp_score",
    "nearest_supermarket_miles",
    "nearest_fastfood_miles", "nearest_convenience_miles",
    "supermarket", "fast_food", "convenience", "dollar_store",
    "produce_market", "wic_stores", "snap_stores",
    "pct_poverty", "pct_no_vehicle", "pct_elderly", "pct_snap",
    "is_swamp_consensus", "usda_desert_flag",
    "dollar_store_desert", "dollar_store_dominance",
    "swamp_rfei_flag", "swamp_mrfei_flag", "swamp_nj_flag",
]
zip_cols = [c for c in zip_cols if c in df.columns]
zip_summary = df[zip_cols].sort_values("composite_vuln_index", ascending=False)

print(f"\n  Top 25 most vulnerable ZIPs:\n")
print(zip_summary.head(25)[["zip", "county", "access_typology",
                              "composite_vuln_index", "mrfei",
                              "nearest_supermarket_miles", "pct_poverty"]].to_string(index=False))
save(zip_summary, "zip_summary.csv")


# ═════════════════════════════════════════════════════════════════════════════
# 5. CENSUS TRACT SUMMARY  (if tract-level columns exist)
# ═════════════════════════════════════════════════════════════════════════════
section("5. Census Tract Summary")

tract_col = next((c for c in df.columns if "tract" in c.lower()), None)

if tract_col:
    tract_agg = (
        df.groupby([tract_col, "county"], dropna=False)
        .agg(
            zip_count              = ("zip",                       "nunique"),
            pct_food_swamp         = ("is_swamp_consensus",        "mean"),
            avg_mrfei              = ("mrfei",                     "mean"),
            avg_rfei               = ("rfei",                      "mean"),
            avg_pct_poverty        = ("pct_poverty",               "mean"),
            avg_pct_no_vehicle     = ("pct_no_vehicle",            "mean"),
            avg_composite_vuln     = ("composite_vuln_index",      "mean"),
            avg_nearest_super_mi   = ("nearest_supermarket_miles", "mean"),
        )
        .reset_index()
        .sort_values("avg_composite_vuln", ascending=False)
    )
    pct_cols = [c for c in tract_agg.columns if c.startswith("pct_")]
    tract_agg[pct_cols] = (tract_agg[pct_cols] * 100).round(1)

    print(f"\n  {len(tract_agg)} census tracts — top 20 by vulnerability:\n")
    print(tract_agg.head(20).to_string(index=False))
    save(tract_agg, "census_tract_summary.csv")
else:
    print("\n  No census tract column detected — skipping.")
    print("  (Add a 'tract' or 'census_tract' column to your input to enable this.)")


# ═════════════════════════════════════════════════════════════════════════════
# 6. ACCESS TYPOLOGY PROFILES
#    Who lives in each typology? Summary of demographics + store mix.
# ═════════════════════════════════════════════════════════════════════════════
section("6. Access Typology Profiles")

typo_profile = (
    df.groupby("access_typology")
    .agg(
        zip_count              = ("zip",                       "nunique"),
        population             = ("population",                "sum")  if "population" in df.columns else ("zip", "count"),
        avg_pct_poverty        = ("pct_poverty",               "mean"),
        avg_pct_no_vehicle     = ("pct_no_vehicle",            "mean"),
        avg_pct_elderly        = ("pct_elderly",               "mean"),
        avg_pct_snap           = ("pct_snap",                  "mean") if "pct_snap" in df.columns else ("zip", "count"),
        avg_rfei               = ("rfei",                      "mean"),
        avg_mrfei              = ("mrfei",                     "mean"),
        avg_nearest_super_mi   = ("nearest_supermarket_miles", "mean"),
        avg_supermarkets       = ("supermarket",               "mean"),
        avg_fast_food          = ("fast_food",                 "mean"),
        avg_convenience        = ("convenience",               "mean"),
        avg_dollar_store       = ("dollar_store",              "mean"),
        avg_composite_vuln     = ("composite_vuln_index",      "mean"),
    )
    .reset_index()
)

# Add % of total ZIPs
typo_profile["pct_of_all_zips"] = (typo_profile["zip_count"] / total_zips * 100).round(1)

# Reorder rows by defined typology order
typo_profile["_sort"] = typo_profile["access_typology"].map(
    {t: i for i, t in enumerate(TYPOLOGY_ORDER)}
)
typo_profile = typo_profile.sort_values("_sort").drop(columns="_sort")

print(f"\n  Profile per access typology:\n")
print(typo_profile.to_string(index=False))
save(typo_profile, "access_typology_profiles.csv")


# ═════════════════════════════════════════════════════════════════════════════
# 7. STORE TYPE DISTRIBUTION
#    What percentage of ZIPs have ≥1 of each store type?
#    Average counts per ZIP, per county, NJ total.
# ═════════════════════════════════════════════════════════════════════════════
section("7. Store Type Distribution")

store_dist_rows = []
for col in STORE_TYPES:
    store_dist_rows.append({
        "store_type"           : col,
        "total_outlets_nj"     : int(df[col].sum()),
        "avg_per_zip"          : round(df[col].mean(), 2),
        "median_per_zip"       : round(df[col].median(), 2),
        "pct_zips_with_any"    : round((df[col] > 0).mean() * 100, 1),
        "pct_zips_with_3plus"  : round((df[col] >= 3).mean() * 100, 1),
        "max_in_single_zip"    : int(df[col].max()),
    })

store_dist = pd.DataFrame(store_dist_rows).sort_values("total_outlets_nj", ascending=False)

print(f"\n  Store type distribution across {total_zips} NJ ZIP codes:\n")
print(store_dist.to_string(index=False))
save(store_dist, "store_type_distribution.csv")

# Per-county store type totals
store_county = (
    df.groupby("county")[STORE_TYPES]
    .sum()
    .reset_index()
    .sort_values("supermarket", ascending=False)
)
save(store_county, "store_type_by_county.csv")

# Per-municipality store type totals
store_muni = (
    df.groupby(["municipality", "county"])[STORE_TYPES]
    .sum()
    .reset_index()
    .sort_values("supermarket", ascending=False)
)
save(store_muni, "store_type_by_municipality.csv")


# ═════════════════════════════════════════════════════════════════════════════
# 8. SWAMP METHOD COMPARISON SUMMARY
#    Agreement / disagreement across the four swamp detection methods
# ═════════════════════════════════════════════════════════════════════════════
section("8. Swamp Method Comparison")

swamp_flags = ["swamp_rfei_flag", "swamp_mrfei_flag", "swamp_mrfei_wic_flag", "swamp_nj_flag"]
swamp_flags = [c for c in swamp_flags if c in df.columns]

swamp_method_rows = []
for col in swamp_flags:
    n = int(df[col].sum())
    swamp_method_rows.append({
        "method"        : col.replace("swamp_", "").replace("_flag", "").upper(),
        "zips_flagged"  : n,
        "pct_of_all"    : round(n / total_zips * 100, 1),
    })

swamp_method_df = pd.DataFrame(swamp_method_rows)

# Consensus breakdown
consensus_breakdown = (
    df["swamp_method_count"]
    .value_counts()
    .sort_index()
    .reset_index()
)
consensus_breakdown.columns = ["methods_agreeing", "zip_count"]
consensus_breakdown["pct_of_all_zips"] = (
    consensus_breakdown["zip_count"] / total_zips * 100
).round(1)

print(f"\n  Swamp flags by method:\n")
print(swamp_method_df.to_string(index=False))
print(f"\n  Consensus (how many methods agree per ZIP):\n")
print(consensus_breakdown.to_string(index=False))
print(f"\n  Consensus swamp ZIPs (≥2 methods): "
      f"{int(df['is_swamp_consensus'].sum())} "
      f"({pct(df['is_swamp_consensus'])}%)")

save(swamp_method_df,      "swamp_method_summary.csv")
save(consensus_breakdown,  "swamp_consensus_breakdown.csv")


# ═════════════════════════════════════════════════════════════════════════════
# 9. DESERT / SWAMP OVERLAP MATRIX
#    How many ZIPs are simultaneously desert AND swamp?
# ═════════════════════════════════════════════════════════════════════════════
section("9. Desert × Swamp Overlap")

if "usda_desert_flag" in df.columns:
    overlap = pd.crosstab(
        df["usda_desert_flag"].map({0: "Not Desert", 1: "USDA Desert"}),
        df["is_swamp_consensus"].map({0: "Not Swamp",  1: "Consensus Swamp"}),
        margins=True
    )
    overlap.index.name   = None
    overlap.columns.name = None
    print(f"\n  ZIP count by desert/swamp status:\n")
    print(overlap.to_string())
    overlap.reset_index().to_csv(REPORTS / "desert_swamp_overlap.csv", index=False)
    print(f"\n  → Saved: reports/desert_swamp_overlap.csv")
else:
    print("\n  usda_desert_flag not found — skipping overlap matrix.")


# ═════════════════════════════════════════════════════════════════════════════
# 10. VULNERABILITY QUINTILE PROFILES
#     Divide ZIPs into 5 equal buckets by composite_vuln_index,
#     show what each quintile looks like demographically + store access.
# ═════════════════════════════════════════════════════════════════════════════
section("10. Vulnerability Quintile Profiles")

df["vuln_quintile"] = pd.qcut(
    df["composite_vuln_index"],
    q=5,
    labels=["Q1 Least Vulnerable", "Q2", "Q3", "Q4", "Q5 Most Vulnerable"]
)

quintile_profile = (
    df.groupby("vuln_quintile", observed=True)
    .agg(
        zip_count              = ("zip",                       "nunique"),
        avg_pct_poverty        = ("pct_poverty",               "mean"),
        avg_pct_no_vehicle     = ("pct_no_vehicle",            "mean"),
        avg_pct_elderly        = ("pct_elderly",               "mean"),
        avg_nearest_super_mi   = ("nearest_supermarket_miles", "mean"),
        avg_rfei               = ("rfei",                      "mean"),
        avg_mrfei              = ("mrfei",                     "mean"),
        avg_supermarkets       = ("supermarket",               "mean"),
        avg_fast_food          = ("fast_food",                 "mean"),
        avg_dollar_store       = ("dollar_store",              "mean"),
        pct_swamp              = ("is_swamp_consensus",        "mean"),
        pct_usda_desert        = ("usda_desert_flag",          "mean") if "usda_desert_flag" in df.columns else ("zip", "count"),
    )
    .reset_index()
)
quintile_profile["pct_swamp"]       = (quintile_profile["pct_swamp"] * 100).round(1)
if "pct_usda_desert" in quintile_profile.columns:
    quintile_profile["pct_usda_desert"] = (quintile_profile["pct_usda_desert"] * 100).round(1)

print(f"\n  Vulnerability quintile profiles:\n")
print(quintile_profile.to_string(index=False))
save(quintile_profile, "vulnerability_quintile_profiles.csv")


# ═════════════════════════════════════════════════════════════════════════════
# 11. TOP / BOTTOM ZIP RANKINGS
# ═════════════════════════════════════════════════════════════════════════════
section("11. ZIP Rankings")

rank_cols = ["zip", "county", "municipality", "access_typology",
             "composite_vuln_index", "mrfei", "rfei",
             "nearest_supermarket_miles", "pct_poverty",
             "pct_no_vehicle", "supermarket", "fast_food", "dollar_store"]
rank_cols = [c for c in rank_cols if c in df.columns]

top25    = df.sort_values("composite_vuln_index", ascending=False).head(25)[rank_cols]
bottom25 = df.sort_values("composite_vuln_index", ascending=True).head(25)[rank_cols]

print(f"\n  Top 25 — Most Vulnerable ZIPs:\n")
print(top25.to_string(index=False))
print(f"\n  Bottom 25 — Least Vulnerable ZIPs:\n")
print(bottom25.to_string(index=False))

save(top25,    "top25_most_vulnerable_zips.csv")
save(bottom25, "top25_least_vulnerable_zips.csv")

# Worst RFEI (most swamp-like by ratio)
worst_rfei = (
    df[df["rfei"].notna()]
    .sort_values("rfei", ascending=False)
    .head(25)[["zip", "county", "municipality", "rfei", "fast_food",
               "convenience", "supermarket", "access_typology"]]
)
print(f"\n  Top 25 — Worst RFEI (food swamp ratio):\n")
print(worst_rfei.to_string(index=False))
save(worst_rfei, "top25_worst_rfei.csv")

# Worst distance to supermarket
worst_dist = (
    df.sort_values("nearest_supermarket_miles", ascending=False)
    .head(25)[["zip", "county", "municipality", "nearest_supermarket_miles",
               "supermarket", "pct_no_vehicle", "access_typology"]]
)
print(f"\n  Top 25 — Furthest from Nearest Supermarket:\n")
print(worst_dist.to_string(index=False))
save(worst_dist, "top25_furthest_from_supermarket.csv")


# ═════════════════════════════════════════════════════════════════════════════
# 12. SNAP / WIC ACCESS SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
section("12. SNAP / WIC Access")

snap_wic_cols = ["snap_stores", "snap_supermarkets", "wic_stores"]
snap_wic_cols = [c for c in snap_wic_cols if c in df.columns]

if snap_wic_cols:
    snap_summary = []
    for col in snap_wic_cols:
        snap_summary.append({
            "program"            : col,
            "total_outlets_nj"   : int(df[col].sum()),
            "pct_zips_with_any"  : round((df[col] > 0).mean() * 100, 1),
            "avg_per_zip"        : round(df[col].mean(), 2),
        })
    snap_df = pd.DataFrame(snap_summary)
    print(f"\n  SNAP / WIC outlet summary:\n")
    print(snap_df.to_string(index=False))

    snap_county = (
        df.groupby("county")[snap_wic_cols]
        .sum()
        .reset_index()
        .sort_values(snap_wic_cols[0], ascending=False)
    )
    print(f"\n  SNAP / WIC by county:\n")
    print(snap_county.to_string(index=False))
    save(snap_df,    "snap_wic_summary.csv")
    save(snap_county,"snap_wic_by_county.csv")
else:
    print("\n  No SNAP/WIC columns found — skipping.")


# ═════════════════════════════════════════════════════════════════════════════
# 13. RACE / ETHNICITY EXPOSURE  (if ACS columns present)
# ═════════════════════════════════════════════════════════════════════════════
section("13. Race / Ethnicity × Food Environment")

race_cols = {
    "pct_black" : "Population Black or African American",
    "pct_hisp"  : "Population Hispanic or Latino",
    "pct_white" : "Population Non-Hispanic White",
    "pct_asian" : "Population Asian Alone",
}

race_available = {
    k: v for k, v in race_cols.items()
    if v in df.columns and "Population Race/Ethnicity Universe" in df.columns
}

if race_available:
    pop_universe = df["Population Race/Ethnicity Universe"].replace(0, np.nan)
    for short, col in race_available.items():
        df[short] = df[col] / pop_universe

    # Quartile by % Black and % Hispanic — compare food environment metrics
    race_env_rows = []
    for short in race_available:
        q75 = df[short].quantile(0.75)
        high = df[df[short] >= q75]
        low  = df[df[short] <  q75]
        race_env_rows.append({
            "group"                    : short,
            "high_group_avg_mrfei"     : round(high["mrfei"].mean(), 1),
            "low_group_avg_mrfei"      : round(low["mrfei"].mean(), 1),
            "high_group_pct_swamp"     : round(high["is_swamp_consensus"].mean() * 100, 1),
            "low_group_pct_swamp"      : round(low["is_swamp_consensus"].mean() * 100, 1),
            "high_group_avg_rfei"      : round(high["rfei"].mean(), 2),
            "low_group_avg_rfei"       : round(low["rfei"].mean(), 2),
            "high_group_avg_super_mi"  : round(high["nearest_supermarket_miles"].mean(), 2),
            "low_group_avg_super_mi"   : round(low["nearest_supermarket_miles"].mean(), 2),
        })

    race_env = pd.DataFrame(race_env_rows)
    print(f"\n  Food environment by high (top quartile) vs low concentration of each group:\n")
    print(race_env.to_string(index=False))
    save(race_env, "race_food_environment.csv")
else:
    print("\n  No race/ethnicity ACS columns detected — skipping.")

# ═════════════════════════════════════════════════════════════════════════════
# 14. MODEL DIAGNOSTICS SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
section("14. Model Diagnostics")

# --- Feature Importance ---
if FI_FILE.exists():
    fi = pd.read_csv(FI_FILE)
    print(f"\n  Top 20 features by importance:\n")
    print(fi.head(20).to_string(index=False))
    # already saved by model pipeline, just display
else:
    print("\n  model_feature_importance.csv not found — skipping.")

# --- Bootstrap Metrics ---
if BOOT_FILE.exists():
    boot = pd.read_csv(BOOT_FILE)
    print(f"\n  Bootstrap CV metrics (already summarized — per metric):\n")

    # bootstrap_metrics.csv IS the summary (one row per metric, mean/ci_lo/ci_hi).
    # Do NOT re-aggregate across rows — that blends unrelated metrics
    # (AUC, Accuracy, Recall[Desert], etc.) into a meaningless composite.
    print(boot.to_string(index=False))
    save(boot, "bootstrap_metrics_summary.csv")
else:
    print("\n  bootstrap_metrics.csv not found — skipping.")

# --- Threshold Tuning ---
if THRESH_FILE.exists():
    thresh = pd.read_csv(THRESH_FILE)
    print(f"\n  Threshold tuning results:\n")
    print(thresh.to_string(index=False))
    # highlight the selected threshold if there's a flag column
    best_col = next((c for c in thresh.columns if "best" in c.lower() or "select" in c.lower()), None)
    if best_col:
        best_row = thresh[thresh[best_col] == 1]
        if not best_row.empty:
            print(f"\n  Selected threshold: {best_row.iloc[0].to_dict()}")
else:
    print("\n  threshold_tuning.csv not found — skipping.")

# --- Spatial CV ---
if SCV_FILE.exists():
    scv = pd.read_csv(SCV_FILE)
    print(f"\n  Spatial CV results ({len(scv)} folds):\n")
    print(scv.to_string(index=False))

    if len(scv) > 1:
        # Exclude 'county' — it's a FIPS identifier, not a quantity to average.
        # Only summarize genuine metrics.
        metric_cols = [c for c in ["n", "deserts", "auc"] if c in scv.columns]
        scv_summary = pd.DataFrame({
            "metric": metric_cols,
            "mean"  : scv[metric_cols].mean().round(4).values,
            "std"   : scv[metric_cols].std().round(4).values,
        })
        print(f"\n  Spatial CV per-fold summary (unweighted across {len(scv)} folds):\n")
        print(scv_summary.to_string(index=False))
        print(f"\n  NOTE: 'auc' above is the unweighted mean of per-fold AUCs and will "
              f"differ from the pooled spatial_cv_auc in pipeline_metadata.json — "
              f"folds vary widely in size (n={scv['n'].min()}–{scv['n'].max()}), so the "
              f"pooled metadata figure is the more defensible headline number.")
        save(scv_summary, "spatial_cv_summary.csv")
else:
    print("\n  spatial_cv_results.csv not found — skipping.")

# --- Pipeline Metadata ---
if META_FILE.exists():
    import json
    with open(META_FILE) as f:
        meta = json.load(f)
    print(f"\n  Pipeline metadata:\n")
    for k, v in meta.items():
        print(f"    {k:<35} {v}")
    meta_df = pd.DataFrame(list(meta.items()), columns=["key", "value"])
    save(meta_df, "pipeline_metadata_report.csv")
else:
    print("\n  pipeline_metadata.json not found — skipping.")

    # ═════════════════════════════════════════════════════════════════════════════
    # 15. PREDICTED OUTCOMES SUMMARY
    # ═════════════════════════════════════════════════════════════════════════════
    section("15. Predicted Outcomes Summary")

    pred_cols = [c for c in df.columns if c.startswith("predicted_")]

    if pred_cols:
        # Split into numeric vs categorical predicted columns
        pred_numeric = [c for c in pred_cols if pd.api.types.is_numeric_dtype(df[c])]
        pred_string = [c for c in pred_cols if not pd.api.types.is_numeric_dtype(df[c])]

        # --- Numeric predicted outcomes by county ---
        if pred_numeric:
            pred_summary = (
                df[["county"] + pred_numeric]
                .groupby("county")
                .mean()
                .round(2)
                .reset_index()
                .sort_values(pred_numeric[0], ascending=False)
            )
            print(f"\n  Mean predicted outcomes by county ({len(pred_numeric)} numeric outcomes):\n")
            print(pred_summary.to_string(index=False))
            save(pred_summary, "predicted_outcomes_by_county.csv")

            # NJ-wide averages
            pred_nj = df[pred_numeric].mean().round(2).reset_index()
            pred_nj.columns = ["outcome", "predicted_mean"]
            pred_nj = pred_nj.sort_values("predicted_mean", ascending=False)
            print(f"\n  NJ-wide predicted outcome averages:\n")
            print(pred_nj.to_string(index=False))
            save(pred_nj, "predicted_outcomes_nj.csv")

        # --- Categorical predicted columns — show value counts instead ---
        if pred_string:
            print(f"\n  Categorical predicted columns — value distributions:\n")
            cat_rows = []
            for col in pred_string:
                counts = df[col].value_counts(dropna=False)
                for label, n in counts.items():
                    cat_rows.append({
                        "column": col,
                        "value": label,
                        "zip_count": n,
                        "pct": round(n / total_zips * 100, 1),
                    })
            cat_df = pd.DataFrame(cat_rows)
            print(cat_df.to_string(index=False))
            save(cat_df, "predicted_categorical_distributions.csv")

        # --- Typology disagreement: model vs rule-based ---
        if "predicted_typology" in df.columns and "access_typology" in df.columns:
            disagreement = df[
                df["predicted_typology"] != df["access_typology"]
                ][["zip", "county", "access_typology", "predicted_typology",
                   "composite_vuln_index", "mrfei"]].copy()
            disagreement = disagreement.sort_values("composite_vuln_index", ascending=False)
            print(f"\n  ZIPs where model typology ≠ rule-based typology: {len(disagreement)}\n")
            print(disagreement.head(20).to_string(index=False))
            save(disagreement, "typology_disagreement_zips.csv")

        # --- typo_prob columns — show NJ averages and top uncertain ZIPs ---
        prob_cols = [c for c in df.columns if c.startswith("typo_prob_")]
        if prob_cols:
            prob_nj = df[prob_cols].mean().round(3).reset_index()
            prob_nj.columns = ["typology", "avg_predicted_probability"]
            prob_nj["typology"] = prob_nj["typology"].str.replace("typo_prob_", "", regex=False)
            prob_nj = prob_nj.sort_values("avg_predicted_probability", ascending=False)
            print(f"\n  NJ-wide average predicted typology probabilities:\n")
            print(prob_nj.to_string(index=False))
            save(prob_nj, "predicted_typology_probabilities_nj.csv")

            # ZIPs with highest uncertainty (max prob closest to 0.5 = least confident)
            df["_max_prob"] = df[prob_cols].max(axis=1)
            uncertain = (
                df.sort_values("_max_prob")
                .head(20)[["zip", "county", "access_typology", "predicted_typology", "_max_prob"] + prob_cols]
            )
            print(f"\n  Top 20 most uncertain ZIP classifications (lowest max probability):\n")
            print(uncertain.to_string(index=False))
            save(uncertain.drop(columns="_max_prob"), "predicted_typology_uncertain_zips.csv")
            df.drop(columns="_max_prob", inplace=True)

    else:
        print("\n  No predicted_* columns found in input — skipping.")
# ═════════════════════════════════════════════════════════════════════════════
# DONE
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("  ALL REPORTS COMPLETE")
print(f"  Output folder: {REPORTS}")
print("═" * 60 + "\n")

