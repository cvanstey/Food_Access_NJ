from pathlib import Path
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INPUT = DATA_DIR / "nj_zip_features_v2.csv"

OUTPUT = DATA_DIR / "nj_zip_features_v5.csv"

df = pd.read_csv(INPUT, dtype={"zip": str})
print(df.columns.tolist())
print(f"\n── Loaded: {df.shape[0]} zips × {df.shape[1]} columns")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def safe_div(num, denom, fill=np.nan):
    num   = pd.Series(num,   index=df.index) if not isinstance(num,   pd.Series) else num
    denom = pd.Series(denom, index=df.index) if not isinstance(denom, pd.Series) else denom
    return np.where((denom == 0) | denom.isna(), fill, num / denom)


def pct_rank_norm(series: pd.Series) -> pd.Series:
    return series.rank(pct=True)


def status(msg: str):
    print(f"  {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# Ensure WIC column exists
# ─────────────────────────────────────────────────────────────────────────────
if "wic_stores" not in df.columns:
    df["wic_stores"] = 0
    status("wic_stores missing → set to 0")


# ═════════════════════════════════════════════════════════════════════════════
# 1. RFEI
# ═════════════════════════════════════════════════════════════════════════════
# Standard FSR — matches Cooksey-Stowers (2017)
df["rfei"] = safe_div(
    df["fast_food"] + df["convenience"],
    df["supermarket"] + df["grocery"] + df["produce_market"],
    fill=np.nan
).round(3)

# Extended FSR — adds dollar stores and liquor (once you pull liquor from OSM)
df["rfei_full"] = safe_div(
    df["fast_food"] + df["convenience"] + df["dollar_store"],
    df["supermarket"] + df["grocery"] + df["produce_market"],
    fill=np.nan
).round(3)

df["is_swamp_rfei"] = (df["rfei"] > 3.0).astype(int)


# ═════════════════════════════════════════════════════════════════════════════
# 2. mRFEI
# ═════════════════════════════════════════════════════════════════════════════
healthy = (
    df["supermarket"]
    + df.get("grocery_store", 0)
    + df.get("produce_market", 0)
)

unhealthy = df["fast_food"] + df["convenience"] + df["dollar_store"]
total     = healthy + unhealthy

df["mrfei"] = np.where(total == 0, -1, (healthy / total) * 100).round(2)

df["low_mrfei_share"] = ((df["mrfei"] >= 0) & (df["mrfei"] < 33)).astype(int)


# ═════════════════════════════════════════════════════════════════════════════
# 3. mRFEI WIC
# ═════════════════════════════════════════════════════════════════════════════
total_wic = df["wic_stores"] + df["fast_food"] + df["convenience"] + df["dollar_store"]

df["mrfei_wic"] = np.where(
    total_wic == 0,
    -1,
    (df["wic_stores"] / total_wic) * 100
).round(2)

df["low_mrfei_share_wic"] = ((df["mrfei_wic"] >= 0) & (df["mrfei_wic"] < 33)).astype(int)

df["mrfei_gap"] = (df["mrfei"] - df["mrfei_wic"]).replace([np.inf, -np.inf], np.nan)

# ═════════════════════════════════════════════════════════════════════════════
# 4. NJ Swamp Score  (distance ratio method — matches NJ DCA definition)
#    Formula: nearest swamp outlet / nearest supermarket, scaled 0–100
#    High score = swamp is closer than supermarket = worse access
# ═════════════════════════════════════════════════════════════════════════════
df["nj_swamp_score_raw"] = safe_div(
    df["nearest_fastfood_miles"].clip(lower=0.1),
    df["nearest_supermarket_miles"].clip(lower=0.1),
    fill=np.nan
)

# Invert rank so high score = worse (swamp much closer than supermarket)
df["nj_swamp_score"] = (
    (1 - df["nj_swamp_score_raw"].rank(pct=True)) * 100
).round(1)

df["is_swamp_nj"] = (df["nj_swamp_score"] >= 75).astype(int)

# ═════════════════════════════════════════════════════════════════════════════
# 5. Dollar stores
# ═════════════════════════════════════════════════════════════════════════════
total_retail = (
    df["supermarket"] + df["fast_food"] + df["convenience"]
    + df["dollar_store"] + df["restaurant"]
)

df["dollar_store_ratio"]     = safe_div(df["dollar_store"], total_retail, fill=0).round(3)
df["dollar_store_dominance"] = (df["dollar_store"] > df["supermarket"]).astype(int)
df["dollar_store_desert"]    = ((df["dollar_store"] > 0) & (df["supermarket"] == 0)).astype(int)


# ═════════════════════════════════════════════════════════════════════════════
# 6. Access Typology
# ═════════════════════════════════════════════════════════════════════════════
def classify(row):
    has_super  = row["supermarket"] > 0
    swamp      = row["is_swamp_rfei"] == 1
    far        = row["nearest_supermarket_miles"] >= 1
    poverty    = row["pct_poverty"] > 20
    no_vehicle = row["pct_no_vehicle"] > 10
    dollar_dom = row["dollar_store_dominance"] == 1


    if not has_super and far:
        return "True Desert"
    if has_super and swamp:
        return "Food Swamp"
    if has_super and (poverty or no_vehicle):
        return "Food Mirage"
    if not has_super and dollar_dom:
        return "Dollar Store Desert"

df["access_typology"] = df.apply(classify, axis=1)


# ═════════════════════════════════════════════════════════════════════════════
# 7. Vulnerability scores
# ═════════════════════════════════════════════════════════════════════════════
df["novehicle_vuln_score"] = (
    pct_rank_norm(df["pct_no_vehicle"])                        * 0.35
    + pct_rank_norm(df["nearest_supermarket_miles"])           * 0.30
    + (1 - pct_rank_norm(df["pct_transit"]))                   * 0.20
    + pct_rank_norm(df["rfei"].fillna(0))                      * 0.15
) * 100

df["elderly_vuln_score"] = (
    pct_rank_norm(df["pct_elderly"])                           * 0.30
    + pct_rank_norm(df["pct_no_vehicle"])                      * 0.25
    + pct_rank_norm(df["nearest_supermarket_miles"])           * 0.25
    + pct_rank_norm(df["pct_poverty"])                         * 0.20
) * 100


# ── rfei null handling ────────────────────────────────────────────────────
# NaN means no food retailers at all — treat as worst case, not best
rfei_median = df["rfei"].median()

df["rfei_imputed"] = df["rfei"].fillna(df["rfei"].max())   # no retailers = worst rfei
df["rfei_no_data"] = df["rfei"].isna().astype(int)         # flag for transparency

# ═════════════════════════════════════════════════════════════════════════════
# 8. Composite vulnerability
# ═════════════════════════════════════════════════════════════════════════════
df["composite_vuln_index"] = (
    pct_rank_norm(df["nearest_supermarket_miles"])             * 0.25
    + pct_rank_norm(df["rfei"].fillna(0))                      * 0.25
    + pct_rank_norm(df["pct_poverty"])                         * 0.20
    + pct_rank_norm(df["pct_no_vehicle"])                      * 0.20
    + pct_rank_norm(df["pct_elderly"])                         * 0.10
) * 100


# ═════════════════════════════════════════════════════════════════════════════
# 9. USDA Desert Flag
# ═════════════════════════════════════════════════════════════════════════════
urban = df["pop_density"] >= 1000
rural = ~urban

df["usda_desert_flag"] = (
    (df["pct_poverty"] >= 20)
    & (
        (urban & (df["nearest_supermarket_miles"] >= 1))
        | (rural & (df["nearest_supermarket_miles"] >= 10))
    )
).astype(int)


# ═════════════════════════════════════════════════════════════════════════════
# 10. Swamp flags — all four methods
# ═════════════════════════════════════════════════════════════════════════════
df["swamp_rfei_flag"]     = (df["rfei"] > 3.0).astype(int)
df["swamp_mrfei_flag"]    = ((df["mrfei"] >= 0) & (df["mrfei"] < 33)).astype(int)
df["swamp_mrfei_wic_flag"]= ((df["mrfei_wic"] >= 0) & (df["mrfei_wic"] < 33)).astype(int)
df["swamp_nj_flag"]       = (df["nj_swamp_score"] >= 75).astype(int)

df["swamp_method_count"]    = (
    df["swamp_rfei_flag"]
    + df["swamp_mrfei_flag"]
    + df["swamp_mrfei_wic_flag"]
    + df["swamp_nj_flag"]
)
df["is_swamp_consensus"]    = (df["swamp_method_count"] >= 2).astype(int)
df["swamp_score_continuous"]= (df["swamp_method_count"] / 4).round(3)

print("\n── Swamp flag totals across all ZIPs")
print(df[["swamp_rfei_flag", "swamp_mrfei_flag", "swamp_mrfei_wic_flag", "swamp_nj_flag"]].sum())


# ═════════════════════════════════════════════════════════════════════════════
# 11. Per-ZIP method comparison functions
# ═════════════════════════════════════════════════════════════════════════════
def explain_zip(zip_code: str, df: pd.DataFrame):
    """Print a detailed method-by-method breakdown for a single ZIP."""
    row = df[df["zip"] == zip_code]
    if row.empty:
        print(f"ZIP {zip_code} not found.")
        return

    r = row.iloc[0]

    print("\n══════════════════════════════════════")
    print(f"  ZIP CODE: {zip_code}")
    print("══════════════════════════════════════\n")

    print("📊 SWAMP METHOD COMPARISON\n")
    methods = [
        ("RFEI",      r["rfei"],           "> 3",  r["swamp_rfei_flag"]),
        ("mRFEI",     r["mrfei"],          "< 33", r["swamp_mrfei_flag"]),
        ("WIC mRFEI", r["mrfei_wic"],      "< 33", r["swamp_mrfei_wic_flag"]),
        ("NJ Swamp",  r["nj_swamp_score"], "≥ 75", r["swamp_nj_flag"]),
    ]
    for name, value, rule, flag in methods:
        marker = "🔴 SWAMP" if flag else "🟢 OK"
        print(f"  {name:12} | value = {value:7.2f} | threshold {rule:6} | {marker}")

    print("\n──────────────────────────────────────")
    count = int(r["swamp_method_count"])
    print(f"\n  Methods flagged : {count}/4")

    if count >= 3:
        label = "HIGH CONFIDENCE SWAMP"
    elif count == 2:
        label = "MIXED SWAMP SIGNAL"
    elif count == 1:
        label = "WEAK SWAMP SIGNAL"
    else:
        label = "NOT A SWAMP"

    print(f"  Classification  : {label}")
    print(f"  Access typology : {r['access_typology']}")

    print("\n  Active drivers:")
    drivers = {
        "swamp_rfei_flag":      "High fast food + convenience density (RFEI)",
        "swamp_mrfei_flag":     "Low healthy retail share (mRFEI)",
        "swamp_mrfei_wic_flag": "Weak WIC-certified access",
        "swamp_nj_flag":        "Poor spatial access + high unhealthy density (NJ score)",
    }
    any_driver = False
    for col, desc in drivers.items():
        if r[col]:
            print(f"    • {desc}")
            any_driver = True
    if not any_driver:
        print("    • None")


def compare_zips(zip_codes: list, df: pd.DataFrame):
    """Print a side-by-side method comparison table for multiple ZIPs."""
    rows = []
    for z in zip_codes:
        row = df[df["zip"] == z]
        if row.empty:
            print(f"  ZIP {z} not found — skipping")
            continue
        r = row.iloc[0]
        rows.append({
            "zip":         z,
            "rfei":        round(r["rfei"], 2) if not pd.isna(r["rfei"]) else "N/A",
            "rfei_swamp":  "✓" if r["swamp_rfei_flag"]      else "–",
            "mrfei":       round(r["mrfei"], 1),
            "mrfei_swamp": "✓" if r["swamp_mrfei_flag"]     else "–",
            "mrfei_wic":   round(r["mrfei_wic"], 1),
            "wic_swamp":   "✓" if r["swamp_mrfei_wic_flag"] else "–",
            "nj_score":    round(r["nj_swamp_score"], 1),
            "nj_swamp":    "✓" if r["swamp_nj_flag"]        else "–",
            "n_methods":   int(r["swamp_method_count"]),
            "typology":    r["access_typology"],
        })

    if not rows:
        return

    out = pd.DataFrame(rows).set_index("zip")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 140)
    print("\n── Method-by-method comparison")
    print(out.to_string())



# ── Desert target labels ──────────────────────────────────────────────────
df["is_food_desert"] = df["usda_desert_flag"]  # USDA definition

df["is_desert_5mi"] = (
    (df["pct_poverty"] >= 20)
    & (df["nearest_supermarket_miles"] >= 1)
).astype(int)

# ── Desert target labels ──────────────────────────────────────────────────

# USDA strict: poverty ≥20% + distance threshold (urban 1mi / rural 10mi)
df["is_desert_usda"] = df["usda_desert_flag"]

# FARA LILA: low income AND low access per USDA tract-level definition
# Already aggregated to ZIP in fara_agg — 1 if ANY tract in ZIP is LILA
df["is_desert_fara"] = (df["usda_lila_1_10"].fillna(0) == 1).astype(int)

# Consensus: flagged by both definitions
df["is_desert_consensus"] = (
    (df["is_desert_usda"] == 1) & (df["is_desert_fara"] == 1)
).astype(int)

# Model target — be explicit about which definition drives the classifier
df["is_food_desert"] = df["is_desert_fara"]   # ← change this one line to swap definitions

print("\n── Desert definition comparison")
print(f"  USDA strict (14 expected)      : {df['is_desert_usda'].sum()}")
print(f"  FARA LILA 1/10 (109 expected)  : {df['is_desert_fara'].sum()}")
print(f"  Consensus (both)               : {df['is_desert_consensus'].sum()}")
print(f"  Model target (is_food_desert)  : {df['is_food_desert'].sum()}")

# ── Food mirage ───────────────────────────────────────────────────────────
# Has a supermarket but poverty + no-vehicle make it effectively inaccessible
df["is_food_mirage"] = (
    (df["supermarket"] > 0)
    & (df["pct_poverty"] > 20)
    & (df["pct_no_vehicle"] > 15)
).astype(int)


# Simple renames
df["median_income"] = df["Median Household Income_acs"]
df["population"]    = df["Total Population_acs"]

# Supermarkets within 5mi — placeholder using ZIP count until spatial buffer computed
df["supermarkets_within_5mi"] = df["supermarket"]

degree_cols = [
    "Population Bachelor's Degree",
    "Population Master's Degree",
    "Population Professional Degree",
    "Population Doctorate Degree",
]
df["pct_college"] = (
    df[degree_cols].sum(axis=1) / df["Population Education Universe"] * 100
)
# ═════════════════════════════════════════════════════════════════════════════
# 12. Save
# ═════════════════════════════════════════════════════════════════════════════
print("\n── Saving output")
df.to_csv(OUTPUT, index=False)
print(f"  Saved → {OUTPUT}")
print(f"  Shape → {df.shape}")

# Also export a focused method-comparison CSV for all ZIPs
method_cols = [
    "zip", "rfei", "swamp_rfei_flag",
    "mrfei", "swamp_mrfei_flag",
    "mrfei_wic", "swamp_mrfei_wic_flag",
    "nj_swamp_score", "swamp_nj_flag",
    "swamp_method_count", "access_typology",
]
df[method_cols].to_csv(DATA_DIR / "swamp_method_comparison.csv", index=False)
print(f"  Saved → swamp_method_comparison.csv")


# ═════════════════════════════════════════════════════════════════════════════
# 13. Example usage — edit ZIP codes here
# ═════════════════════════════════════════════════════════════════════════════
explain_zip("07103", df)

compare_zips(["08087", "07201", "08401"], df)

df_check = pd.read_csv(OUTPUT, dtype={"zip": str})

print(f"Total columns: {df_check.shape[1]}")
print("\nAll columns:")

for i, c in enumerate(df_check.columns):
    print(f"{i:3d}  {c}")