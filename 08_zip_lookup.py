# =========================================================
# zip_lookup.py (OVERVIEW + TARGETED ANALYSIS VERSION)
# Interactive ZIP code food access intelligence tool
# =========================================================

from pathlib import Path
import re
import pandas as pd

DATA = Path(r"C:\Users\crook\PROJECTS\food_access\data")

# =========================================================
# LOAD DATA
# =========================================================

print("Loading lookup tables...")

profile = pd.read_csv(DATA / "retail_profile.csv", dtype={"zip": str})

feat = pd.read_csv(DATA / "nj_zip_features_v5.csv", dtype={"zip": str})
feat["zip"] = feat["zip"].str.zfill(5)

scores = pd.read_csv(DATA / "nj_zip_scores_1.csv", dtype={"zip": str})
scores["zip"] = scores["zip"].str.zfill(5)

# Drop columns from scores that already exist in feat (excluding the join key)
duplicate_cols = [col for col in scores.columns if col in feat.columns and col != "zip"]
scores_clean = scores.drop(columns=duplicate_cols)

# Merge deduplicated scores into feature table
feat = feat.merge(scores_clean, on="zip", how="left")

print("Ready.\n")


# =========================================================
# DEDUPLICATION
# =========================================================

SOURCE_PRIORITY = {"SNAP": 0, "WIC": 1, "OSM": 2}


def _norm_name(name: str) -> str | None:
    if pd.isna(name):
        return None
    name = name.lower()
    name = re.sub(r"\b\d+\b", "", name)
    name = re.sub(r"[^a-z0-9 ]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def make_store_key(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["store_key"] = (
        df["name"].fillna("").apply(_norm_name).fillna("")
        + " | "
        + df["zip"].fillna("")
    )
    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    df = make_store_key(df)
    df["_src_rank"] = df["source"].map(SOURCE_PRIORITY).fillna(99).astype(int)

    no_key  = df[df["store_key"].str.strip() == " | "]
    has_key = df[df["store_key"].str.strip() != " | "]

    has_key = (
        has_key
        .sort_values("_src_rank")
        .drop_duplicates(subset="store_key", keep="first")
    )

    return (
        pd.concat([has_key, no_key], ignore_index=True)
        .drop(columns=["store_key", "_src_rank"])
    )


# =========================================================
# LOOKUP FUNCTION
# =========================================================

def lookup(zip_code: str) -> None:
    zip_code = zip_code.strip().zfill(5)

    # ── Resolve the feature row first so everything below can reference it ──
    risk = feat[feat["zip"] == zip_code]
    if risk.empty:
        print(f"\nZIP {zip_code} not found.\n")
        return

    row = risk.iloc[0]

    # ── Header ──────────────────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  ZIP {zip_code} · {row.get('municipality', '—')} · {row.get('county', '—')}")
    print(f"{'═' * 60}")

    # ── Basic Risk Overview ──────────────────────────────────────────────────
    print("\n── BASIC RISK OVERVIEW ─────────────────────────────")

    typology = row.get("access_typology")          # extracted so policy block can use it

    basic_flags = {
        "Food Desert":         row.get("is_food_desert"),
        "Food Swamp":          row.get("is_swamp_consensus"),
        "Typology":            typology,
        "Vulnerability Index": row.get("composite_vuln_index"),
        "Swamp Score":         row.get("swamp_score_continuous"),
        "mRFEI":               row.get("mrfei"),
        "Poverty %":           row.get("pct_poverty"),
        "No Vehicle %":        row.get("pct_no_vehicle"),
        "Elderly %":           row.get("pct_elderly"),
    }

    for k, v in basic_flags.items():
        if pd.notna(v):
            print(f"  {k:<25} {v:.2f}" if isinstance(v, float) else f"  {k:<25} {v}")

    # ── Model Driver Features ────────────────────────────────────────────────
    print("\n── MODEL DRIVER FEATURES ───────────────────────────")

    driver_features = {
        "Population":                    row.get("population"),
        "Median Income":                 row.get("median_income"),
        "Poverty %":                     row.get("pct_poverty"),
        "SNAP %":                        row.get("pct_snap"),
        "No Vehicle %":                  row.get("pct_no_vehicle"),
        "Elderly %":                     row.get("pct_elderly"),
        "Transit %":                     row.get("pct_transit"),

        "Food Insecurity %":             row.get("Food Insecurity % (Adults)"),
        "Housing Insecurity %":          row.get("Housing Insecurity % (Adults)"),
        "Transportation Barrier %":      row.get("Lack of Transportation % (Adults)"),

        "Any Disability %":              row.get("Any Disability % (Adults)"),
        "Mobility Disability %":         row.get("Mobility Disability % (Adults)"),
        "Physical Inactivity %":         row.get("Physical Inactivity % (Adults)"),
        "Diabetes %":                    row.get("Diabetes % (Adults)"),

        "Nearest Supermarket (mi)":      row.get("nearest_supermarket_miles"),
        "Nearest Fast Food (mi)":        row.get("nearest_fastfood_miles"),

        "Supermarkets":                  row.get("supermarket"),
        "Grocery Stores":                row.get("grocery"),
        "Produce Markets":               row.get("produce_market"),
        "Convenience Stores":            row.get("convenience"),
        "Fast Food":                     row.get("fast_food"),
        "Restaurants":                   row.get("restaurant"),
        "Dollar Stores":                 row.get("dollar_store"),

        "RFEI":                          row.get("rfei"),
        "mRFEI":                         row.get("mrfei"),
        "WIC mRFEI":                     row.get("mrfei_wic"),
        "NJ Swamp Score":                row.get("nj_swamp_score"),

        "Swamp Methods":                 row.get("swamp_method_count"),
        "Swamp Consensus":               row.get("is_swamp_consensus"),
    }

    for k, v in driver_features.items():
        if pd.notna(v):
            print(f"  {k:<30} {v:.2f}" if isinstance(v, (float, int)) else f"  {k:<30} {v}")

    # ── Swamp Method Breakdown ───────────────────────────────────────────────
    print("\n── SWAMP METHOD BREAKDOWN ──────────────────────────")

    methods = [
        ("RFEI",     row.get("rfei"),         "> 3",  row.get("swamp_rfei_flag")),
        ("mRFEI",    row.get("mrfei"),         "< 33", row.get("swamp_mrfei_flag")),
        ("WIC mRFEI",row.get("mrfei_wic"),    "< 33", row.get("swamp_mrfei_wic_flag")),
        ("NJ Score", row.get("nj_swamp_score"),">= 75",row.get("swamp_nj_flag")),
    ]

    for name, value, rule, flag in methods:
        if pd.notna(value):
            status = "SWAMP" if flag == 1 else "OK"
            print(
                f"  {name:<12} "
                f"value={value:>8.2f} "
                f"rule={rule:<6} "
                f"{status}"
            )

    # ── Targeted Analysis Signals ────────────────────────────────────────────
    print("\n── TARGETED ANALYSIS SIGNALS ───────────────────────")

    targeted_flags = {
        "Elderly Vulnerability": row.get("elderly_food_vuln_score"),
        "SNAP/WIC Gap Score":    row.get("snap_wic_combined_gap"),
        "SNAP Gap Flag":         row.get("is_snap_gap_zip"),
        "WIC Gap Flag":          row.get("is_wic_gap_zip"),
        "Combined Gap Flag":     row.get("is_combined_gap_zip"),
        "Desert Probability":    row.get("desert_probability"),
        "Predicted Desert":      row.get("predicted_desert"),
    }

    for k, v in targeted_flags.items():
        if pd.notna(v):
            print(f"  {k:<25} {float(v):.2f}" if isinstance(v, (float, int)) else f"  {k:<25} {v}")

    # ── Policy Interpretation Layer ──────────────────────────────────────────
    print("\n── POLICY FLAGS ────────────────────────────────────")

    pred_desert  = row.get("predicted_desert")
    desert_prob  = row.get("desert_probability", 0) or 0
    food_insec   = row.get("Food Insecurity % (Adults)", 0) or 0
    pct_elderly  = row.get("pct_elderly", 0) or 0

    flagged = False

    if typology == "Food Swamp":
        print("  🚨 RETAIL IMBALANCE: Food Swamp")
        flagged = True

    if pred_desert == 1:
        print(f"  ⚠️  MODEL ALERT: Predicted Food Desert (probability={desert_prob:.2f})")
        flagged = True

    if pct_elderly > 25:
        print("  ⚠️  AGING POPULATION: Elevated elderly concentration")
        flagged = True

    if food_insec > 10:
        print("  ⚠️  FOOD INSECURITY ABOVE STATE NORM")
        flagged = True

    if not flagged:
        print("  ✅  No major policy flags triggered.")

    # ── Food Establishments ──────────────────────────────────────────────────
    pois = profile[profile["zip"] == zip_code]

    if not pois.empty:
        pois = deduplicate(pois)

        print(f"\n── FOOD ESTABLISHMENTS ({len(pois)} unique) ─────────")

        for source, group in pois.groupby("source"):
            print(f"\n  Source: {source}")
            for t, sub in group.groupby("type"):
                print(f"\n    {t}")
                for _, r in sub.iterrows():
                    print(f"      • {r['name']}")

        print("\n── TYPE SUMMARY ────────────────────────────────────")
        print(pois["type"].value_counts().to_string())
    else:
        print("\n  (No retail establishments found for this ZIP.)")


# =========================================================
# INTERACTIVE LOOP
# =========================================================

print("Enter ZIP (q to quit)\n")

while True:
    z = input("ZIP > ").strip()

    if z.lower() in {"q", "quit", "exit"}:
        break

    if not z.isdigit():
        print("Enter a valid ZIP code.")
        continue

    lookup(z)
