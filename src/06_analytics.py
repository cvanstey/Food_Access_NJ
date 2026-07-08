"""
analytics_descriptive.py
=========================
Extended Descriptive Analytics — NJ Food Access Pipeline
Builds on existing feature set to produce deep, publication-quality analysis.

Sections
--------
  1.  Dataset Overview & Data Quality Audit
  2.  Composite Vulnerability Index — Full Profile
  3.  Food Desert Classification Cross-Tabulation
  4.  Food Swamp Multi-Method Agreement Analysis
  5.  Health Outcome Profiles by Food Environment Typology
  6.  Store-Type Access Equity Analysis (by income & race)
  7.  Transportation Barrier Deep-Dive
  8.  County-Level Benchmarking
  9.  Dual-Burden ZIPs — Desert AND Swamp
 10.  Spatial Isolation vs. Socioeconomic Vulnerability Matrix
 11.  Correlation Heatmap — Food Environment × Health Outcomes
 12.  Population-Weighted Summary Statistics
 13.  Racial Equity Analysis — Store Access Disparities
 14.  Access Typology Transition Matrix
 15.  Outlier & Extreme-Case Narratives

Outputs
-------
  plots/  — PNG figures (publication quality, 300 DPI)
  data/   — CSV summary tables

Dependencies
------------
  pip install pandas numpy matplotlib seaborn scipy scikit-learn
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import importlib
_sns_module = importlib.import_module('seaborn')
sns = _sns_module  # avoids alias shadowing by local variables named 'sns'
from scipy import stats
from scipy.stats import kruskal, mannwhitneyu, spearmanr

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────



ROOT_DIR  = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT_DIR / "data"
PLOTS_DIR = ROOT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_FILE = DATA_DIR / "nj_zip_features_v5.csv"
SCORES_FILE   = DATA_DIR / "nj_zip_scores_1.csv"

# Colour palette — consistent across all figures
PALETTE = {
    "desert":      "#C0392B",
    "swamp":       "#E67E22",
    "dual":        "#8E44AD",
    "healthy":     "#27AE60",
    "neutral":     "#2980B9",
    "light_grey":  "#ECF0F1",
    "dark_grey":   "#7F8C8D",
}

TYPOLOGY_ORDER = [
    "Food Desert",
    "Food Swamp",
    "Food Mirage",
    "Rural Thin Access",
    "Adequate Access",
]

plt.rcParams.update({
    "font.family":   "DejaVu Sans",
    "font.size":     11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":    150,
    "savefig.dpi":   300,
    "savefig.bbox":  "tight",
})


# ── Helpers ───────────────────────────────────────────────────────────────────

def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def cohens_d(a: pd.Series, b: pd.Series) -> float:
    """Pooled Cohen's d."""
    a, b = a.dropna(), b.dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled_std = np.sqrt(((len(a) - 1) * a.std() ** 2 + (len(b) - 1) * b.std() ** 2)
                         / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / pooled_std if pooled_std else np.nan


def effect_label(d: float) -> str:
    ad = abs(d)
    if ad < 0.2:  return "negligible"
    if ad < 0.5:  return "small"
    if ad < 0.8:  return "medium"
    return "large"


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna() & (weights > 0)
    return np.average(values[mask], weights=weights[mask])


def pct_fmt(n: float, total: float) -> str:
    return f"{n:,.0f} ({n / total * 100:.1f}%)"


def mannwhitney_stars(a, b):
    """Return significance stars from Mann-Whitney U test."""
    try:
        _, p = mannwhitneyu(a.dropna(), b.dropna(), alternative="two-sided")
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return "ns"
    except Exception:
        return ""


# ── Load Data ─────────────────────────────────────────────────────────────────

print("\nLoading dataset...")
df = pd.read_csv(FEATURES_FILE, dtype={"zip": str})
df["zip"] = df["zip"].str.zfill(5)
print(f"  Features : {len(df)} rows × {len(df.columns)} columns  ({FEATURES_FILE.name})")

if SCORES_FILE.exists():
    scores = pd.read_csv(SCORES_FILE, dtype={"zip": str})
    scores["zip"] = scores["zip"].str.zfill(5)

    pred_cols      = [c for c in scores.columns if c.startswith("predicted_") or c.startswith("typo_prob_")]
    model_cols     = [c for c in ["desert_probability", "predicted_desert",
                                   "predicted_desert_tuned", "predicted_typology"]
                      if c in scores.columns]
    usda_cols      = [c for c in scores.columns if c.startswith("usda_")]

    cols_to_merge  = ["zip"] + model_cols + pred_cols + usda_cols
    cols_to_merge  = [c for c in cols_to_merge if c in scores.columns]
    new_cols       = [c for c in cols_to_merge if c not in df.columns or c == "zip"]

    df = df.merge(scores[new_cols], on="zip", how="left")
    print(f"  Scores   : merged {len(new_cols)-1} model columns from {SCORES_FILE.name}")
else:
    print(f"  Scores   : {SCORES_FILE.name} not found — predicted columns unavailable")

print(f"  Combined : {len(df)} rows × {len(df.columns)} columns")

# Derived convenience flags
if "is_food_desert" not in df.columns:
    df["is_food_desert"] = df.get("desert_type", "").str.contains("Desert", na=False).astype(int)
if "is_swamp_consensus" not in df.columns:
    swamp_cols = [c for c in df.columns if "swamp_flag" in c or c == "swamp_method_count"]
    if "swamp_method_count" in df.columns:
        df["is_swamp_consensus"] = (df["swamp_method_count"] >= 3).astype(int)

df["is_dual_burden"] = (
    df.get("is_food_desert", 0).fillna(0).astype(int) &
    df.get("is_swamp_consensus", 0).fillna(0).astype(int)
).astype(int)

# Composite vulnerability quintile (use existing if present)
if "composite_vuln_index" in df.columns:
    df["vuln_quintile"] = pd.qcut(
        df["composite_vuln_index"], q=5,
        labels=["Very Low", "Low", "Moderate", "High", "Very High"]
    )

# Income tertile
if "median_income" in df.columns:
    df["income_tertile"] = pd.qcut(
        df["median_income"].rank(method="first"), q=3,
        labels=["Low Income", "Middle Income", "High Income"]
    )

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1  Dataset Overview & Data Quality Audit
# ─────────────────────────────────────────────────────────────────────────────

section("SECTION 1 — Dataset Overview & Data Quality Audit")

AUDIT_COLS = [
    "composite_vuln_index", "median_income", "population", "pop_density",
    "pct_poverty", "pct_snap", "pct_no_vehicle", "pct_elderly", "pct_college",
    "supermarket", "fast_food", "convenience", "produce_market",
    "nearest_supermarket_miles", "supermarkets_within_5mi",
    "swamp_score_continuous", "mrfei", "rfei",
    "Diabetes % (Adults)", "Obesity % (Adults)",
    "High Blood Pressure % (Adults)", "Coronary Heart Disease % (Adults)",
    "Depression % (Adults)", "Food Insecurity % (Adults)",
    "Physical Inactivity % (Adults)", "Poor Mental Health % (Adults)",
]
AUDIT_COLS = [c for c in AUDIT_COLS if c in df.columns]

audit = df[AUDIT_COLS].agg(["count", "min", "median", "mean", "max", "std"]).T
audit["pct_null"] = df[AUDIT_COLS].isnull().mean().mul(100).round(1)
audit["skewness"] = df[AUDIT_COLS].skew().round(2)
print(audit.to_string())
audit.to_csv(DATA_DIR / "data_audit_extended.csv")
print("\n  Saved → data/data_audit_extended.csv")

# Missing-value heatmap
null_pct = df.isnull().mean().mul(100).sort_values(ascending=False)
high_null = null_pct[null_pct > 5]
if len(high_null) > 0:
    fig, ax = plt.subplots(figsize=(10, max(4, len(high_null) * 0.3)))
    high_null.plot(kind="barh", color=PALETTE["neutral"], ax=ax)
    ax.set_xlabel("% Missing")
    ax.set_title("Columns with >5% Missing Data", fontweight="bold")
    ax.axvline(20, color=PALETTE["desert"], linestyle="--", lw=1.5, label="20% threshold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "00_missing_data_profile.png")
    plt.close()
    print("  Saved → plots/00_missing_data_profile.png")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2  Composite Vulnerability Index — Full Profile
# ─────────────────────────────────────────────────────────────────────────────

section("SECTION 2 — Composite Vulnerability Index Full Profile")

VULN_DRIVERS = [c for c in [
    "pct_poverty", "pct_no_vehicle", "pct_elderly", "pct_snap",
    "median_income", "pct_college", "pop_density",
    "nearest_supermarket_miles", "swamp_score_continuous",
] if c in df.columns]

if "composite_vuln_index" in df.columns and "vuln_quintile" in df.columns:
    vuln_profile = df.groupby("vuln_quintile", observed=True)[VULN_DRIVERS].mean()
    print("\n  Mean driver values by vulnerability quintile:")
    print(vuln_profile.round(2).to_string())
    vuln_profile.to_csv(DATA_DIR / "vuln_quintile_profiles.csv")

    # Radar-like grouped bar chart
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharey=False)
    axes = axes.flatten()
    colors = ["#27AE60", "#2ECC71", "#F39C12", "#E67E22", "#C0392B"]

    for i, col in enumerate(VULN_DRIVERS[:8]):
        ax = axes[i]
        vals = df.groupby("vuln_quintile", observed=True)[col].mean()
        bars = ax.bar(range(len(vals)), vals.values, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(["VL", "L", "M", "H", "VH"], fontsize=9)
        ax.set_title(col.replace("pct_", "% ").replace("_", " ").title(), fontsize=10)
        ax.set_xlabel("Vulnerability Quintile", fontsize=8)

    if len(axes) > len(VULN_DRIVERS):
        for j in range(len(VULN_DRIVERS), len(axes)):
            axes[j].set_visible(False)

    fig.suptitle("Mean Driver Values by Composite Vulnerability Quintile",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "06_vuln_driver_profiles.png")
    plt.close()
    print("  Saved → plots/06_vuln_driver_profiles.png")


df = df.loc[:, ~df.columns.duplicated()]

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3  Food Desert Classification Cross-Tabulation
# ─────────────────────────────────────────────────────────────────────────────

section("SECTION 3 — Food Desert Classification Cross-Tabulation")

DESERT_FLAGS = [c for c in [
    "is_food_desert", "is_desert_5mi",
    "usda_lila_1_10", "usda_lila_half_10", "usda_lila_1_20",
    "predicted_desert", "predicted_desert_tuned",
] if c in df.columns]

if len(DESERT_FLAGS) >= 2:
    # Binarise all desert flags once (handles floats, NaN, non-0/1 values)
    desert_bin = pd.DataFrame({
        c: pd.to_numeric(df[c], errors='coerce').fillna(0).gt(0).astype(int)
        for c in DESERT_FLAGS
    })

    # Agreement matrix — use numpy scalar to avoid Series ambiguity
    agree_data = {}
    for a in DESERT_FLAGS:
        agree_data[a] = {}
        for b in DESERT_FLAGS:
            val = float(np.mean((desert_bin[a].values == desert_bin[b].values))) * 100
            agree_data[a][b] = round(val, 2)
    agree_matrix = pd.DataFrame(agree_data).T  # rows=a, cols=b

    print("\n  Method agreement matrix (% ZIPs classified identically):")
    print(agree_matrix.round(1).to_string())
    agree_matrix.to_csv(DATA_DIR / "desert_method_agreement.csv")

    fig, ax = plt.subplots(figsize=(9, 7))
    labels = [c.replace("usda_", "USDA ").replace("_", " ").title() for c in DESERT_FLAGS]
    sns.heatmap(
        agree_matrix.astype(float), annot=True, fmt=".1f", cmap="RdYlGn",
        vmin=50, vmax=100, ax=ax,
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, linecolor="white",
    )
    ax.set_title("Desert Classification Agreement Matrix\n(% ZIPs Classified Identically)",
                 fontweight="bold")
    plt.xticks(rotation=35, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "07_desert_method_agreement.png")
    plt.close()
    print("  Saved → plots/07_desert_method_agreement.png")

    # Count table
    desert_counts = pd.DataFrame({
        m: [df[m].fillna(0).astype(int).sum(),
            df[m].fillna(0).astype(int).mean() * 100]
        for m in DESERT_FLAGS
    }, index=["n_flagged", "pct_flagged"]).T
    print("\n  Desert flags by method:")
    print(desert_counts.round(1).to_string())



# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4  Food Swamp Multi-Method Agreement Analysis
# ─────────────────────────────────────────────────────────────────────────────

section("SECTION 4 — Food Swamp Multi-Method Agreement Analysis")

SWAMP_FLAGS = [c for c in [
    "swamp_rfei_flag", "swamp_mrfei_flag", "swamp_mrfei_wic_flag", "swamp_nj_flag",
] if c in df.columns]

if "swamp_method_count" in df.columns:
    smc = df["swamp_method_count"].value_counts().sort_index()
    print("\n  Swamp method agreement counts (0 = no methods agree → 4 = all agree):")
    for k, v in smc.items():
        bar = "█" * int(v / len(df) * 40)
        print(f"    {int(k)} methods: {v:>4} ZIPs ({v/len(df)*100:4.1f}%)  {bar}")

    # Stacked bar by method count
    if len(SWAMP_FLAGS) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: bar of agreement counts
        ax = axes[0]
        colors_bar = ["#27AE60", "#F1C40F", "#E67E22", "#E74C3C", "#8E44AD"]
        smc.plot(kind="bar", ax=ax, color=colors_bar[:len(smc)], edgecolor="white", width=0.7)
        ax.set_xlabel("Number of Methods Flagging ZIP as Swamp")
        ax.set_ylabel("ZIP Count")
        ax.set_title("Swamp Method Agreement\nDistribution", fontweight="bold")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

        # Right: upset-style method co-occurrence
        ax = axes[1]
        combos = df[SWAMP_FLAGS].fillna(0).astype(int)
        combo_counts = combos.groupby(SWAMP_FLAGS).size().reset_index(name="count")
        combo_counts = combo_counts.sort_values("count", ascending=False).head(10)
        labels = [
            "+".join([f.replace("swamp_", "").replace("_flag", "").upper()
                      for f, v in zip(SWAMP_FLAGS, row[SWAMP_FLAGS]) if v == 1])
            or "None"
            for _, row in combo_counts.iterrows()
        ]
        ax.barh(range(len(labels)), combo_counts["count"].values,
                color=PALETTE["swamp"], edgecolor="white")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("ZIP Count")
        ax.set_title("Top 10 Swamp Flag Combinations", fontweight="bold")
        ax.invert_yaxis()

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "08_swamp_method_agreement.png")
        plt.close()
        print("  Saved → plots/08_swamp_method_agreement.png")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5  Health Outcome Profiles by Food Environment Typology
# ─────────────────────────────────────────────────────────────────────────────

section("SECTION 5 — Health Outcome Profiles by Food Environment Typology")

HEALTH_OUTCOMES = [c for c in [
    "Diabetes % (Adults)", "Obesity % (Adults)",
    "High Blood Pressure % (Adults)", "Coronary Heart Disease % (Adults)",
    "Physical Inactivity % (Adults)", "Food Insecurity % (Adults)",
    "Depression % (Adults)", "Poor Mental Health % (Adults)",
    "Current Smoking % (Adults)", "COPD % (Adults)",
    "Stroke % (Adults)", "Any Disability % (Adults)",
] if c in df.columns]

GROUP_COL = "food_access_type" if "food_access_type" in df.columns else \
            "access_typology" if "access_typology" in df.columns else None

if GROUP_COL and len(HEALTH_OUTCOMES) >= 4:
    health_by_type = df.groupby(GROUP_COL)[HEALTH_OUTCOMES].mean().round(2)
    print("\n  Mean health outcomes by food access typology:")
    print(health_by_type.to_string())
    health_by_type.to_csv(DATA_DIR / "health_by_typology.csv")

    # Heatmap (z-scored for comparability)
    health_z = (health_by_type - health_by_type.mean()) / health_by_type.std()
    short_names = [c.replace(" % (Adults)", "").replace(" % (Women)", "") for c in HEALTH_OUTCOMES]

    fig, ax = plt.subplots(figsize=(14, max(5, len(health_by_type) * 0.8 + 2)))
    sns.heatmap(
        health_z.T, annot=health_by_type.T.round(1), fmt=".1f",
        cmap="RdYlGn_r", center=0, ax=ax,
        linewidths=0.4, linecolor="white",
        xticklabels=health_by_type.index.tolist(),
        yticklabels=short_names,
    )
    ax.set_title("Health Outcome Profiles by Food Environment Typology\n"
                 "(Annotated: raw %, color: z-score vs. mean)", fontweight="bold")
    plt.xticks(rotation=25, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "09_health_by_typology_heatmap.png")
    plt.close()
    print("  Saved → plots/09_health_by_typology_heatmap.png")

# Desert vs. non-desert health effect sizes
if "is_food_desert" in df.columns and len(HEALTH_OUTCOMES) >= 2:
    desert_rows   = df[df["is_food_desert"] == 1]
    nodesert_rows = df[df["is_food_desert"] == 0]

    effect_rows = []
    for col in HEALTH_OUTCOMES:
        d  = cohens_d(desert_rows[col], nodesert_rows[col])
        mw = mannwhitney_stars(desert_rows[col], nodesert_rows[col])
        effect_rows.append({
            "outcome": col.replace(" % (Adults)", ""),
            "mean_desert":    desert_rows[col].mean(),
            "mean_nondesert": nodesert_rows[col].mean(),
            "diff":           desert_rows[col].mean() - nodesert_rows[col].mean(),
            "cohens_d":       round(d, 3),
            "effect_label":   effect_label(d),
            "significance":   mw,
        })

    health_effects = pd.DataFrame(effect_rows).sort_values("cohens_d", ascending=False)
    print("\n  Health effect sizes — Desert vs. Non-Desert:")
    print(health_effects.to_string(index=False))
    health_effects.to_csv(DATA_DIR / "health_effect_sizes_desert.csv", index=False)

    # Lollipop chart
    fig, ax = plt.subplots(figsize=(10, max(5, len(health_effects) * 0.55 + 1)))
    colors_lol = [PALETTE["desert"] if d > 0 else PALETTE["healthy"]
                  for d in health_effects["diff"]]
    ax.hlines(range(len(health_effects)), 0, health_effects["diff"].values,
              color=PALETTE["dark_grey"], linewidth=1.2, zorder=1)
    ax.scatter(health_effects["diff"].values, range(len(health_effects)),
               color=colors_lol, s=90, zorder=2)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(range(len(health_effects)))
    ax.set_yticklabels(health_effects["outcome"].values, fontsize=9)
    ax.set_xlabel("Mean Difference (Desert − Non-Desert), percentage points")
    ax.set_title("Health Outcome Gap: Food Desert vs. Non-Desert ZIPs", fontweight="bold")

    for i, (_, row) in enumerate(health_effects.iterrows()):
        if row["significance"] not in ("ns", ""):
            ax.text(row["diff"] + 0.05, i, row["significance"], va="center", fontsize=9,
                    color=PALETTE["desert"])
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "10_health_gap_desert_lollipop.png")
    plt.close()
    print("  Saved → plots/10_health_gap_desert_lollipop.png")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6  Store-Type Access Equity (by income & race)
# ─────────────────────────────────────────────────────────────────────────────

section("SECTION 6 — Store-Type Access Equity by Income & Race")

STORE_TYPES = [c for c in
    ["supermarket", "fast_food", "convenience", "produce_market", "dollar_store"]
    if c in df.columns]

# ── By income tertile ──────────────────────────────────────────────────────
if "income_tertile" in df.columns and STORE_TYPES:
    store_by_income = df.groupby("income_tertile", observed=True)[STORE_TYPES].mean().round(2)
    print("\n  Mean store counts by income tertile:")
    print(store_by_income.to_string())
    store_by_income.to_csv(DATA_DIR / "store_by_income_tertile.csv")

    fig, axes = plt.subplots(1, len(STORE_TYPES), figsize=(16, 5), sharey=False)
    for i, col in enumerate(STORE_TYPES):
        ax = axes[i]
        vals = df.groupby("income_tertile", observed=True)[col].mean()
        bars = ax.bar(range(len(vals)), vals.values,
                      color=["#C0392B", "#F39C12", "#27AE60"], edgecolor="white")
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(["Low", "Mid", "High"], rotation=20, ha="right", fontsize=9)
        ax.set_title(col.replace("_", " ").title(), fontsize=10, fontweight="bold")
        ax.set_ylabel("Mean Count per ZIP" if i == 0 else "")

        # Low vs high stars
        lo = df[df["income_tertile"] == "Low Income"][col]
        hi = df[df["income_tertile"] == "High Income"][col]
        stars = mannwhitney_stars(lo, hi)
        if stars not in ("ns", ""):
            ymax = max(vals.values)
            ax.text(1, ymax * 1.08, stars, ha="center", fontsize=12, color="black")

    fig.suptitle("Mean Store Count by Income Tertile\n(* p<.05  ** p<.01  *** p<.001 Low vs. High)",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "11_store_equity_income.png")
    plt.close()
    print("  Saved → plots/11_store_equity_income.png")

# ── By dominant racial group (majority-minority vs. majority-white) ──────────
if "Population Non-Hispanic White" in df.columns and "population" in df.columns:
    df["pct_white"] = df["Population Non-Hispanic White"] / df["population"].replace(0, np.nan)
    df["majority_group"] = np.where(df["pct_white"] >= 0.60, "Majority White",
                           np.where(df["pct_white"] <= 0.40, "Majority Non-White", "Mixed"))

    if STORE_TYPES:
        store_by_race = df.groupby("majority_group")[STORE_TYPES].mean().round(2)
        print("\n  Mean store counts by racial composition:")
        print(store_by_race.to_string())
        store_by_race.to_csv(DATA_DIR / "store_by_racial_composition.csv")

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(STORE_TYPES))
        groups = store_by_race.index.tolist()
        grp_colors = {"Majority White": "#2980B9", "Mixed": "#F39C12", "Majority Non-White": "#C0392B"}
        width = 0.25

        for j, grp in enumerate(groups):
            vals = store_by_race.loc[grp, STORE_TYPES].values
            ax.bar(x + j * width, vals, width=width, label=grp,
                   color=grp_colors.get(grp, "#95A5A6"), edgecolor="white")

        ax.set_xticks(x + width)
        ax.set_xticklabels([s.replace("_", " ").title() for s in STORE_TYPES])
        ax.set_ylabel("Mean Count per ZIP")
        ax.set_title("Store Access Equity by Racial Composition of ZIP", fontweight="bold")
        ax.legend()
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "12_store_equity_race.png")
        plt.close()
        print("  Saved → plots/12_store_equity_race.png")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7  Transportation Barrier Deep-Dive
# ─────────────────────────────────────────────────────────────────────────────

section("SECTION 7 — Transportation Barrier Deep-Dive")

if "pct_no_vehicle" in df.columns and "nearest_supermarket_miles" in df.columns:
    # Quartile cross-tab: no-vehicle rate × supermarket distance
    df["noveh_q"] = pd.qcut(df["pct_no_vehicle"].rank(method="first"), q=4,
                             labels=["Low NoVeh", "Med-Low", "Med-High", "High NoVeh"])
    df["dist_q"]  = pd.qcut(df["nearest_supermarket_miles"].rank(method="first"), q=4,
                             labels=["Near", "Med-Near", "Med-Far", "Far"])

    transport_xtab = pd.crosstab(df["noveh_q"], df["dist_q"])
    print("\n  ZIP count by no-vehicle rate × supermarket distance quartile:")
    print(transport_xtab.to_string())
    transport_xtab.to_csv(DATA_DIR / "transport_barrier_crosstab.csv")

    # What share of 'high noveh + far' ZIPs are also flagged as deserts?
    barrier_mask = (df["noveh_q"] == "High NoVeh") & (df["dist_q"] == "Far")
    barrier_zips = df[barrier_mask]
    print(f"\n  ZIPs with HIGH no-vehicle rate AND FAR from supermarket: {barrier_mask.sum()}")
    if "is_food_desert" in df.columns:
        print(f"    → of these, {barrier_zips['is_food_desert'].sum():.0f} "
              f"({barrier_zips['is_food_desert'].mean()*100:.1f}%) also flagged as food desert")

    # Scatter: no-vehicle vs distance, coloured by desert
    fig, ax = plt.subplots(figsize=(9, 7))
    colors_scatter = df.get("is_food_desert", pd.Series(0, index=df.index)).fillna(0).map(
        {0: PALETTE["neutral"], 1: PALETTE["desert"]}
    )
    ax.scatter(df["pct_no_vehicle"], df["nearest_supermarket_miles"],
               c=colors_scatter, alpha=0.5, s=40, edgecolors="none")
    ax.set_xlabel("% Households with No Vehicle")
    ax.set_ylabel("Distance to Nearest Supermarket (miles)")
    ax.set_title("Transportation Barriers vs. Supermarket Access\n"
                 "(Red = food desert ZIP)", fontweight="bold")

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=PALETTE["desert"], label="Food Desert"),
                        Patch(color=PALETTE["neutral"], label="Non-Desert")], loc="upper right")

    # Annotate high-stress quadrant
    q75_noveh = df["pct_no_vehicle"].quantile(0.75)
    q75_dist  = df["nearest_supermarket_miles"].quantile(0.75)
    ax.axvline(q75_noveh, color=PALETTE["dark_grey"], linestyle="--", lw=1, alpha=0.6)
    ax.axhline(q75_dist,  color=PALETTE["dark_grey"], linestyle="--", lw=1, alpha=0.6)
    ax.text(q75_noveh + 0.5, q75_dist + 0.2, "High Dual\nBarrier Zone",
            color=PALETTE["desert"], fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "13_transport_barrier_scatter.png")
    plt.close()
    print("  Saved → plots/13_transport_barrier_scatter.png")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8  County-Level Benchmarking
# ─────────────────────────────────────────────────────────────────────────────

section("SECTION 8 — County-Level Benchmarking")

COUNTY_METRICS = [c for c in [
    "composite_vuln_index", "pct_poverty", "pct_no_vehicle",
    "nearest_supermarket_miles", "supermarket", "fast_food",
    "swamp_score_continuous", "mrfei",
    "Diabetes % (Adults)", "Obesity % (Adults)", "Food Insecurity % (Adults)",
] if c in df.columns]

if "county" in df.columns and COUNTY_METRICS:
    county_stats = df.groupby("county")[COUNTY_METRICS].agg(["mean", "median"]).round(2)
    county_stats.columns = ["_".join(c) for c in county_stats.columns]

    # Population-weighted vulnerability
    if "population" in df.columns and "composite_vuln_index" in df.columns:
        def pw_mean(grp):
            return weighted_mean(grp["composite_vuln_index"], grp["population"])
        county_stats["vuln_popwt"] = df.groupby("county").apply(pw_mean).round(2)

    # Desert / swamp counts per county
    if "is_food_desert" in df.columns:
        county_stats["n_desert_zips"] = df.groupby("county")["is_food_desert"].sum().astype(int)
        county_stats["pct_desert_zips"] = (
            df.groupby("county")["is_food_desert"].mean() * 100
        ).round(1)

    if "is_swamp_consensus" in df.columns:
        county_stats["n_swamp_zips"] = df.groupby("county")["is_swamp_consensus"].sum().astype(int)

    county_stats = county_stats.sort_values("composite_vuln_index_mean", ascending=False)
    print("\n  County benchmarks (top 10 most vulnerable):")
    print(county_stats.head(10).to_string())
    county_stats.to_csv(DATA_DIR / "county_benchmarks_extended.csv")

    # Bubble chart: county vulnerability × diabetes × ZIP count
    county_plot = df.groupby("county").agg(
        vuln  = ("composite_vuln_index", "mean"),
        diab  = ("Diabetes % (Adults)", "mean"),
        n_zip = ("zip", "count"),
        pov   = ("pct_poverty", "mean"),
    ).dropna()

    fig, ax = plt.subplots(figsize=(12, 8))
    sc = ax.scatter(
        county_plot["vuln"], county_plot["diab"],
        s=county_plot["n_zip"] * 8,
        c=county_plot["pov"], cmap="YlOrRd",
        alpha=0.75, edgecolors="grey", linewidths=0.5,
    )
    for name, row in county_plot.iterrows():
        ax.annotate(name, (row["vuln"], row["diab"]),
                    fontsize=7.5, ha="center", va="bottom",
                    xytext=(0, 5), textcoords="offset points")

    plt.colorbar(sc, ax=ax, label="Mean % Poverty")
    ax.set_xlabel("Mean Composite Vulnerability Index")
    ax.set_ylabel("Mean Diabetes Rate %")
    ax.set_title("County-Level: Vulnerability vs. Diabetes\n"
                 "(Bubble size = # ZIPs, color = poverty rate)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "14_county_bubble_chart.png")
    plt.close()
    print("  Saved → plots/14_county_bubble_chart.png")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9  Dual-Burden ZIPs — Desert AND Swamp
# ─────────────────────────────────────────────────────────────────────────────

section("SECTION 9 — Dual-Burden ZIPs (Desert + Swamp)")

BURDEN_GROUPS = {
    "Neither":      (df.get("is_food_desert", 0).fillna(0) == 0) & (df.get("is_swamp_consensus", 0).fillna(0) == 0),
    "Desert Only":  (df.get("is_food_desert", 0).fillna(0) == 1) & (df.get("is_swamp_consensus", 0).fillna(0) == 0),
    "Swamp Only":   (df.get("is_food_desert", 0).fillna(0) == 0) & (df.get("is_swamp_consensus", 0).fillna(0) == 1),
    "Dual Burden":  (df.get("is_food_desert", 0).fillna(0) == 1) & (df.get("is_swamp_consensus", 0).fillna(0) == 1),
}

burden_profiles = []
for label, mask in BURDEN_GROUPS.items():
    sub = df[mask]
    row = {"group": label, "n_zips": mask.sum()}
    for col in [c for c in HEALTH_OUTCOMES[:6] + ["pct_poverty", "composite_vuln_index"] if c in df.columns]:
        row[col] = sub[col].mean()
    if "population" in df.columns:
        row["total_population"] = sub["population"].sum()
    burden_profiles.append(row)

burden_df = pd.DataFrame(burden_profiles).set_index("group")
print("\n  Dual-burden group profiles:")
print(burden_df.round(2).to_string())
burden_df.to_csv(DATA_DIR / "dual_burden_profiles.csv")

# Grouped bar: health outcomes across 4 burden groups
if len(HEALTH_OUTCOMES) >= 4:
    metrics_plot = [c for c in HEALTH_OUTCOMES[:4] if c in burden_df.columns]
    fig, axes = plt.subplots(1, len(metrics_plot), figsize=(16, 5))
    group_colors = [PALETTE["neutral"], PALETTE["desert"], PALETTE["swamp"], PALETTE["dual"]]

    for i, metric in enumerate(metrics_plot):
        ax = axes[i]
        vals = burden_df[metric].values
        ax.bar(range(4), vals, color=group_colors, edgecolor="white")
        ax.set_xticks(range(4))
        ax.set_xticklabels(burden_df.index.tolist(), rotation=25, ha="right", fontsize=8)
        ax.set_title(metric.replace(" % (Adults)", ""), fontsize=10, fontweight="bold")
        ax.set_ylabel("Mean %" if i == 0 else "")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    fig.suptitle("Health Outcomes by Food Environment Burden Status",
                 fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "15_dual_burden_health.png")
    plt.close()
    print("  Saved → plots/15_dual_burden_health.png")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10  Spatial Isolation × Socioeconomic Vulnerability 2×2 Matrix
# ─────────────────────────────────────────────────────────────────────────────

section("SECTION 10 — Spatial Isolation × Vulnerability 2×2 Matrix")

if "nearest_supermarket_miles" in df.columns and "composite_vuln_index" in df.columns:
    dist_med  = df["nearest_supermarket_miles"].median()
    vuln_med  = df["composite_vuln_index"].median()

    df["spatial_isolation"] = np.where(df["nearest_supermarket_miles"] >= dist_med,
                                        "High Isolation", "Low Isolation")
    df["vuln_level"]        = np.where(df["composite_vuln_index"] >= vuln_med,
                                        "High Vulnerability", "Low Vulnerability")
    df["quadrant"] = df["spatial_isolation"] + " / " + df["vuln_level"]

    quad_summary = df.groupby("quadrant").agg(
        n_zips=("zip", "count"),
        total_pop=("population", "sum") if "population" in df.columns else ("zip", "count"),
        mean_poverty=("pct_poverty", "mean"),
        mean_diabetes=("Diabetes % (Adults)", "mean"),
        mean_vuln=("composite_vuln_index", "mean"),
    ).round(2)

    print("\n  2×2 spatial isolation × vulnerability quadrant summary:")
    print(quad_summary.to_string())
    quad_summary.to_csv(DATA_DIR / "isolation_vuln_quadrants.csv")

    # 2×2 heatmap of ZIP counts
    pivot_n = df.pivot_table(index="spatial_isolation", columns="vuln_level",
                              values="zip", aggfunc="count")
    pivot_d = df.pivot_table(index="spatial_isolation", columns="vuln_level",
                              values="Diabetes % (Adults)", aggfunc="mean") if "Diabetes % (Adults)" in df.columns else None

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(pivot_n, annot=True, fmt="g", cmap="Blues",
                linewidths=0.5, ax=axes[0])
    axes[0].set_title("ZIP Count per Quadrant", fontweight="bold")

    if pivot_d is not None:
        sns.heatmap(pivot_d, annot=True, fmt=".1f", cmap="Reds",
                    linewidths=0.5, ax=axes[1])
        axes[1].set_title("Mean Diabetes Rate % per Quadrant", fontweight="bold")

    fig.suptitle("Spatial Isolation × Vulnerability 2×2 Matrix", fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "16_isolation_vuln_matrix.png")
    plt.close()
    print("  Saved → plots/16_isolation_vuln_matrix.png")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11  Spearman Correlation Heatmap — Food Environment × Health
# ─────────────────────────────────────────────────────────────────────────────

section("SECTION 11 — Correlation Heatmap: Food Environment × Health")

ENV_VARS = [c for c in [
    "composite_vuln_index", "pct_poverty", "pct_no_vehicle", "pct_elderly",
    "median_income", "nearest_supermarket_miles", "supermarkets_within_5mi",
    "swamp_score_continuous", "mrfei", "rfei",
    "supermarket", "fast_food", "convenience", "produce_market",
] if c in df.columns]

HEALTH_VARS = [c for c in HEALTH_OUTCOMES if c in df.columns]

if ENV_VARS and HEALTH_VARS:
    corr_matrix = pd.DataFrame(index=ENV_VARS, columns=HEALTH_VARS, dtype=float)
    pval_matrix = pd.DataFrame(index=ENV_VARS, columns=HEALTH_VARS, dtype=float)

    for ev in ENV_VARS:
        for hv in HEALTH_VARS:
            both = df[[ev, hv]].dropna()
            if len(both) < 10:
                continue
            r, p = spearmanr(both[ev], both[hv])
            corr_matrix.loc[ev, hv] = round(r, 3)
            pval_matrix.loc[ev, hv] = p

    print("\n  Top 10 food environment → health Spearman correlations:")
    flat = corr_matrix.stack().rename("r").reset_index()
    flat.columns = ["env_var", "health_var", "r"]
    flat["abs_r"] = flat["r"].abs()
    print(flat.sort_values("abs_r", ascending=False).head(10).drop("abs_r", axis=1).to_string(index=False))
    flat.to_csv(DATA_DIR / "env_health_correlations.csv", index=False)

    short_env    = [c.replace("pct_", "% ").replace("_", " ").replace("composite vuln index", "Vulnerability").title()
                    for c in ENV_VARS]
    short_health = [c.replace(" % (Adults)", "").replace(" % (Women)", "") for c in HEALTH_VARS]

    # Significance mask — blank out p >= 0.05
    sig_mask = pval_matrix.astype(float) >= 0.05

    fig, ax = plt.subplots(figsize=(max(10, len(HEALTH_VARS) * 0.9),
                                     max(6, len(ENV_VARS) * 0.55)))
    sns.heatmap(
        corr_matrix.astype(float), mask=sig_mask,
        annot=True, fmt=".2f", cmap="RdBu_r", center=0,
        vmin=-1, vmax=1, ax=ax,
        xticklabels=short_health, yticklabels=short_env,
        linewidths=0.3, linecolor="white",
        annot_kws={"size": 8},
    )
    ax.set_title("Spearman Correlations: Food Environment × Health Outcomes\n"
                 "(Blank cells = p ≥ 0.05)", fontweight="bold")
    plt.xticks(rotation=35, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "17_correlation_heatmap.png")
    plt.close()
    print("  Saved → plots/17_correlation_heatmap.png")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12  Population-Weighted Summary Statistics
# ─────────────────────────────────────────────────────────────────────────────

section("SECTION 12 — Population-Weighted Summary Statistics")

if "population" in df.columns:
    total_pop = df["population"].sum()

    pw_stats = {}
    for col in [c for c in HEALTH_OUTCOMES + VULN_DRIVERS if c in df.columns]:
        raw_mean = df[col].mean()
        pop_mean = weighted_mean(df[col], df["population"])
        pw_stats[col] = {
            "raw_mean":      round(raw_mean, 3),
            "pop_wt_mean":   round(pop_mean, 3),
            "diff":          round(pop_mean - raw_mean, 3),
            "n_zips":        df[col].notna().sum(),
        }

    pw_df = pd.DataFrame(pw_stats).T.sort_values("diff", ascending=False)
    print("\n  Population-weighted vs. unweighted means (top 10 largest gaps):")
    print(pw_df.head(10).to_string())
    pw_df.to_csv(DATA_DIR / "population_weighted_stats.csv")

    # Affected population estimates
    print("\n  Key population estimates:")
    pop_rows = []
    for flag, label in [
        ("is_food_desert",    "Food Desert residents"),
        ("is_swamp_consensus","Food Swamp residents"),
        ("is_dual_burden",    "Dual-Burden residents"),
    ]:
        if flag in df.columns:
            pop_affected = df.loc[df[flag].fillna(0) == 1, "population"].sum()
            print(f"    {label:30s}: {pct_fmt(pop_affected, total_pop)} of NJ pop")
            pop_rows.append({"category": label, "population": pop_affected,
                             "pct_nj": pop_affected / total_pop * 100})

    pd.DataFrame(pop_rows).to_csv(DATA_DIR / "affected_population_estimates.csv", index=False)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 13  Racial Equity — Store Access Disparities
# ─────────────────────────────────────────────────────────────────────────────

section("SECTION 13 — Racial Equity: Store Access Disparities")

RACE_COLS = {
    "pct_black":    "Population Non-Hispanic Black",
    "pct_hispanic": "Population Hispanic or Latino",
    "pct_asian":    "Population Non-Hispanic Asian",
    "pct_white":    "Population Non-Hispanic White",
}

for new_col, src_col in RACE_COLS.items():
    if src_col in df.columns and "population" in df.columns:
        df[new_col] = df[src_col] / df["population"].replace(0, np.nan)

# Categorise ZIPs by dominant group (≥40% share)
race_present = [c for c in RACE_COLS if c in df.columns]
if race_present:
    def dominant_race(row):
        shares = {rc: row[rc] for rc in race_present if pd.notna(row[rc])}
        if not shares:
            return "Unknown"
        best = max(shares, key=shares.get)
        return best.replace("pct_", "").title() if shares[best] >= 0.40 else "Mixed"

    df["dominant_race"] = df.apply(dominant_race, axis=1)

    race_access = df.groupby("dominant_race")[
        [c for c in STORE_TYPES + ["nearest_supermarket_miles", "composite_vuln_index"]
         if c in df.columns]
    ].mean().round(2)

    print("\n  Store access & vulnerability by dominant racial group:")
    print(race_access.to_string())
    race_access.to_csv(DATA_DIR / "racial_equity_store_access.csv")

    # Bar chart: nearest supermarket distance by race group
    if "nearest_supermarket_miles" in race_access.columns:
        fig, ax = plt.subplots(figsize=(9, 5))
        race_sorted = race_access["nearest_supermarket_miles"].sort_values(ascending=False)
        bars = ax.bar(range(len(race_sorted)), race_sorted.values,
                      color=[PALETTE["desert"], PALETTE["swamp"], PALETTE["neutral"],
                             PALETTE["healthy"], PALETTE["dual"]][:len(race_sorted)],
                      edgecolor="white")
        ax.set_xticks(range(len(race_sorted)))
        ax.set_xticklabels(race_sorted.index.tolist(), rotation=20, ha="right")
        ax.set_ylabel("Mean Distance to Nearest Supermarket (miles)")
        ax.set_title("Supermarket Access Disparities by Racial Composition of ZIP",
                     fontweight="bold")
        ax.axhline(df["nearest_supermarket_miles"].mean(), color="black",
                   linestyle="--", lw=1.2, label="NJ mean")
        ax.legend()
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "18_racial_equity_distance.png")
        plt.close()
        print("  Saved → plots/18_racial_equity_distance.png")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 14  Access Typology Transition Matrix
# ─────────────────────────────────────────────────────────────────────────────

section("SECTION 14 — Access Typology Transition: Desert × Swamp × Mirage")

transition_flags = [c for c in [
    "is_food_desert", "is_swamp_consensus", "is_food_mirage", "is_rural_mirage",
] if c in df.columns]

if len(transition_flags) >= 2:
    transition_df = df[transition_flags].fillna(0).astype(int)
    transition_df["n_flags"] = transition_df.sum(axis=1)
    transition_df["typology_combo"] = transition_df[transition_flags].apply(
        lambda r: " + ".join(
            [f.replace("is_", "").replace("_consensus", "").replace("_", " ").title()
             for f in transition_flags if r[f] == 1]
        ) or "None", axis=1
    )

    combo_counts = transition_df["typology_combo"].value_counts()
    print("\n  Food environment typology combinations:")
    print(combo_counts.to_string())
    combo_counts.to_csv(DATA_DIR / "typology_combinations.csv")

    fig, ax = plt.subplots(figsize=(10, 5))
    combo_counts.head(12).plot(kind="barh", ax=ax, color=PALETTE["neutral"], edgecolor="white")
    ax.set_xlabel("Number of ZIPs")
    ax.set_title("Food Environment Typology Combinations", fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "19_typology_combinations.png")
    plt.close()
    print("  Saved → plots/19_typology_combinations.png")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 15  Extreme-Case Narratives
# ─────────────────────────────────────────────────────────────────────────────

section("SECTION 15 — Extreme-Case Narratives")

NARRATIVE_COLS = [
    "zip", "county", "municipality", "population",
    "median_income",
    "composite_vuln_index",
    "pct_poverty",
    "pct_no_vehicle",
    "pct_elderly",
    "nearest_supermarket_miles",
    "supermarket",
    "fast_food",
    "swamp_score_continuous",
    "mrfei",
    "Diabetes % (Adults)",
    "Obesity % (Adults)",
    "Food Insecurity % (Adults)",
    "is_food_desert",
    "is_swamp_consensus",
    "is_dual_burden",
]

# Top 20 most vulnerable ZIPs with meaningful population
if "composite_vuln_index" in df.columns and "population" in df.columns:
    top_vuln = (
        df[df["population"] >= 500]
        .nlargest(20, "composite_vuln_index")[NARRATIVE_COLS]
    )
    print("\n  Top 20 most vulnerable ZIPs (pop ≥ 500):")
    print(top_vuln.to_string(index=False))
    top_vuln.to_csv(DATA_DIR / "top_vulnerable_zips.csv", index=False)

# Worst health × worst food environment overlap
if "Diabetes % (Adults)" in df.columns and "composite_vuln_index" in df.columns:
    df["rank_diab"] = df["Diabetes % (Adults)"].rank(ascending=False)
    df["rank_vuln"] = df["composite_vuln_index"].rank(ascending=False)
    df["dual_rank"]  = (df["rank_diab"] + df["rank_vuln"]) / 2

    top_dual = df[df.get("population", pd.Series(1, index=df.index)) >= 500] \
               .nsmallest(15, "dual_rank")[NARRATIVE_COLS]
    print("\n  Top 15 ZIPs — worst health + worst food environment combined:")
    print(top_dual.to_string(index=False))
    top_dual.to_csv(DATA_DIR / "top_dual_crisis_zips.csv", index=False)

# Hidden-strength ZIPs: high vulnerability but GOOD food access
if "composite_vuln_index" in df.columns and "nearest_supermarket_miles" in df.columns:
    hidden = df[
        (df["composite_vuln_index"] >= df["composite_vuln_index"].quantile(0.75)) &
        (df["nearest_supermarket_miles"] <= df["nearest_supermarket_miles"].quantile(0.25))
    ][NARRATIVE_COLS]
    print(f"\n  'Hidden strength' ZIPs (high vuln + good physical access): {len(hidden)}")
    print("  (High vulnerability but within 1st quartile distance to supermarket)")
    hidden.to_csv(DATA_DIR / "hidden_strength_zips.csv", index=False)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 16  Kruskal-Wallis + Dunn Post-Hoc Tests
# ─────────────────────────────────────────────────────────────────────────────

section("SECTION 16 — Kruskal-Wallis + Dunn Post-Hoc Tests")

try:
    from scikit_posthocs import posthoc_dunn
    HAS_DUNN = True
except ImportError:
    try:
        from scikit_posthocs import posthoc_dunn
        HAS_DUNN = True
    except ImportError:
        HAS_DUNN = False
        print("  Note: scikit-posthocs not installed — KW omnibus only, no Dunn post-hoc.")
        print("  Install with: pip install scikit-posthocs")

KW_OUTCOMES = [c for c in [
    "Diabetes % (Adults)", "Obesity % (Adults)",
    "High Blood Pressure % (Adults)", "Coronary Heart Disease % (Adults)",
    "Physical Inactivity % (Adults)", "Food Insecurity % (Adults)",
    "Depression % (Adults)", "Poor Mental Health % (Adults)",
    "Current Smoking % (Adults)", "Any Disability % (Adults)",
] if c in df.columns]

ALPHA = 0.05

def kw_label(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def run_kw_block(group_col, group_label, outcomes, min_group_size=5):
    """
    Run Kruskal-Wallis across all outcomes for one grouping variable.
    Returns a summary DataFrame and a dict of Dunn matrices.
    """
    if group_col not in df.columns:
        print(f"  Skipping {group_label} — column not found")
        return None, {}

    groups = df[group_col].dropna().unique()
    valid_groups = [g for g in groups
                    if df[df[group_col] == g].shape[0] >= min_group_size]

    if len(valid_groups) < 2:
        print(f"  Skipping {group_label} — fewer than 2 groups with n >= {min_group_size}")
        return None, {}

    print(f"\n  ── {group_label} ({len(valid_groups)} groups) ──")

    kw_rows = []
    dunn_matrices = {}

    for outcome in outcomes:
        col_data = df[[group_col, outcome]].dropna()
        group_arrays = [
            col_data.loc[col_data[group_col] == g, outcome].values
            for g in valid_groups
        ]
        # Drop groups with no variance (KW will error)
        group_arrays = [a for a in group_arrays if len(a) >= 2 and a.std() > 0]
        if len(group_arrays) < 2:
            continue

        stat, p = kruskal(*group_arrays)
        eta_sq = (stat - len(group_arrays) + 1) / (len(col_data) - len(group_arrays))
        eta_sq = max(0.0, eta_sq)  # clamp to 0

        row = {
            "outcome":     outcome.replace(" % (Adults)", "").replace(" % (Women)", ""),
            "H_stat":      round(stat, 3),
            "p_value":     round(p, 5),
            "sig":         kw_label(p),
            "eta_sq":      round(eta_sq, 4),
            "n_groups":    len(group_arrays),
            "n_total":     len(col_data),
        }
        kw_rows.append(row)

        # Dunn post-hoc (only if KW is significant and library available)
        if p < ALPHA and HAS_DUNN:
            try:
                dunn_result = posthoc_dunn(
                    col_data, val_col=outcome, group_col=group_col,
                    p_adjust="bonferroni"
                )
                dunn_matrices[outcome] = dunn_result
            except Exception as e:
                print(f"    Dunn failed for {outcome}: {e}")

    if not kw_rows:
        return None, {}

    kw_df = pd.DataFrame(kw_rows).sort_values("H_stat", ascending=False)
    print(kw_df.to_string(index=False))
    return kw_df, dunn_matrices


# ── 16a. By Food Access Typology ─────────────────────────────────────────────

TYPOLOGY_COL = next(
    (c for c in ["food_access_type", "access_typology"] if c in df.columns), None
)

kw_typology, dunn_typology = run_kw_block(
    TYPOLOGY_COL, "Food Access Typology", KW_OUTCOMES
)

if kw_typology is not None:
    kw_typology.to_csv(DATA_DIR / "kw_typology.csv", index=False)
    print("  Saved → data/kw_typology.csv")

    # Save each significant Dunn matrix
    for outcome, matrix in dunn_typology.items():
        safe_name = outcome.replace(" % (Adults)", "").replace(" ", "_").replace("/", "_")
        matrix.to_csv(DATA_DIR / f"dunn_typology_{safe_name}.csv")

    # Plot: H-statistic bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    colors_kw = [PALETTE["desert"] if p < 0.05 else PALETTE["dark_grey"]
                 for p in kw_typology["p_value"]]
    bars = ax.barh(kw_typology["outcome"], kw_typology["H_stat"],
                   color=colors_kw, edgecolor="white")
    for bar, (_, row) in zip(bars, kw_typology.iterrows()):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                row["sig"], va="center", fontsize=11,
                color=PALETTE["desert"] if row["p_value"] < 0.05 else PALETTE["dark_grey"])
    ax.set_xlabel("Kruskal-Wallis H Statistic")
    ax.set_title("Kruskal-Wallis Test: Health Outcomes by Food Access Typology\n"
                 "(Red = p < .05, *** p<.001, ** p<.01, * p<.05, ns = not significant)",
                 fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "20_kw_typology.png")
    plt.close()
    print("  Saved → plots/20_kw_typology.png")

    # Plot: Dunn post-hoc heatmaps for top 3 most significant outcomes
    if dunn_typology:
        top_outcomes = kw_typology[kw_typology["p_value"] < ALPHA].head(3)["outcome"].tolist()
        # Re-map back to original column names
        full_names = {
            o.replace(" % (Adults)", "").replace(" % (Women)", ""): o_full
            for o_full in KW_OUTCOMES
            for o in [o_full.replace(" % (Adults)", "").replace(" % (Women)", "")]
        }
        sig_dunn = {k: v for k, v in dunn_typology.items()
                    if k.replace(" % (Adults)", "").replace(" % (Women)", "") in top_outcomes}

        if sig_dunn:
            n_plots = len(sig_dunn)
            fig, axes = plt.subplots(1, n_plots,
                                     figsize=(max(8, n_plots * 7), 6))
            if n_plots == 1:
                axes = [axes]

            for ax, (outcome, matrix) in zip(axes, sig_dunn.items()):
                # Convert p-values to significance labels for annotation
                def sig_label(p):
                    if p < 0.001: return "***"
                    if p < 0.01:  return "**"
                    if p < 0.05:  return "*"
                    return "ns"


                annot = matrix.map(sig_label) if hasattr(matrix, "map") else matrix.applymap(sig_label)
                sns.heatmap(
                    matrix, annot=annot, fmt="", cmap="RdYlGn",
                    vmin=0, vmax=0.1, ax=ax,
                    linewidths=0.4, linecolor="white",
                    annot_kws={"size": 8},
                )
                short = outcome.replace(" % (Adults)", "")
                ax.set_title(f"Dunn Post-Hoc: {short}\n(Bonferroni-adjusted p-values)",
                             fontweight="bold", fontsize=10)
                plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
                plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)

            fig.suptitle("Pairwise Dunn Tests — Food Access Typology",
                         fontweight="bold", fontsize=13)
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / "21_dunn_typology.png")
            plt.close()
            print("  Saved → plots/21_dunn_typology.png")


# ── 16b. By Vulnerability Quintile ───────────────────────────────────────────

kw_vuln, dunn_vuln = run_kw_block(
    "vuln_quintile", "Vulnerability Quintile", KW_OUTCOMES
)

if kw_vuln is not None:
    kw_vuln.to_csv(DATA_DIR / "kw_vuln_quintile.csv", index=False)
    print("  Saved → data/kw_vuln_quintile.csv")

    for outcome, matrix in dunn_vuln.items():
        safe_name = outcome.replace(" % (Adults)", "").replace(" ", "_").replace("/", "_")
        matrix.to_csv(DATA_DIR / f"dunn_vuln_{safe_name}.csv")

    # Eta-squared plot — effect size across quintiles
    fig, ax = plt.subplots(figsize=(10, 5))
    kw_sorted = kw_vuln.sort_values("eta_sq", ascending=True)
    colors_eta = [PALETTE["desert"] if e >= 0.14 else
                  PALETTE["swamp"] if e >= 0.06 else
                  PALETTE["neutral"] if e >= 0.01 else
                  PALETTE["dark_grey"]
                  for e in kw_sorted["eta_sq"]]
    ax.barh(kw_sorted["outcome"], kw_sorted["eta_sq"],
            color=colors_eta, edgecolor="white")
    ax.axvline(0.01, color=PALETTE["dark_grey"], linestyle=":", lw=1, label="Small (η²=.01)")
    ax.axvline(0.06, color=PALETTE["neutral"],   linestyle=":", lw=1, label="Medium (η²=.06)")
    ax.axvline(0.14, color=PALETTE["desert"],    linestyle=":", lw=1, label="Large (η²=.14)")
    ax.set_xlabel("Eta-Squared (η²) — Proportion of Variance Explained")
    ax.set_title("Effect Size (η²) by Health Outcome\nGrouped by Vulnerability Quintile",
                 fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "22_kw_eta_sq_vuln.png")
    plt.close()
    print("  Saved → plots/22_kw_eta_sq_vuln.png")


# ── 16c. By Income Tertile ────────────────────────────────────────────────────

kw_income, dunn_income = run_kw_block(
    "income_tertile", "Income Tertile", KW_OUTCOMES
)

if kw_income is not None:
    kw_income.to_csv(DATA_DIR / "kw_income_tertile.csv", index=False)
    print("  Saved → data/kw_income_tertile.csv")

    for outcome, matrix in dunn_income.items():
        safe_name = outcome.replace(" % (Adults)", "").replace(" ", "_").replace("/", "_")
        matrix.to_csv(DATA_DIR / f"dunn_income_{safe_name}.csv")


# ── 16d. By Racial Composition ────────────────────────────────────────────────

kw_race, dunn_race = run_kw_block(
    "dominant_race", "Dominant Racial Group", KW_OUTCOMES
)

if kw_race is not None:
    kw_race.to_csv(DATA_DIR / "kw_dominant_race.csv", index=False)
    print("  Saved → data/kw_dominant_race.csv")

    for outcome, matrix in dunn_race.items():
        safe_name = outcome.replace(" % (Adults)", "").replace(" ", "_").replace("/", "_")
        matrix.to_csv(DATA_DIR / f"dunn_race_{safe_name}.csv")


# ── 16e. Master KW Summary Table ─────────────────────────────────────────────

print("\n  ── Master Kruskal-Wallis Summary ──")

all_kw = []
for label, kw_df in [
    ("Typology",       kw_typology),
    ("Vuln Quintile",  kw_vuln),
    ("Income Tertile", kw_income),
    ("Race Group",     kw_race),
]:
    if kw_df is not None:
        tmp = kw_df[["outcome", "H_stat", "p_value", "sig", "eta_sq"]].copy()
        tmp.insert(0, "grouping", label)
        all_kw.append(tmp)

if all_kw:
    master_kw = pd.concat(all_kw, ignore_index=True).sort_values(
        ["grouping", "H_stat"], ascending=[True, False]
    )
    print(master_kw.to_string(index=False))
    master_kw.to_csv(DATA_DIR / "kw_master_summary.csv", index=False)
    print("  Saved → data/kw_master_summary.csv")

    # Bubble chart: H-stat × eta_sq, one bubble per (grouping × outcome)
    fig, ax = plt.subplots(figsize=(12, 7))
    group_colors_map = {
        "Typology":       PALETTE["desert"],
        "Vuln Quintile":  PALETTE["swamp"],
        "Income Tertile": PALETTE["neutral"],
        "Race Group":     PALETTE["dual"],
    }
    for grp, sub in master_kw.groupby("grouping"):
        sig_mask = sub["p_value"] < ALPHA
        ax.scatter(
            sub.loc[sig_mask, "H_stat"],
            sub.loc[sig_mask, "eta_sq"],
            s=120, alpha=0.85,
            color=group_colors_map.get(grp, "grey"),
            edgecolors="white", linewidths=0.5,
            label=grp, zorder=3,
        )
        ax.scatter(
            sub.loc[~sig_mask, "H_stat"],
            sub.loc[~sig_mask, "eta_sq"],
            s=60, alpha=0.3,
            color=group_colors_map.get(grp, "grey"),
            edgecolors="none", zorder=2,
        )

    # Reference lines for eta_sq thresholds
    ax.axhline(0.01, color=PALETTE["dark_grey"], linestyle=":", lw=1, label="Small η²=.01")
    ax.axhline(0.06, color=PALETTE["neutral"],   linestyle=":", lw=1, label="Medium η²=.06")
    ax.axhline(0.14, color=PALETTE["desert"],    linestyle=":", lw=1, label="Large η²=.14")

    ax.set_xlabel("Kruskal-Wallis H Statistic (larger = more separation)")
    ax.set_ylabel("Eta-Squared η² (effect size)")
    ax.set_title("KW Test Results Across All Groupings\n"
                 "(Solid = significant p<.05, faded = ns)", fontweight="bold")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "23_kw_master_bubble.png")
    plt.close()
    print("  Saved → plots/23_kw_master_bubble.png")


# ── Final Summary ─────────────────────────────────────────────────────────────

print(f"""
{'=' * 60}
  ANALYTICS COMPLETE — EXTENDED OUTPUTS
{'=' * 60}

  DATA FILES
  ───────────────────────────────────────────────────────
  data/data_audit_extended.csv           Full data quality audit
  data/vuln_quintile_profiles.csv        Driver values by vulnerability tier
  data/desert_method_agreement.csv       Cross-method classification agreement
  data/health_by_typology.csv            Health outcomes by food access typology
  data/health_effect_sizes_desert.csv    Cohen's d + significance, desert vs. not
  data/store_by_income_tertile.csv       Store access by income group
  data/store_by_racial_composition.csv   Store access by racial composition
  data/transport_barrier_crosstab.csv    No-vehicle × distance cross-tabulation
  data/county_benchmarks_extended.csv    County-level multi-metric benchmarks
  data/dual_burden_profiles.csv          Desert + swamp dual burden profiles
  data/isolation_vuln_quadrants.csv      2×2 isolation × vulnerability matrix
  data/env_health_correlations.csv       Spearman r — all env × health pairs
  data/population_weighted_stats.csv     Weighted vs. unweighted means
  data/affected_population_estimates.csv Population counts by burden type
  data/racial_equity_store_access.csv    Store access by dominant racial group
  data/typology_combinations.csv         Flag co-occurrence counts
  data/top_vulnerable_zips.csv           20 most vulnerable ZIPs (pop ≥ 500)
  data/top_dual_crisis_zips.csv          15 ZIPs with worst health + food env
  data/hidden_strength_zips.csv          High-vuln ZIPs with good physical access

  PLOTS
  ───────────────────────────────────────────────────────
  plots/06_vuln_driver_profiles.png      Driver values by vulnerability quintile
  plots/07_desert_method_agreement.png   Desert classification agreement matrix
  plots/08_swamp_method_agreement.png    Swamp flag combinations
  plots/09_health_by_typology_heatmap.png  Health z-scores by typology
  plots/10_health_gap_desert_lollipop.png  Desert vs. non-desert health gaps
  plots/11_store_equity_income.png       Store counts by income tertile
  plots/12_store_equity_race.png         Store counts by racial composition
  plots/13_transport_barrier_scatter.png  No-vehicle vs. distance scatter
  plots/14_county_bubble_chart.png       County vulnerability × diabetes bubble
  plots/15_dual_burden_health.png        Health by desert+swamp burden status
  plots/16_isolation_vuln_matrix.png     2×2 quadrant heatmaps
  plots/17_correlation_heatmap.png       Full env × health Spearman heatmap
  plots/18_racial_equity_distance.png    Distance to supermarket by race
  plots/19_typology_combinations.png     Typology flag combination counts
  plots/20_kw_typology.png               KW H-stats by food access typology
  plots/21_dunn_typology.png             Dunn post-hoc heatmaps (top outcomes)
  plots/22_kw_eta_sq_vuln.png            Effect size (eta-sq) by vuln quintile
  plots/23_kw_master_bubble.png          Master KW bubble: H-stat vs eta-sq

  STATISTICAL TESTS (Section 16)
  ───────────────────────────────────────────────────────
  data/kw_typology.csv                   KW results — food access typology
  data/kw_vuln_quintile.csv              KW results — vulnerability quintile
  data/kw_income_tertile.csv             KW results — income tertile
  data/kw_dominant_race.csv              KW results — racial composition
  data/kw_master_summary.csv             All KW results combined
  data/dunn_typology_*.csv               Dunn matrices per significant outcome
  data/dunn_vuln_*.csv                   Dunn matrices — vuln quintile
  data/dunn_income_*.csv                 Dunn matrices — income tertile
  data/dunn_race_*.csv                   Dunn matrices — racial composition
""")