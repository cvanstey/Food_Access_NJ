"""
08_poverty_requirement_mcnemar.py
==================
Research Question 1: How does removing the poverty requirement change
which ZIPs are classified as food deserts?

Compares two USDA-defined flags for the SAME 598 ZIPs:
  usda_lila_1_10  — Low-Income AND Low-Access (poverty/income requirement included)
  usda_la_1_10    — Low-Access only (same distance threshold, poverty requirement dropped)

Since both flags classify the identical set of units twice under different
rules, this is the paired/repeated-classification case from the decision
tree (Step 3a: "same unit classified more than once -> McNemar's test"),
NOT a chi-square test of independence between two different groups.

Requires: pip install statsmodels

Run this AFTER merge_sources.py.
"""

from pathlib import Path

import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
INPUT    = DATA_DIR / "nj_zip_features_v2.csv"

FLAG_WITH_POVERTY    = "usda_lila_1_10"   # Low-Income AND Low-Access
FLAG_WITHOUT_POVERTY = "usda_la_1_10"     # Low-Access only

CONTEXT_COLS = ["zip", "county", "municipality", "pct_poverty",
                "nearest_supermarket_miles"]


def section(title: str):
    print(f"\n{'─' * 72}\n{title}\n{'─' * 72}")


def main():
    df = pd.read_csv(INPUT, dtype={"zip": str})

    for col in (FLAG_WITH_POVERTY, FLAG_WITHOUT_POVERTY):
        if col not in df.columns:
            raise SystemExit(f"Column '{col}' not found in {INPUT.name}")

    # ── Step 0 / Step 3a precondition: same unit, classified twice ──────────
    section("Precondition check")
    sub = df[["zip", FLAG_WITH_POVERTY, FLAG_WITHOUT_POVERTY]].dropna()
    dupes = sub["zip"].duplicated().sum()
    n_dropped = len(df) - len(sub)
    print(f"  ZIPs with both flags present : {len(sub)}")
    print(f"  ZIPs dropped (missing either flag): {n_dropped}")
    if dupes:
        raise SystemExit(
            f"{dupes} ZIPs appear more than once — fix duplicate rows "
            f"before running McNemar's test."
        )
    print("  [OK] Each ZIP appears once — paired-classification "
          "precondition satisfied.")

    # ── Build the 2x2 paired contingency table ───────────────────────────────
    section("Contingency table (rows=WITH poverty req., cols=WITHOUT)")
    table = pd.crosstab(sub[FLAG_WITH_POVERTY], sub[FLAG_WITHOUT_POVERTY])
    # Ensure both 0/1 rows and columns exist even if a cell is empty
    for idx in (0, 1):
        if idx not in table.index:
            table.loc[idx] = 0
        if idx not in table.columns:
            table[idx] = 0
    table = table.sort_index()[sorted(table.columns)]
    print(table)

    # Off-diagonal cells are the ZIPs that CHANGE status between definitions
    only_with_poverty    = table.loc[1, 0]  # flagged WITH poverty req., NOT without
    only_without_poverty = table.loc[0, 1]  # flagged WITHOUT poverty req., NOT with
    both_flagged   = table.loc[1, 1]
    neither_flagged = table.loc[0, 0]

    print(f"\n  Flagged under BOTH definitions          : {both_flagged}")
    print(f"  Flagged under NEITHER definition         : {neither_flagged}")
    print(f"  Flagged ONLY with poverty requirement    : {only_with_poverty}  "
          f"(these ZIPs LOSE desert status if poverty req. is dropped)")
    print(f"  Flagged ONLY without poverty requirement : {only_without_poverty}  "
          f"(these ZIPs GAIN desert status if poverty req. is dropped)")

    # ── Small-cell check, same logic as chi-square Step 3a ──────────────────
    small_cells = (table < 5).sum().sum()
    use_exact = small_cells > 0
    if small_cells:
        print(f"\n  [NOTE] {small_cells} cell(s) < 5 — using EXACT McNemar's "
              f"test (binomial), not the chi-square approximation.")
    else:
        print(f"\n  All cells >= 5 — chi-square approximation would also be "
              f"valid, but exact is used regardless for correctness.")

    # ── McNemar's test ────────────────────────────────────────────────────────
    section("McNemar's test")
    result = mcnemar(table.values, exact=use_exact)
    print(f"  Statistic : {result.statistic:.4f}")
    print(f"  p-value   : {result.pvalue:.6f}")
    if result.pvalue < 0.05:
        print("  => Significant: removing the poverty requirement changes "
              "classification for a non-trivial, non-symmetric share of ZIPs "
              "(not just random noise in one direction).")
    else:
        print("  => Not significant: the ZIPs that gain vs. lose desert "
              "status under the two definitions are roughly balanced.")

    # ── Export the actual ZIPs that change status ────────────────────────────
    section("Exporting changed-status ZIP lists")
    cols_present = [c for c in CONTEXT_COLS if c in df.columns]

    lost_status = df[
        (df[FLAG_WITH_POVERTY] == 1) & (df[FLAG_WITHOUT_POVERTY] == 0)
    ][cols_present].copy()
    lost_status["change"] = "Loses desert status (poverty req. removed)"

    gained_status = df[
        (df[FLAG_WITH_POVERTY] == 0) & (df[FLAG_WITHOUT_POVERTY] == 1)
    ][cols_present].copy()
    gained_status["change"] = "Gains desert status (poverty req. removed)"

    changed = pd.concat([lost_status, gained_status], ignore_index=True)
    changed = changed.sort_values(["change", "pct_poverty"], ascending=[True, False])

    out_path = DATA_DIR / "poverty_requirement_status_changes.csv"
    changed.to_csv(out_path, index=False)
    print(f"  {len(changed)} ZIPs change status — saved -> {out_path}")

    # ── Summary stats on the changed ZIPs ────────────────────────────────────
    section("Profile of ZIPs that change status")
    if "pct_poverty" in changed.columns:
        profile = changed.groupby("change")["pct_poverty"].describe().round(2)
        print(profile.to_string())

        missing_poverty = changed["pct_poverty"].isna().sum()
        if missing_poverty:
            print(f"\n  [NOTE] {missing_poverty} changed-status ZIPs have missing "
                  f"pct_poverty and are excluded from the stats above.")

    # ── Are the 332 "gained" ZIPs meaningfully different from the 108 ───────
    # core LILA ZIPs, or just near-identical ZIPs that sat just under the
    # income cutoff? NOTE: there is no "lost status" group to compare against
    # here — usda_lila_1_10 is a strict subset of usda_la_1_10 by USDA's own
    # definition, so that cell is structurally 0, not an empirical finding.
    section("Are the 332 'gained' ZIPs meaningfully different from the 108 core LILA ZIPs?")
    core_lila = df[df[FLAG_WITH_POVERTY] == 1]
    gained_only = df[(df[FLAG_WITH_POVERTY] == 0) & (df[FLAG_WITHOUT_POVERTY] == 1)]

    from scipy.stats import mannwhitneyu
    compare_cols = ["pct_poverty", "usda_median_income", "nearest_supermarket_miles"]
    for col in compare_cols:
        if col not in df.columns:
            continue
        a, b = core_lila[col].dropna(), gained_only[col].dropna()
        if len(a) < 2 or len(b) < 2:
            print(f"  {col:<28} [SKIP] insufficient non-null data")
            continue
        stat, p = mannwhitneyu(a, b, alternative="two-sided")
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {col:<28} core LILA mean={a.mean():>8.2f}  "
              f"gained-only mean={b.mean():>8.2f}  p={p:.4f} {sig}")

    print("\n  Interpretation: if 'gained-only' ZIPs have meaningfully lower")
    print("  poverty/higher income than core LILA ZIPs, dropping the poverty")
    print("  requirement sweeps in a genuinely different, less-poor population")
    print("  under the food-desert label. If the two groups look similar,")
    print("  dropping the requirement mainly catches ZIPs just under the cutoff.")


if __name__ == "__main__":
    main()