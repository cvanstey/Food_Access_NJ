"""
pipeline_utils.py
==================
Shared helpers used across the Food_Access_NJ pipeline scripts.

Extracted from duplicated code found independently in:
  - section():        01_load_data.py, 04_model.py
  - normalize_zip():  02_merge_sources.py (defined),
                       01_load_data.py + 04_model.py (equivalent logic inlined)

Import what you need:
    from pipeline_utils import section, normalize_zip
"""

import pandas as pd


def section(title: str) -> None:
    """Print a formatted section header to the console."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def normalize_zip(series: pd.Series) -> pd.Series:
    """Clean and zero-pad a column of ZIP codes to 5-digit strings.

    Strips whitespace, then zero-pads to 5 digits. This is the
    canonical version — 01_load_data.py previously skipped the
    .str.strip() step, which meant ZIPs with stray whitespace
    (e.g. from PDF-extracted sources like the NJEDA deck) would
    silently fail to match ZIPs normalized elsewhere in the pipeline.
    """
    return series.astype(str).str.strip().str.zfill(5)

def load_features_and_scores(data_dir, features_filename="nj_zip_features_v5.csv",
                              scores_filename="nj_zip_scores_1.csv"):
    """
    Load the base feature file and merge in model/prediction/USDA columns
    from the scores file. Single source of truth for this join — both
    analytics_descriptive.py and the report generator should call this
    instead of re-implementing it.
    """
    features_file = data_dir / features_filename
    scores_file   = data_dir / scores_filename

    df = pd.read_csv(features_file, dtype={"zip": str})
    df["zip"] = df["zip"].str.zfill(5)
    print(f"  Features : {len(df)} rows × {len(df.columns)} columns  ({features_file.name})")

    if scores_file.exists():
        scores = pd.read_csv(scores_file, dtype={"zip": str})
        scores["zip"] = scores["zip"].str.zfill(5)

        pred_cols  = [c for c in scores.columns if c.startswith("predicted_") or c.startswith("typo_prob_")]
        model_cols = [c for c in ["desert_probability", "predicted_desert",
                                   "predicted_desert_tuned", "predicted_typology"]
                      if c in scores.columns]
        usda_cols  = [c for c in scores.columns if c.startswith("usda_")]

        cols_to_merge = ["zip"] + model_cols + pred_cols + usda_cols
        cols_to_merge = [c for c in cols_to_merge if c in scores.columns]
        new_cols      = [c for c in cols_to_merge if c not in df.columns or c == "zip"]

        df = df.merge(scores[new_cols], on="zip", how="left")
        print(f"  Scores   : merged {len(new_cols)-1} model columns from {scores_file.name}")
    else:
        print(f"  Scores   : {scores_file.name} not found — predicted columns unavailable")

    print(f"  Combined : {len(df)} rows × {len(df.columns)} columns")
    return df

