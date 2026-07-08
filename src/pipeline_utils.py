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
