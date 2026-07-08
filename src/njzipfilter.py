"""
nj_zip_filter.py
=================
Canonical NJ ZIP/ZCTA filtering, built from the Census Gazetteer file
rather than any pipeline-internal crosswalk.

Why the Gazetteer and not nj_zip_crosswalk.csv:
  NJ is the only state ever assigned ZIP prefixes 07x and 08x (this is
  a fixed USPS allocation, not something that changes year to year).
  Filtering the *national* Gazetteer to those two prefixes gives a
  complete, authoritative list of every real NJ ZCTA with zero
  dependency on any other file in this pipeline -- so it can't
  silently inherit a bug from nj_zip_crosswalk.csv or any upstream
  join.

Usage:
    from nj_zip_filter import load_valid_nj_zips, filter_to_nj

    valid_nj_zips = load_valid_nj_zips(DATA_DIR / "2024_Gaz_zcta_national.txt")
    df = filter_to_nj(df, valid_nj_zips, label="clean_nj_zip_features")
"""

from pathlib import Path
import pandas as pd

NJ_ZIP_PREFIXES = ("07", "08")


def load_valid_nj_zips(gazetteer_path: Path) -> set:
    """
    Build the authoritative set of NJ ZCTA codes from the Census
    Gazetteer national ZCTA file (tab-delimited, GEOID column).
    """
    gaz = pd.read_csv(gazetteer_path, sep="\t", dtype={"GEOID": str})
    gaz.columns = gaz.columns.str.strip()
    gaz["GEOID"] = gaz["GEOID"].str.strip()
    nj = gaz[gaz["GEOID"].str.startswith(NJ_ZIP_PREFIXES)]
    valid = set(nj["GEOID"])
    print(f"  [nj_zip_filter] Loaded {len(valid)} authoritative NJ ZCTAs "
          f"from Gazetteer ({gazetteer_path.name})")
    return valid


def filter_to_nj(df: pd.DataFrame, valid_nj_zips: set,
                  zip_col: str = "zip", label: str = "") -> pd.DataFrame:
    """
    Drop any row whose zip_col value is not in valid_nj_zips.
    Never drops silently -- always prints what was removed.
    """
    zips = df[zip_col].astype(str).str.strip().str.zfill(5)
    mask_valid = zips.isin(valid_nj_zips)
    n_dropped = int((~mask_valid).sum())

    tag = f" [{label}]" if label else ""
    if n_dropped > 0:
        dropped_zips = sorted(zips[~mask_valid].unique())
        preview = dropped_zips[:20]
        more = f" ... (+{len(dropped_zips) - 20} more)" if len(dropped_zips) > 20 else ""
        print(f"  [nj_zip_filter]{tag} Dropping {n_dropped} out-of-state/invalid "
              f"rows: {preview}{more}")
    else:
        print(f"  [nj_zip_filter]{tag} No out-of-state rows found — nothing dropped.")

    return df.loc[mask_valid].reset_index(drop=True)