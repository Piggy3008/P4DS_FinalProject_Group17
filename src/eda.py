"""Helpers for the data-exploration notebook (01_collection_exploration).

Every function here returns plain DataFrames/Series, so the notebook keeps
the decisions and the interpretation visible.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_FILE = "Electric_Vehicle_Population_Data.csv"

# Values that mean "not researched / unknown", not a real measurement.
PLACEHOLDER_ZERO_COLS = ["Electric Range", "Base MSRP"]

# Geographic codes stored as float only because of NaN; they are identifiers.
ID_LIKE_COLS = ["Postal Code", "Legislative District", "2020 Census Tract"]


def load_raw(data_dir: str | Path = "../data/raw") -> pd.DataFrame:
    """Read the raw EV population CSV without modifying it.

    `pd.read_csv` handles quoted fields that contain commas (e.g. some
    `Electric Utility` values), so never split rows on commas manually.
    """
    return pd.read_csv(Path(data_dir) / RAW_FILE)


def parse_point(wkt: pd.Series) -> pd.DataFrame:
    """Parse WKT strings 'POINT (lon lat)' into float `lon` and `lat` columns.

    Missing or malformed strings become NaN.
    """
    coords = wkt.str.extract(r"POINT\s*\(\s*(-?[\d.]+)\s+(-?[\d.]+)\s*\)")
    coords.columns = ["lon", "lat"]
    return coords.astype("float64")


def missing_report(df: pd.DataFrame, placeholder_zero_cols=PLACEHOLDER_ZERO_COLS) -> pd.DataFrame:
    """Count explicit NaNs and placeholder zeros per column.

    Returns one row per column with `n_null`, `pct_null`, `n_placeholder_zero`,
    `pct_placeholder_zero`, sorted by total missing-like share.
    """
    n = len(df)
    out = pd.DataFrame({"n_null": df.isna().sum()})
    out["n_placeholder_zero"] = 0
    for col in placeholder_zero_cols:
        if col in df:
            out.loc[col, "n_placeholder_zero"] = int((df[col] == 0).sum())
    out["pct_null"] = (out["n_null"] / n * 100).round(2)
    out["pct_placeholder_zero"] = (out["n_placeholder_zero"] / n * 100).round(2)
    total = out["n_null"] + out["n_placeholder_zero"]
    return out.loc[total.sort_values(ascending=False).index,
                   ["n_null", "pct_null", "n_placeholder_zero", "pct_placeholder_zero"]]


def outlier_summary(s: pd.Series, z_thresh: float = 3.0) -> pd.Series:
    """Summarise one numeric series: centre, spread, shape, IQR and z-score outliers."""
    s = s.dropna()
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    z = (s - s.mean()) / s.std(ddof=1)
    return pd.Series({
        "n": len(s),
        "mean": s.mean(),
        "median": s.median(),
        "std": s.std(ddof=1),
        "skew": s.skew(),
        "min": s.min(),
        "q1": q1,
        "q3": q3,
        "max": s.max(),
        "iqr_low_fence": lo,
        "iqr_high_fence": hi,
        "n_iqr_outliers": int(((s < lo) | (s > hi)).sum()),
        f"n_|z|>{z_thresh:g}": int((z.abs() > z_thresh).sum()),
    })


def label_collisions(s: pd.Series) -> pd.DataFrame:
    """Find labels that differ only by case / surrounding or repeated whitespace.

    Returns the groups that have more than one raw spelling (empty if consistent).
    """
    vals = pd.Series(s.dropna().unique())
    key = vals.str.strip().str.replace(r"\s+", " ", regex=True).str.upper()
    groups = vals.groupby(key).agg(list)
    groups = groups[groups.str.len() > 1]
    return groups.rename("raw_spellings").rename_axis("normalised").reset_index()


def utility_count(s: pd.Series) -> pd.Series:
    """Number of utilities listed in each `Electric Utility` string (split on '|')."""
    return s.str.split("|").str.len().astype("Int64")


def top_share(s: pd.Series, k: int = 10) -> pd.DataFrame:
    """Top-k value counts with percentage and cumulative percentage."""
    vc = s.value_counts(dropna=False)
    out = vc.head(k).to_frame("count")
    out["pct"] = (out["count"] / len(s) * 100).round(2)
    out["cum_pct"] = out["pct"].cumsum().round(2)
    return out


def rare_categories(s: pd.Series, min_share: float = 0.001) -> pd.Series:
    """Categories whose share of rows is below `min_share` (default 0.1%)."""
    share = s.value_counts(normalize=True)
    return share[share < min_share]


__all__ = [
    "RAW_FILE", "PLACEHOLDER_ZERO_COLS", "ID_LIKE_COLS", "load_raw", "parse_point",
    "missing_report", "outlier_summary", "label_collisions", "utility_count",
    "top_share", "rare_categories",
]
