"""DataFrame overview helpers: ``summarize``, ``missing_values``, ``numeric_summary``.

Each takes a pandas DataFrame and returns a DataFrame; nothing mutates its
input, so results compose naturally in a notebook.
"""

from __future__ import annotations

import pandas as pd

from ._coerce import _numeric_frame
from ._validate import _require_dataframe


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """One-row-per-column overview: ``dtype``, ``non_null``, ``nulls``,
    ``null_pct`` and ``unique`` — the fastest feel for a new dataset."""
    _require_dataframe(df)
    n = len(df)
    out = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "non_null": df.notna().sum(),
        "nulls": df.isna().sum(),
        "null_pct": (df.isna().sum() / n * 100).round(2) if n else 0.0,
        "unique": df.nunique(dropna=True),
    })
    out.index.name = "column"
    return out


def missing_values(df: pd.DataFrame, only_missing: bool = True) -> pd.DataFrame:
    """Missing-value counts and percentages per column, largest first.

    By default only columns that contain missing values are returned; pass
    ``only_missing=False`` to see every column.
    """
    _require_dataframe(df)
    n = len(df)
    counts = df.isna().sum()
    out = pd.DataFrame({
        "nulls": counts,
        "null_pct": (counts / n * 100).round(2) if n else 0.0,
    })
    if only_missing:
        out = out[out["nulls"] > 0]
    out = out.sort_values("nulls", ascending=False)
    out.index.name = "column"
    return out


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """``describe()`` for the numeric (or numeric-coercible) columns, transposed.

    Raises a clear error when there is nothing numeric to summarize.
    """
    _require_dataframe(df)
    numeric = _numeric_frame(df)
    if numeric.empty:
        raise ValueError("no numeric columns to summarize")
    out = numeric.describe().T
    out.index.name = "column"
    return out
