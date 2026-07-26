"""Core exploratory-data-analysis helpers for vizlib.

Every function takes a pandas object and returns a pandas object (or a
plain string for the ASCII charts), so results compose naturally with the
rest of your notebook. Nothing here mutates its input.
"""

from __future__ import annotations

import pandas as pd

__all__ = [
    "summarize",
    "missing_values",
    "numeric_summary",
    "value_counts_bar",
    "histogram",
]


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Return a one-row-per-column overview of a DataFrame.

    Columns of the result: ``dtype``, ``non_null``, ``nulls``,
    ``null_pct`` and ``unique``. This is the fastest way to get a feel
    for a new dataset.
    """
    _require_dataframe(df)
    n = len(df)
    rows = {
        "dtype": df.dtypes.astype(str),
        "non_null": df.notna().sum(),
        "nulls": df.isna().sum(),
        "null_pct": (df.isna().sum() / n * 100).round(2) if n else 0.0,
        "unique": df.nunique(dropna=True),
    }
    out = pd.DataFrame(rows)
    out.index.name = "column"
    return out


def missing_values(df: pd.DataFrame, only_missing: bool = True) -> pd.DataFrame:
    """Report missing-value counts per column, largest first.

    By default only columns that actually contain missing values are
    returned. Pass ``only_missing=False`` to see every column.
    """
    _require_dataframe(df)
    n = len(df)
    counts = df.isna().sum()
    out = pd.DataFrame(
        {
            "nulls": counts,
            "null_pct": (counts / n * 100).round(2) if n else 0.0,
        }
    )
    if only_missing:
        out = out[out["nulls"] > 0]
    out = out.sort_values("nulls", ascending=False)
    out.index.name = "column"
    return out


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Describe the numeric columns of a DataFrame.

    A thin, friendly wrapper around ``DataFrame.describe`` that returns
    one row per numeric column (transposed) and raises a clear error when
    there is nothing numeric to summarize.
    """
    _require_dataframe(df)
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        raise ValueError("no numeric columns to summarize")
    out = numeric.describe().T
    out.index.name = "column"
    return out


def value_counts_bar(series: pd.Series, top: int = 10, width: int = 40) -> str:
    """Return an ASCII horizontal bar chart of a Series' value counts.

    Great for a quick look at a categorical column without importing a
    plotting library. ``top`` limits how many categories are shown and
    ``width`` sets the length of the longest bar in characters.
    """
    _require_series(series)
    counts = series.value_counts(dropna=False).head(top)
    if counts.empty:
        return "(no data)"
    label_w = max(len(str(idx)) for idx in counts.index)
    biggest = int(counts.iloc[0]) or 1
    lines = []
    for label, count in counts.items():
        bar = "#" * max(1, round(int(count) / biggest * width))
        lines.append(f"{str(label):<{label_w}} | {bar} {int(count)}")
    return "\n".join(lines)


def histogram(series: pd.Series, bins: int = 10, width: int = 40) -> str:
    """Return an ASCII histogram of a numeric Series.

    The value range is split into ``bins`` equal-width buckets and each
    bucket is drawn as a bar. Missing values are ignored.
    """
    _require_series(series)
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        raise ValueError("no numeric values to plot")
    counts = pd.cut(values, bins=bins).value_counts().sort_index()
    biggest = int(counts.max()) or 1
    edge_w = max(len(f"{iv.left:.2f}") for iv in counts.index)
    lines = []
    for interval, count in counts.items():
        bar = "#" * round(int(count) / biggest * width)
        left = f"{interval.left:.2f}".rjust(edge_w)
        right = f"{interval.right:.2f}".rjust(edge_w)
        lines.append(f"[{left}, {right}) | {bar} {int(count)}")
    return "\n".join(lines)


def _require_dataframe(obj: object) -> None:
    if not isinstance(obj, pd.DataFrame):
        raise TypeError(f"expected a pandas DataFrame, got {type(obj).__name__}")


def _require_series(obj: object) -> None:
    if not isinstance(obj, pd.Series):
        raise TypeError(f"expected a pandas Series, got {type(obj).__name__}")
