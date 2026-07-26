"""Core exploratory-data-analysis helpers for vizlib.

Every function takes a pandas object and returns a pandas object (or a
plain string for the ASCII charts), so results compose naturally with the
rest of your notebook. Nothing here mutates its input.
"""

from __future__ import annotations

import re

import pandas as pd

__all__ = [
    "load",
    "summarize",
    "missing_values",
    "numeric_summary",
    "value_counts_bar",
    "histogram",
]

# NA tokens recognised by ``load`` in addition to pandas' defaults.
_DEFAULT_NA = ["", "NA", "N/A", "null", "none", "unknown", "?"]

# A run of characters that decorate a number but aren't part of its value:
# currency symbols, thousands separators, percent signs and whitespace.
_NUMERIC_JUNK = re.compile(r"[,$£€%\s]")

# Value/name hints used to spot date-like object columns.
_DATE_VALUE = re.compile(r"\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{2,4}")
_DATE_NAME = re.compile(r"(?:^|_)(date|time|timestamp|datetime)|_(?:at|on)$", re.I)


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
    numeric = _numeric_frame(df)
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


def load(
    path,
    *,
    parse_dates="auto",
    numeric="auto",
    na_values=None,
    sample: int | None = None,
    random_state: int = 0,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """Read a real-world CSV into a clean, plot-ready DataFrame.

    A thin, pandas-only wrapper around :func:`pandas.read_csv` that does the
    tidying you'd otherwise do by hand:

    - Treats a generous set of NA tokens as missing (``""``, ``NA``,
      ``N/A``, ``null``, ``none``, ``unknown``, ``?``, case-insensitively),
      merged with any ``na_values`` you pass.
    - Falls back from ``utf-8`` to ``latin-1`` on a decode error.
    - ``numeric="auto"`` coerces object columns that are *mostly* numeric
      once currency symbols, thousands separators and stray ``%`` are
      stripped — leaving ID/text columns alone. Pass ``False`` to skip, or a
      list of column names to coerce exactly those.
    - ``parse_dates="auto"`` parses object columns that look like dates.
      Pass ``False`` to skip, or a list to parse exactly those.
    - ``sample`` returns a reproducible random subset (``random_state``),
      handy for very large files.

    The input file is never modified and a fresh DataFrame is returned.
    """
    na = _merge_na_values(na_values)
    read_csv_kwargs.setdefault("na_values", na)
    read_csv_kwargs.setdefault("keep_default_na", True)
    try:
        df = pd.read_csv(path, **read_csv_kwargs)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1", **read_csv_kwargs)

    if numeric == "auto":
        df = _auto_numeric(df)
    elif numeric is False:
        pass
    else:  # explicit list of columns
        for col in numeric:
            df[col] = _coerce_numeric(df[col])

    if parse_dates == "auto":
        df = _auto_dates(df)
    elif parse_dates not in (False, None):
        for col in parse_dates:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if sample is not None and sample < len(df):
        df = df.sample(n=sample, random_state=random_state).reset_index(drop=True)
    return df


def _merge_na_values(na_values) -> list:
    """Combine the default NA tokens with user tokens, case-insensitively."""
    tokens = set(_DEFAULT_NA)
    if na_values is not None:
        extra = [na_values] if isinstance(na_values, str) else list(na_values)
        tokens.update(extra)
    for token in list(tokens):
        if isinstance(token, str) and token:
            tokens.update({token.lower(), token.upper(), token.capitalize()})
    return list(tokens)


def _coerce_numeric(series: pd.Series) -> pd.Series:
    """Coerce a Series to numeric, first stripping currency/thousands/``%``.

    Returns a new numeric Series (non-parsing entries become ``NaN``); the
    input is never mutated. This is the shared cleaner reused by the plots.
    """
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = series.astype("string").str.strip().str.replace(
        _NUMERIC_JUNK, "", regex=True
    )
    cleaned = cleaned.replace("", pd.NA)
    numeric = pd.to_numeric(cleaned, errors="coerce")
    # Normalise pandas' nullable Int64/Float64 to numpy float so downstream
    # matplotlib/seaborn calls receive a plain ndarray.
    if pd.api.types.is_extension_array_dtype(numeric):
        numeric = numeric.astype("float64")
    return numeric


def _numeric_frame(df: pd.DataFrame, min_frac: float = 0.8) -> pd.DataFrame:
    """Return the columns that are numeric or cleanly numeric-coercible.

    A column qualifies when at least ``min_frac`` of its non-null values
    parse as numbers after stripping currency/thousands/``%``. Guards
    against duplicate column names with a clear error.
    """
    if df.columns.duplicated().any():
        raise ValueError("duplicate column names are not supported")
    cols: dict = {}
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            cols[col] = pd.to_numeric(series, errors="coerce")
            continue
        if pd.api.types.is_datetime64_any_dtype(series):
            continue
        coerced = _coerce_numeric(series)
        non_null = int(series.notna().sum())
        if non_null and coerced.notna().sum() / non_null >= min_frac:
            cols[col] = coerced
    return pd.DataFrame(cols)


def _auto_numeric(df: pd.DataFrame, min_frac: float = 0.9) -> pd.DataFrame:
    """Return a copy with mostly-numeric object columns coerced to numbers."""
    out = df.copy()
    for col in out.columns:
        series = out[col]
        if not (series.dtype == object or pd.api.types.is_string_dtype(series)):
            continue
        coerced = _coerce_numeric(series)
        non_null = int(series.notna().sum())
        if non_null and coerced.notna().sum() / non_null >= min_frac:
            out[col] = coerced
    return out


def _auto_dates(df: pd.DataFrame, min_frac: float = 0.8) -> pd.DataFrame:
    """Return a copy with date-like object columns parsed to datetimes."""
    out = df.copy()
    for col in out.columns:
        series = out[col]
        if not (series.dtype == object or pd.api.types.is_string_dtype(series)):
            continue
        if not _looks_datetime(col, series):
            continue
        parsed = pd.to_datetime(series, errors="coerce")
        non_null = int(series.notna().sum())
        if non_null and parsed.notna().sum() / non_null >= min_frac:
            out[col] = parsed
    return out


def _looks_datetime(name, series: pd.Series) -> bool:
    """Heuristic: does this object column hold date/datetime strings?"""
    sample = series.dropna().astype(str).head(20)
    if sample.empty:
        return False
    value_hits = sample.str.contains(_DATE_VALUE).mean()
    if value_hits >= 0.8:
        return True
    return bool(_DATE_NAME.search(str(name))) and value_hits > 0


def _require_dataframe(obj: object) -> None:
    if not isinstance(obj, pd.DataFrame):
        raise TypeError(f"expected a pandas DataFrame, got {type(obj).__name__}")


def _require_series(obj: object) -> None:
    if not isinstance(obj, pd.Series):
        raise TypeError(f"expected a pandas Series, got {type(obj).__name__}")
