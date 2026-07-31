"""Numeric coercion helpers (pandas-only, never mutate their input).

Strip currency/thousands/``%`` decoration from numeric-looking columns and
decide which columns are cleanly numeric. Shared by ``load`` and the plots.
"""

from __future__ import annotations

import re

import pandas as pd

# A run of characters that decorate a number but aren't part of its value:
# currency symbols, thousands separators, percent signs and whitespace.
_NUMERIC_JUNK = re.compile(r"[,$£€%\s]")


def _coerce_numeric(series: pd.Series) -> pd.Series:
    """Coerce a Series to numeric, first stripping currency/thousands/``%``.

    Returns a new numeric Series (non-parsing entries become ``NaN``); the
    input is never mutated.
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
    parse as numbers after stripping decoration. Duplicate column names raise.
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
