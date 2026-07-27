"""Shared pandas-only cleaning, dtype-coercion and validation helpers.

These back the public :mod:`vizlib.core` functions (and the plotting layer's
input handling). Everything here is pandas-only and never mutates its input,
keeping ``import vizlib`` free of heavy dependencies.
"""

from __future__ import annotations

import re

import pandas as pd

# NA tokens recognised by ``load`` in addition to pandas' defaults.
_DEFAULT_NA = ["", "NA", "N/A", "null", "none", "unknown", "?"]

# A run of characters that decorate a number but aren't part of its value:
# currency symbols, thousands separators, percent signs and whitespace.
_NUMERIC_JUNK = re.compile(r"[,$£€%\s]")

# Value/name hints used to spot date-like object columns.
_DATE_VALUE = re.compile(r"\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{2,4}")
_DATE_NAME = re.compile(r"(?:^|_)(date|time|timestamp|datetime)|_(?:at|on)$", re.I)


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
