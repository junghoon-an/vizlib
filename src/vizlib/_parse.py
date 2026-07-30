"""NA-token and date-parsing helpers for ``load`` (pandas-only)."""

from __future__ import annotations

import re

import pandas as pd

# NA tokens recognised by ``load`` in addition to pandas' defaults.
_DEFAULT_NA = ["", "NA", "N/A", "null", "none", "unknown", "?"]

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


def _looks_datetime(name, series: pd.Series) -> bool:
    """Heuristic: does this object column hold date/datetime strings?"""
    sample = series.dropna().astype(str).head(20)
    if sample.empty:
        return False
    value_hits = sample.str.contains(_DATE_VALUE).mean()
    if value_hits >= 0.8:
        return True
    return bool(_DATE_NAME.search(str(name))) and value_hits > 0


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
