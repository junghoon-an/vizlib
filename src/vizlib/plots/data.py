"""Small data-wrangling helpers shared by the plotting functions.

These lean on the pandas-only cleaners in :mod:`vizlib._coerce` so the plots
accept numeric-looking string columns and messy inputs without the caller
tidying first. Nothing here mutates its input.
"""

from __future__ import annotations

import pandas as pd

from .._coerce import _coerce_numeric
from .._validate import _require_series

# Row count above which scatter auto-samples (deterministically) to stay
# responsive. Override or disable with an explicit ``sample=``.
_AUTO_SAMPLE_SCATTER = 20_000


def _resolve_column(data, column: str | None) -> pd.Series:
    """Return the Series to plot from a Series or a (DataFrame, column) pair."""
    if isinstance(data, pd.DataFrame):
        if column is None:
            raise ValueError("column is required when data is a DataFrame")
        _require_columns(data, [column])
        return data[column]
    _require_series(data)
    return data


def _require_columns(df: pd.DataFrame, columns) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"column(s) not found in DataFrame: {missing}")


def _numeric_values(series: pd.Series) -> pd.Series:
    """Coerce a Series to numeric and drop NaN.

    Strips currency symbols, thousands separators and stray ``%`` first (via
    the shared ``core._coerce_numeric``), so a numeric-looking string column
    plots without manual cleaning. Raises when nothing parses.
    """
    _require_series(series)
    values = _coerce_numeric(series).dropna()
    if values.empty:
        raise ValueError("no numeric values to plot")
    return values


def _maybe_sample(frame: pd.DataFrame, sample, auto_threshold: int, random_state: int):
    """Return a reproducible row subset, honouring an explicit or auto cap."""
    n = sample if sample is not None else (
        auto_threshold if len(frame) > auto_threshold else None
    )
    if n is not None and n < len(frame):
        return frame.sample(n=n, random_state=random_state)
    return frame
