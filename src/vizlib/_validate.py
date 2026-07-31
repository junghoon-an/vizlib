"""Type guards shared across the pandas-only core and the plotting layer."""

from __future__ import annotations

import pandas as pd


def _require_dataframe(obj: object) -> None:
    if not isinstance(obj, pd.DataFrame):
        raise TypeError(f"expected a pandas DataFrame, got {type(obj).__name__}")


def _require_series(obj: object) -> None:
    if not isinstance(obj, pd.Series):
        raise TypeError(f"expected a pandas Series, got {type(obj).__name__}")
