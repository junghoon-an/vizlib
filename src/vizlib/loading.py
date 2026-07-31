"""``load`` — read a messy real-world CSV into a clean, plot-ready DataFrame."""

from __future__ import annotations

import pandas as pd

from ._coerce import _auto_numeric, _coerce_numeric
from ._parse import _auto_dates, _merge_na_values


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

    A pandas-only wrapper around :func:`pandas.read_csv` that does the tidying
    you'd otherwise do by hand:

    - Treats a generous set of NA tokens as missing (``""``, ``NA``, ``N/A``,
      ``null``, ``none``, ``unknown``, ``?``, case-insensitively), merged with
      any ``na_values`` you pass.
    - Falls back from ``utf-8`` to ``latin-1`` on a decode error.
    - ``numeric="auto"`` coerces object columns that are *mostly* numeric once
      currency symbols, thousands separators and stray ``%`` are stripped —
      leaving ID/text columns alone. Pass ``False`` to skip, or a list of
      column names to coerce exactly those.
    - ``parse_dates="auto"`` parses object columns that look like dates. Pass
      ``False`` to skip, or a list to parse exactly those.
    - ``sample`` returns a reproducible random subset (``random_state``).

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
    elif numeric is not False:  # explicit list of columns
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
