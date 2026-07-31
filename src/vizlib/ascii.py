"""Dependency-free ASCII charts: ``value_counts_bar`` and ``histogram``.

A quick look at a column without importing a plotting library. Both return a
plain string and never mutate their input.
"""

from __future__ import annotations

import pandas as pd

from ._validate import _require_series


def value_counts_bar(series: pd.Series, top: int = 10, width: int = 40) -> str:
    """ASCII horizontal bar chart of a Series' value counts.

    ``top`` limits how many categories are shown; ``width`` sets the length of
    the longest bar in characters.
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
    """ASCII histogram of a numeric Series over ``bins`` equal-width buckets.

    Missing values are ignored; raises when nothing is numeric.
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
