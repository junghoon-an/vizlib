"""The ``missing_bar`` chart — percent missing per column.

A thin data-prep wrapper over the shared renderer in :mod:`vizlib.plots.hbar`.
"""

from __future__ import annotations

import pandas as pd

from .._validate import _require_dataframe
from .hbar import _hbar


def missing_bar(
    df: pd.DataFrame,
    *,
    highlight=None,
    value_labels: bool = True,
    precision: int = 1,
    fmt: str | None = None,
    label_padding: int = 5,
    max_label_chars: int | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    source: str | None = None,
    ax=None,
    **kwargs,
):
    """Bar chart of the percentage of missing values per column, largest first.

    The graphical twin of :func:`vizlib.missing_values`. With labels hidden the
    axis runs a full 0–100 % so magnitudes read honestly; by default each bar
    is labelled directly and the value axis is hidden. ``highlight`` accents the
    chosen column(s); ``precision``/``fmt`` control the label. Returns the
    ``Axes``; the input is untouched.
    """
    _require_dataframe(df)
    if df.shape[1] == 0:
        raise ValueError("no columns to plot")
    n = len(df)
    pct = (df.isna().sum() / n * 100) if n else df.isna().sum() * 0.0
    plot = pct.sort_values(ascending=False).iloc[::-1]  # biggest on top

    value_strings = None
    if value_labels:
        value_strings = [(fmt % v) if fmt else f"{v:.{precision}f}%"
                         for v in plot.to_numpy()]
        xlim = (0, max(float(plot.max()) * 1.12, 1.0))  # headroom for edge labels
    else:
        xlim = (0, 100)  # honest 0–100 scale when the axis is shown
    return _hbar(
        plot, highlight=highlight, value_strings=value_strings, xlim=xlim,
        xlabel="% missing", ylabel="column",
        title=title if title is not None else "Share of missing values by column",
        subtitle=subtitle, source=source, label_padding=label_padding,
        max_label_chars=max_label_chars, ax=ax, **kwargs,
    )
