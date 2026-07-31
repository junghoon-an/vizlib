"""The ``missing_bar`` chart — percent missing per column.

A thin data-prep wrapper over the shared renderer in :mod:`vizlib.plots.hbar`.
"""

from __future__ import annotations

import pandas as pd

from .._validate import _require_dataframe
from .hbar import _hbar
from .options import Captions, ValueLabels


def missing_bar(
    df: pd.DataFrame,
    *,
    highlight=None,
    labels: ValueLabels | None = None,
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
    chosen column(s); pass a :class:`~vizlib.plots.ValueLabels` as ``labels`` to
    tune the on-data labels (defaults to one-decimal percentages). Returns the
    ``Axes``; the input is untouched.
    """
    labels = labels or ValueLabels(precision=1)
    _require_dataframe(df)
    if df.shape[1] == 0:
        raise ValueError("no columns to plot")
    n = len(df)
    pct = (df.isna().sum() / n * 100) if n else df.isna().sum() * 0.0
    plot = pct.sort_values(ascending=False).iloc[::-1]  # biggest on top

    if labels.show:
        value_strings = labels.format(plot.to_numpy(), percent=True)
        xlim = (0, max(float(plot.max()) * 1.12, 1.0))  # headroom for edge labels
    else:
        value_strings, xlim = None, (0, 100)  # honest 0–100 scale when axis shown
    return _hbar(
        plot, highlight=highlight, value_strings=value_strings, xlim=xlim,
        xlabel="% missing", ylabel="column",
        captions=Captions(title if title is not None
                          else "Share of missing values by column", subtitle, source),
        labels=labels, max_label_chars=max_label_chars, ax=ax, **kwargs,
    )
