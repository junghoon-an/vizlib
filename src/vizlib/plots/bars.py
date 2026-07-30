"""The ``bar`` value-counts chart.

A thin data-prep wrapper over the shared renderer in :mod:`vizlib.plots.hbar`;
``missing_bar`` is its sibling in :mod:`vizlib.plots.missing_bar`.
"""

from __future__ import annotations

import pandas as pd

from .data import _resolve_column
from .hbar import _hbar
from .options import Captions, ValueLabels


def bar(
    data,
    column: str | None = None,
    *,
    top: int = 15,
    sort: bool = True,
    highlight=None,
    as_percent: bool = False,
    labels: ValueLabels | None = None,
    max_label_chars: int | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    source: str | None = None,
    ax=None,
    **kwargs,
):
    """Draw a value-counts bar chart — the graphical twin of ``value_counts_bar``.

    Keeps the busiest ``top`` categories, folds the rest into ``"Other"``,
    sorts by frequency unless the column is an ordered categorical or
    ``sort=False``, and anchors the count axis at zero. By default each bar is
    labelled directly and the redundant value axis is hidden. ``highlight`` (a
    label or list) paints the chosen bars in the accent colour, the rest muted.
    Set ``as_percent=True`` to plot each category's share. Pass a
    :class:`~vizlib.plots.ValueLabels` as ``labels`` to tune the on-data labels
    (show/precision/fmt/padding). ``max_label_chars`` optionally ellipsizes very
    long names. Returns the ``Axes``; the input is never mutated.
    """
    labels = labels or ValueLabels()
    series = _resolve_column(data, column)
    ordered = isinstance(series.dtype, pd.CategoricalDtype) and series.dtype.ordered
    counts = series.value_counts(dropna=True)
    if counts.empty:
        raise ValueError("no data to plot")

    if ordered:
        counts = counts.reindex(list(series.cat.categories)).dropna()
    else:
        if top is not None and len(counts) > top:
            other = counts.iloc[top:].sum()
            counts = pd.concat([counts.iloc[:top], pd.Series({"Other": other})])
        if sort:
            counts = counts.sort_values(ascending=False)
    if as_percent:
        total = counts.sum()
        counts = counts / total * 100 if total else counts * 0.0

    plot = counts.iloc[::-1]  # biggest on top for a horizontal bar
    biggest = float(plot.max())
    name = series.name if series.name is not None else "value"
    value_strings = (labels.format(plot.to_numpy(), percent=as_percent)
                     if labels.show else None)
    return _hbar(
        plot, highlight=highlight, value_strings=value_strings,
        xlim=(0, biggest * 1.18 if biggest else 1),
        xlabel="% of total" if as_percent else "count", ylabel=str(name),
        captions=Captions(title if title is not None else f"Record count by {name}",
                          subtitle, source),
        labels=labels, max_label_chars=max_label_chars, ax=ax, **kwargs,
    )
