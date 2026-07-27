"""Horizontal bar charts: ``bar`` (value counts) and ``missing_bar``.

Both reserve explicit, worst-case left/right margins (see
:mod:`vizlib.plots.margins`) so the axes never move on a style switch and the
category/column names can never overlap the bars or their value labels.
"""

from __future__ import annotations

import pandas as pd

from ..core import _require_dataframe
from .chrome import _finish, _new_ax
from .data import _resolve_column
from .margins import _reserve_hbar_margins
from .marks import _bar_colors, _draw_value_labels, _ellipsize_yticklabels
from .theme import _THEME


def bar(
    data,
    column: str | None = None,
    *,
    top: int = 15,
    sort: bool = True,
    highlight=None,
    value_labels: bool = True,
    precision: int = 0,
    as_percent: bool = False,
    fmt: str | None = None,
    label_padding: int = 5,
    max_label_chars: int | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    source: str | None = None,
    ax=None,
    **kwargs,
):
    """Draw a value-counts bar chart — the graphical twin of ``value_counts_bar``.

    Keeps the busiest ``top`` categories, folds the rest into ``"Other"``,
    sorts by frequency (descending) unless the column is an ordered
    categorical or ``sort=False``, and anchors the count axis at zero. By
    default each bar is labelled directly (``value_labels=True``) and the
    now-redundant value axis and tick marks are hidden. ``highlight`` (a
    label or list of labels) paints the chosen bars in the accent color and
    the rest muted. Set ``as_percent=True`` to plot each category's share of
    the total; ``precision`` sets the decimals and ``fmt`` overrides the label
    format string. Labels always sit a fixed ``label_padding`` (points) past
    each bar's tip — clear of the y-axis category labels — and the infographic
    preset only makes them larger and bold. vizlib-owned figures reserve the
    left margin against the active fonts automatically, so long category names
    are not clipped; ``max_label_chars`` optionally ellipsizes very long names
    as a last resort (off by default — reserving space is preferred). Returns
    the ``Axes``; the input is never mutated. The default ``title`` is a
    neutral placeholder — override it with your finding.
    """
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
    # horizontal bars reserve explicit worst-case margins, not auto-layout
    ax = _new_ax(ax, min_height=0.4 * len(plot) + 1, constrained=False)
    kwargs.setdefault("color", _bar_colors(plot.index, highlight))
    if _THEME.get("linewidth"):
        kwargs.setdefault("edgecolor", "white")
    container = ax.barh([str(i) for i in plot.index], plot.to_numpy(), **kwargs)
    _ellipsize_yticklabels(ax, max_label_chars)
    biggest = float(plot.max())
    # zero-based, with right headroom so edge labels never clip
    ax.set_xlim(0, biggest * 1.18 if biggest else 1)

    name = series.name if series.name is not None else "value"
    grid_axis, xlabel = "x", ("% of total" if as_percent else "count")
    value_strings = []
    if value_labels:
        default_fmt = f"%.{precision}f%%" if as_percent else f"%.{precision}f"
        used_fmt = fmt or default_fmt
        value_strings = [used_fmt % v for v in plot.to_numpy()]
        _draw_value_labels(ax, container, fmt=used_fmt, padding=label_padding)
        ax.xaxis.set_visible(False)                 # value axis is redundant now
        ax.spines["bottom"].set_visible(False)
        grid_axis, xlabel = None, None
    ax.tick_params(length=0)  # no tick marks on bar-type charts
    if title is None:
        title = f"Record count by {name}"
    ax = _finish(ax, title=title, subtitle=subtitle, source=source,
                 xlabel=xlabel, ylabel=str(name), grid_axis=grid_axis)
    _reserve_hbar_margins(ax, value_strings, n_bars=len(plot),
                          max_label_chars=max_label_chars)
    return ax


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

    The graphical twin of :func:`vizlib.core.missing_values`. The axis runs a
    full 0–100 % so magnitudes read honestly; by default each bar is labelled
    directly and the value axis/tick marks are hidden. ``highlight`` accents
    the chosen column(s); ``precision`` sets the percent decimals and ``fmt``
    overrides the label format. Labels sit a fixed ``label_padding`` (points)
    past each bar's tip — clear of the column names on the left — and the
    infographic preset only makes them larger and bold. The left margin is
    reserved against the active fonts automatically so long column names are
    not clipped; ``max_label_chars`` optionally ellipsizes very long names as a
    last resort (off by default). Returns the ``Axes``; the input is untouched.
    """
    _require_dataframe(df)
    if df.shape[1] == 0:
        raise ValueError("no columns to plot")
    n = len(df)
    pct = (df.isna().sum() / n * 100) if n else df.isna().sum() * 0.0
    pct = pct.sort_values(ascending=False)

    plot = pct.iloc[::-1]  # biggest on top
    # horizontal bars reserve explicit worst-case margins, not auto-layout
    ax = _new_ax(ax, min_height=0.4 * len(plot) + 1, constrained=False)
    kwargs.setdefault("color", _bar_colors(plot.index, highlight))
    if _THEME.get("linewidth"):
        kwargs.setdefault("edgecolor", "white")
    container = ax.barh([str(i) for i in plot.index], plot.to_numpy(), **kwargs)
    _ellipsize_yticklabels(ax, max_label_chars)

    grid_axis, xlabel = "x", "% missing"
    value_strings = []
    if value_labels:
        labels = [(fmt % v) if fmt else f"{v:.{precision}f}%" for v in plot.to_numpy()]
        value_strings = labels
        _draw_value_labels(ax, container, labels=labels, padding=label_padding)
        # right headroom so edge labels clear the axis; scale hidden anyway
        ax.set_xlim(0, max(float(plot.max()) * 1.12, 1.0))
        ax.xaxis.set_visible(False)
        ax.spines["bottom"].set_visible(False)
        grid_axis, xlabel = None, None
    else:
        ax.set_xlim(0, 100)  # honest 0–100 scale when the axis is shown
    ax.tick_params(length=0)
    if title is None:
        title = "Share of missing values by column"
    ax = _finish(ax, title=title, subtitle=subtitle, source=source,
                 xlabel=xlabel, ylabel="column", grid_axis=grid_axis)
    _reserve_hbar_margins(ax, value_strings, n_bars=len(plot),
                          max_label_chars=max_label_chars)
    return ax
