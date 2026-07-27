"""Distribution charts: ``hist``, ``distribution`` and ``box``."""

from __future__ import annotations

import pandas as pd
import seaborn as sns

from ..core import _numeric_frame, _require_dataframe
from .chrome import _finish, _new_ax, _rotate_xticklabels
from .data import _numeric_values
from .marks import _base_color


def hist(
    series: pd.Series,
    *,
    bins="auto",
    kde: bool = False,
    title: str | None = None,
    subtitle: str | None = None,
    source: str | None = None,
    ax=None,
    **kwargs,
):
    """Plot a histogram of a numeric Series — the graphical twin of ``histogram``.

    Non-numeric entries and missing values are dropped (never in place) and
    the count axis starts at zero. Set ``kde=True`` for a density overlay.
    ``title``/``subtitle``/``source`` add checklist-style captions. Returns
    the ``Axes``.
    """
    values = _numeric_values(series)
    ax = _new_ax(ax)
    kwargs.setdefault("color", _base_color())
    sns.histplot(x=values, bins=bins, kde=kde, ax=ax, **kwargs)
    ax.set_ylim(bottom=0)  # honest, zero-based count axis
    name = series.name if series.name is not None else "value"
    if title is None:
        title = f"Distribution of {name}"
    return _finish(ax, title=title, subtitle=subtitle, source=source,
                   xlabel=str(name), ylabel="count", grid_axis="y")


def distribution(
    series: pd.Series,
    *,
    title: str | None = None,
    subtitle: str | None = None,
    source: str | None = None,
    ax=None,
    **kwargs,
):
    """Show a distribution at a glance: histogram + KDE + rug.

    A convenience over :func:`hist` for a quick read on shape, spread and
    outliers. Missing/non-numeric values are dropped without mutating the
    input. Returns the ``Axes``.
    """
    values = _numeric_values(series)
    ax = _new_ax(ax)
    color = kwargs.pop("color", _base_color())
    sns.histplot(x=values, kde=True, ax=ax, color=color, **kwargs)
    sns.rugplot(x=values, ax=ax, color=color, height=0.04, alpha=0.6)
    ax.set_ylim(bottom=0)
    name = series.name if series.name is not None else "value"
    if title is None:
        title = f"Distribution of {name}"
    return _finish(ax, title=title, subtitle=subtitle, source=source,
                   xlabel=str(name), ylabel="count", grid_axis="y")


def box(
    df: pd.DataFrame,
    column: str | None = None,
    *,
    by: str | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    source: str | None = None,
    ax=None,
    **kwargs,
):
    """Draw a boxplot for spread and outlier inspection.

    With no ``column`` every numeric column becomes a box. Give ``column``
    for a single distribution, and add ``by`` to split it across the levels
    of a categorical column — unordered groups are ordered by median (an
    ordered categorical keeps its natural order). Rows missing a plotted
    value are dropped. Returns the ``Axes``; the input is untouched.
    """
    _require_dataframe(df)
    ax = _new_ax(ax)
    kwargs.setdefault("color", _base_color())  # one colour; position carries the group
    rotate = False

    if column is None:
        numeric = _numeric_frame(df)
        if numeric.empty:
            raise ValueError("no numeric columns to plot")
        sns.boxplot(data=numeric, ax=ax, **kwargs)
        default_title, xlabel, ylabel = "Value spread across numeric columns", "", "value"
        rotate = numeric.shape[1] > 4
    elif by is None:
        values = _numeric_values(df[column])
        sns.boxplot(y=values, ax=ax, **kwargs)
        default_title, xlabel, ylabel = f"Value spread for {column}", "", str(column)
    else:
        sub = df[[column, by]].dropna()
        if sub.empty:
            raise ValueError("no rows left after dropping missing values")
        by_series = sub[by]
        if isinstance(by_series.dtype, pd.CategoricalDtype) and by_series.dtype.ordered:
            order = [c for c in by_series.cat.categories if c in set(by_series)]
        else:  # unordered: order boxes by group median
            order = sub.groupby(by)[column].median().sort_values().index.tolist()
        sns.boxplot(data=sub, x=by, y=column, order=order, ax=ax, **kwargs)
        default_title, xlabel, ylabel = f"{column} across {by} groups", str(by), str(column)
        rotate = True

    ax = _finish(ax, title=title or default_title, subtitle=subtitle,
                 source=source, xlabel=xlabel, ylabel=ylabel, grid_axis="y")
    if rotate:
        _rotate_xticklabels(ax)
    return ax
