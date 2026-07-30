"""The ``box`` chart for spread and outlier inspection."""

from __future__ import annotations

import pandas as pd
import seaborn as sns

from .._coerce import _numeric_frame
from .._validate import _require_dataframe
from .axes import _rotate_xticklabels
from .chrome import _finish, _new_ax
from .colors import _base_color
from .data import _numeric_values


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

    With no ``column`` every numeric column becomes a box. Give ``column`` for
    a single distribution, and add ``by`` to split it across the levels of a
    categorical column — unordered groups are ordered by median (an ordered
    categorical keeps its natural order). Rows missing a plotted value are
    dropped. Returns the ``Axes``; the input is untouched.
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
