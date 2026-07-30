"""The ``line`` chart, including gradient and stacked area fills."""

from __future__ import annotations

import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns

from .._coerce import _coerce_numeric
from .._parse import _looks_datetime
from .._validate import _require_dataframe
from .annotate import _draw_callouts, _gradient_fill, _to_num
from .area import _hued_lines, _stacked_area
from .axes import _rotate_xticklabels
from .chrome import _finish, _new_ax
from .colors import _base_color
from .data import _require_columns
from .theme import _THEME


def line(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    hue: str | None = None,
    area: bool = False,
    stack: bool = False,
    annotations=None,
    title: str | None = None,
    subtitle: str | None = None,
    source: str | None = None,
    ax=None,
    **kwargs,
):
    """Draw a line plot for ordered or time-series data.

    ``y`` is coerced to numeric; a date-like ``x`` is parsed to datetimes and
    given concise date ticks. Rows are sorted by ``x`` and rows missing a
    plotted value are dropped (without mutating the caller's frame).

    - ``area=True`` fills under the line(s) with a vertical gradient.
    - ``area=True, stack=True`` with ``hue`` draws a **stacked** area.
    - ``annotations`` attaches leader-line callouts.

    With ``hue`` (and no swatch legend) each line is labelled at its right end.
    Returns the ``Axes``.
    """
    _require_dataframe(df)
    cols = [x, y] + ([hue] if hue else [])
    _require_columns(df, cols)
    sub = df[cols].dropna().copy()
    sub[y] = _coerce_numeric(sub[y])
    if sub[x].dtype == object and _looks_datetime(x, sub[x]):
        sub[x] = pd.to_datetime(sub[x], errors="coerce")
    sub = sub.dropna(subset=[x, y]).sort_values(x)
    if sub.empty:
        raise ValueError("no rows left after dropping missing values")

    ax = _new_ax(ax)
    if _THEME.get("linewidth"):
        kwargs.setdefault("linewidth", _THEME["linewidth"])

    if area and stack and hue:
        _stacked_area(ax, sub, x, y, hue)
    elif hue:
        _hued_lines(ax, sub, x, y, hue, area, kwargs)
    else:
        color = _base_color()
        sns.lineplot(data=sub, x=x, y=y, color=color, ax=ax, **kwargs)
        if area:
            _gradient_fill(ax, _to_num(sub[x]), sub[y].to_numpy(float), color)

    if pd.api.types.is_datetime64_any_dtype(sub[x]):  # readable date ticks
        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    _draw_callouts(ax, annotations, sub[x].tolist(), sub[y].tolist())
    if title is None:
        title = f"{y} over {x}"
    ax = _finish(ax, title=title, subtitle=subtitle, source=source,
                 xlabel=str(x), ylabel=str(y), grid_axis="both")
    _rotate_xticklabels(ax)
    return ax
