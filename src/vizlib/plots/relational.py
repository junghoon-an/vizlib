"""Relational charts: ``scatter`` and ``line`` (incl. gradient/stacked area)."""

from __future__ import annotations

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

from ..core import _coerce_numeric, _looks_datetime, _require_dataframe
from .chrome import _finish, _new_ax, _rotate_xticklabels
from .data import _AUTO_SAMPLE_SCATTER, _maybe_sample, _require_columns
from .marks import (
    _base_color,
    _draw_callouts,
    _gradient_fill,
    _hue_palette,
    _stack_colors,
    _swatch_legend,
    _to_num,
)
from .theme import _THEME


def scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    hue: str | None = None,
    reg: bool = False,
    sample: int | None = None,
    random_state: int = 0,
    annotations=None,
    title: str | None = None,
    subtitle: str | None = None,
    source: str | None = None,
    ax=None,
    **kwargs,
):
    """Plot the relationship between two numeric columns.

    ``x`` and ``y`` are coerced to numeric (currency/thousands/``%`` stripped),
    so numeric-looking string columns just work. Optionally colour points by
    ``hue`` and overlay a single accent-colored regression line with
    ``reg=True``. The regression line is drawn without a shaded confidence
    band (``ci=None``): a cleaner trend line, at the cost of no longer showing
    the fit's uncertainty. Rows missing any plotted value are dropped (never
    in place).
    Large frames auto-sample for responsiveness; pass ``sample=`` for an
    explicit reproducible subset (``random_state``). ``annotations`` attaches
    leader-line callouts — a list of ``(x, y, text)`` points. When ``hue`` is
    given a clean legend is drawn (a swatch legend under the infographic
    preset). Returns the ``Axes``.
    """
    _require_dataframe(df)
    cols = [x, y] + ([hue] if hue else [])
    _require_columns(df, cols)
    sub = df[cols].dropna().copy()
    sub[x] = _coerce_numeric(sub[x])
    sub[y] = _coerce_numeric(sub[y])
    sub = sub.dropna(subset=[x, y])
    if sub.empty:
        raise ValueError("no numeric values to plot")
    sub = _maybe_sample(sub, sample, _AUTO_SAMPLE_SCATTER, random_state)

    ax = _new_ax(ax)
    palette = _hue_palette(sub[hue]) if hue else None
    if not hue:
        kwargs.setdefault("color", _base_color())
    sns.scatterplot(data=sub, x=x, y=y, hue=hue, palette=palette, ax=ax, **kwargs)
    if reg:
        sns.regplot(data=sub, x=x, y=y, ax=ax, scatter=False, ci=None,
                    color=_THEME["accent"])
    if hue:
        handles, labels = ax.get_legend_handles_labels()
        if _THEME.get("swatch_legend"):
            colors = [h.get_markerfacecolor() if hasattr(h, "get_markerfacecolor")
                      else h.get_color() for h in handles]
            _swatch_legend(ax, labels, colors, title=str(hue))
        else:
            ax.legend(frameon=False, title=str(hue),
                      fontsize=_THEME["font_sizes"]["tick"],
                      title_fontsize=_THEME["font_sizes"]["label"])
    _draw_callouts(ax, annotations, sub[x].tolist(), sub[y].tolist())
    if title is None:
        title = f"Relationship between {x} and {y}"
    return _finish(ax, title=title, subtitle=subtitle, source=source,
                   xlabel=str(x), ylabel=str(y), grid_axis="both")


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

    The value axis ``y`` is coerced to numeric. A date-like ``x`` is parsed to
    datetimes and given readable, concise date ticks; otherwise ``x`` is left
    as-is. Rows are sorted by ``x`` and rows missing a plotted value are
    dropped (without mutating the caller's frame).

    - ``area=True`` fills under the line(s) with a vertical gradient.
    - ``area=True, stack=True`` with ``hue`` draws a **stacked** area; bands
      use the traffic-light scale when the hue is a low/medium/high category,
      otherwise the palette, and a swatch legend is drawn.
    - ``annotations`` attaches leader-line callouts — a list of ``(x, text)``
      (``y`` read from the nearest point) or ``(x, y, text)``.

    With ``hue`` (and no swatch legend) each line is labelled directly at its
    right end. Returns the ``Axes``.
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
        wide = (sub.pivot_table(index=x, columns=hue, values=y, aggfunc="sum")
                .fillna(0).sort_index())
        cats = list(wide.columns)
        rank = {"low": 0, "l": 0, "medium": 1, "med": 1, "m": 1, "high": 2, "h": 2}
        if all(str(c).strip().lower() in rank for c in cats):
            cats = sorted(cats, key=lambda c: rank[str(c).strip().lower()])
        colors = _stack_colors(cats)
        bottom = np.zeros(len(wide))
        for cat, col in zip(cats, colors):
            top = bottom + wide[cat].to_numpy(dtype=float)
            ax.fill_between(wide.index, bottom, top, color=col, alpha=0.85,
                            linewidth=0, label=str(cat))
            bottom = top
        _swatch_legend(ax, [str(c) for c in cats], colors, title=str(hue))
    elif hue:
        sns.lineplot(data=sub, x=x, y=y, hue=hue, palette=_hue_palette(sub[hue]),
                     ax=ax, **kwargs)
        handles, labels = ax.get_legend_handles_labels()
        color_by = {lab: h.get_color() for h, lab in zip(handles, labels)}
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        if area:
            for label, group in sub.groupby(hue):
                _gradient_fill(ax, _to_num(group[x]), group[y].to_numpy(float),
                               color_by.get(str(label), _base_color()))
        if _THEME.get("swatch_legend"):
            _swatch_legend(ax, list(color_by), list(color_by.values()),
                           title=str(hue))
        else:  # direct right-end labels
            for label, group in sub.groupby(hue):
                xl, yl = group[x].iloc[-1], group[y].iloc[-1]
                ax.text(xl, yl, f"  {label}", va="center", ha="left",
                        fontsize=_THEME["font_sizes"]["label"],
                        color=color_by.get(str(label), _THEME["text_color"]))
            ax.margins(x=0.15)
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
