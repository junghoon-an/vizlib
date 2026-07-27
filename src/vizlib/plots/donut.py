"""The ``donut`` (ring) chart — an infographic-style presentation extra."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import _require_series
from .chrome import _finalize_layout, _new_ax, _source, _titles
from .data import _resolve_column
from .marks import _luminance, _stack_colors, _surface_color
from .theme import _THEME


def donut(
    data,
    column: str | None = None,
    *,
    top: int = 8,
    center_text: str | None = None,
    explode=None,
    title: str | None = None,
    subtitle: str | None = None,
    source: str | None = None,
    ax=None,
    **kwargs,
):
    """Draw a donut (ring) chart of category shares — an infographic-style extra.

    Accepts a Series of raw values (its value counts are used), a counts
    Series (numeric values on a labelled index), or a ``(DataFrame, column)``
    pair. Wedges carry bold percentage labels, category names are attached
    with leader lines (so small slices stay readable), and ``center_text``
    prints a bold caption in the hole. ``explode`` (a label or list) pulls
    wedges out; the busiest ``top`` categories are kept and the rest folded
    into ``"Other"``.

    A donut trades proportion-accuracy for looks — angles and areas are harder
    to read than position, so for analysis prefer :func:`bar`. Kept as a
    presentation extra; never 3-D. Returns the ``Axes``; input not mutated.
    """
    if isinstance(data, pd.DataFrame):
        counts = _resolve_column(data, column).value_counts(dropna=True)
    else:
        _require_series(data)
        if pd.api.types.is_numeric_dtype(data) and not isinstance(
            data.index, pd.RangeIndex
        ):
            counts = data[data.notna()].astype(float)  # already label -> count
        else:
            counts = data.value_counts(dropna=True)

    counts = counts[counts > 0]
    if counts.empty:
        raise ValueError("no data to plot")
    if top is not None and len(counts) > top:
        other = counts.iloc[top:].sum()
        counts = pd.concat([counts.iloc[:top], pd.Series({"Other": other})])

    labels = [str(i) for i in counts.index]
    values = counts.to_numpy(dtype=float)
    colors = _stack_colors(labels)  # traffic-light for low/med/high, else palette
    explode_arr = None
    if explode is not None:
        keys = {explode} if isinstance(explode, str) else set(explode)
        keys |= {str(k) for k in keys}
        explode_arr = [0.08 if lab in keys else 0.0 for lab in labels]

    ax = _new_ax(ax)
    kwargs.setdefault("startangle", 90)
    surface = _surface_color()  # ring separators match the (light/dark) background
    wedges, _texts, autotexts = ax.pie(
        values, colors=colors, explode=explode_arr, counterclock=False,
        autopct=lambda p: f"{p:.0f}%", pctdistance=0.78,
        wedgeprops=dict(width=0.42, edgecolor=surface, linewidth=2),
        textprops=dict(color=_THEME["text_color"]), **kwargs,
    )
    for autotext, wedge in zip(autotexts, wedges):
        autotext.set_fontweight("bold")
        autotext.set_fontsize(_THEME["font_sizes"]["label"])
        dark = _luminance(wedge.get_facecolor()) < 0.55
        autotext.set_color("white" if dark else _THEME["text_color"])
    # leader-line category labels (keep small wedges readable)
    for wedge, label in zip(wedges, labels):
        angle = np.deg2rad((wedge.theta1 + wedge.theta2) / 2)
        xr, yr = np.cos(angle), np.sin(angle)
        ax.annotate(
            label, xy=(xr, yr), xytext=(1.25 * np.sign(xr) or 1.25, 1.15 * yr),
            ha="left" if xr >= 0 else "right", va="center",
            fontsize=_THEME["font_sizes"]["tick"], color=_THEME["text_color"],
            arrowprops=dict(arrowstyle="-", color=_THEME["muted"], lw=0.8),
            bbox=dict(boxstyle="round,pad=0.2", fc=surface, ec="none"),
        )
    if center_text:
        ax.text(0, 0, center_text, ha="center", va="center", fontweight="bold",
                fontsize=_THEME["font_sizes"]["title"], color=_THEME["text_color"])
    ax.set_aspect("equal")

    background = _THEME.get("background")
    if background:
        ax.figure.set_facecolor(background)
    if title is None:
        title = "Share by category"
    _titles(ax, title, subtitle)
    _source(ax, source)
    _finalize_layout(ax)
    return ax
