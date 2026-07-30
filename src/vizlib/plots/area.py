"""Area-fill helpers for the ``line`` chart: stacked bands and per-hue fills."""

from __future__ import annotations

import numpy as np
import seaborn as sns

from .annotate import _gradient_fill, _to_num
from .colors import _base_color, _hue_palette, _stack_colors
from .labels import _swatch_legend
from .theme import _THEME


def _stacked_area(ax, sub, x, y, hue) -> None:
    """Draw a stacked area: bands ordered low->high when categorical, else palette."""
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


def _hued_lines(ax, sub, x, y, hue, area, kwargs) -> None:
    """One line per hue level, with optional gradient fill and direct labels."""
    sns.lineplot(data=sub, x=x, y=y, hue=hue, palette=_hue_palette(sub[hue]),
                 ax=ax, **kwargs)
    handles, labels = ax.get_legend_handles_labels()
    color_by = {lab: h.get_color() for h, lab in zip(handles, labels)}
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    if area:
        for label, group in sub.groupby(hue):
            _gradient_fill(ax, _to_num(group[x]), group[y].to_numpy(float),
                           color_by.get(str(label), _base_color()))
    if _THEME.get("swatch_legend"):
        _swatch_legend(ax, list(color_by), list(color_by.values()), title=str(hue))
    else:  # direct right-end labels
        for label, group in sub.groupby(hue):
            xl, yl = group[x].iloc[-1], group[y].iloc[-1]
            ax.text(xl, yl, f"  {label}", va="center", ha="left",
                    fontsize=_THEME["font_sizes"]["label"],
                    color=color_by.get(str(label), _THEME["text_color"]))
        ax.margins(x=0.15)
