"""The ``scatter`` relational chart."""

from __future__ import annotations

import pandas as pd
import seaborn as sns

from .._coerce import _coerce_numeric
from .._validate import _require_dataframe
from .annotate import _draw_callouts
from .chrome import _finish, _new_ax
from .colors import _base_color, _hue_palette
from .data import _AUTO_SAMPLE_SCATTER, _maybe_sample, _require_columns
from .labels import _swatch_legend
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
    so numeric-looking string columns just work. Colour points by ``hue`` and
    overlay a single accent-colored regression line with ``reg=True``. The
    regression line is drawn without a shaded confidence band (``ci=None``): a
    cleaner trend line, at the cost of no longer showing the fit's uncertainty.
    Rows missing any plotted value are dropped (never in place). Large frames
    auto-sample; pass ``sample=`` for an explicit reproducible subset.
    ``annotations`` attaches leader-line callouts. Returns the ``Axes``.
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
