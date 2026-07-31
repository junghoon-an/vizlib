"""The ``pairplot`` scatter-matrix (returns the seaborn grid, not an Axes)."""

from __future__ import annotations

import pandas as pd
import seaborn as sns

from .._coerce import _numeric_frame
from .._validate import _require_dataframe
from .colors import _hue_palette
from .data import _AUTO_SAMPLE_PAIRPLOT, _maybe_sample, _require_columns
from .theme import _THEME


def pairplot(
    df: pd.DataFrame,
    *,
    hue: str | None = None,
    columns: list[str] | None = None,
    sample: int | None = None,
    random_state: int = 0,
    title: str | None = None,
    subtitle: str | None = None,
    source: str | None = None,
    **kwargs,
):
    """Scatter-matrix of the numeric columns; returns the seaborn ``PairGrid``.

    A multi-panel figure, so it returns the grid (use ``grid.figure`` for the
    ``Figure``) rather than a single ``Axes``. Restrict with ``columns`` and
    colour by ``hue``. Large frames auto-sample; pass ``sample=`` for an
    explicit reproducible subset. The input is not mutated.
    """
    _require_dataframe(df)
    if columns is not None:
        _require_columns(df, columns)
        variables = columns
    else:
        variables = list(_numeric_frame(df).columns)
    if len(variables) < 2:
        raise ValueError("need at least two numeric columns for a pairplot")

    plot_df = _maybe_sample(df, sample, _AUTO_SAMPLE_PAIRPLOT, random_state)
    kwargs.setdefault("corner", True)
    kwargs.setdefault("diag_kind", "kde")
    palette = _hue_palette(df[hue]) if hue else None
    grid = sns.pairplot(plot_df, vars=variables, hue=hue, palette=palette, **kwargs)
    fig = grid.figure
    sizes, tc = _THEME["font_sizes"], _THEME["text_color"]
    if title is None:
        title = "Pairwise relationships among numeric columns"
    fig.suptitle(title, x=0.02, ha="left", fontsize=sizes["title"],
                 fontweight="bold", color=tc)
    if subtitle:
        fig.text(0.02, 0.965, subtitle, ha="left", va="top",
                 fontsize=sizes["subtitle"], color=_THEME["muted"])
    if source:
        fig.text(0.02, 0.005, source, ha="left", va="bottom",
                 fontsize=sizes["source"], color=_THEME["muted"])
    fig.tight_layout()
    return grid
