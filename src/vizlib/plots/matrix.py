"""Matrix-shaped charts: ``correlation_heatmap``, ``missing_matrix``, ``pairplot``."""

from __future__ import annotations

import pandas as pd
import seaborn as sns

from ..core import _numeric_frame, _require_dataframe
from .chrome import _finalize_layout, _new_ax, _rotate_xticklabels, _source, _titles
from .data import (
    _AUTO_SAMPLE_PAIRPLOT,
    _maybe_sample,
    _require_columns,
    _upper_triangle_mask,
)
from .marks import _base_color, _hue_palette
from .theme import _THEME


def correlation_heatmap(
    df: pd.DataFrame,
    *,
    method: str = "pearson",
    annot: bool = True,
    title: str | None = None,
    subtitle: str | None = None,
    source: str | None = None,
    ax=None,
    **kwargs,
):
    """Draw a correlation heatmap of the numeric columns, done right.

    The redundant upper triangle is masked, cells are annotated to two
    decimals, and the diverging colour scale is centred at 0 and fixed to
    ``[-1, 1]`` so colours are comparable across datasets. Needs at least two
    numeric columns. Returns the ``Axes``; the input is not mutated.
    """
    _require_dataframe(df)
    numeric = _numeric_frame(df)
    if numeric.shape[1] < 2:
        raise ValueError("need at least two numeric columns for a correlation heatmap")

    corr = numeric.corr(method=method)
    mask = _upper_triangle_mask(corr)
    ax = _new_ax(ax)
    kwargs.setdefault("cmap", "vlag")
    kwargs.setdefault("linewidths", 0.5)
    sns.heatmap(corr, mask=mask, annot=annot, fmt=".2f", vmin=-1, vmax=1, center=0,
                square=True, cbar_kws={"shrink": 0.8, "label": "correlation"},
                ax=ax, **kwargs)
    for spine in ax.spines.values():
        spine.set_visible(False)  # no surrounding border box
    if title is None:
        title = f"Correlation among numeric columns ({method})"
    _titles(ax, title, subtitle)
    ax.tick_params(colors=_THEME["text_color"],
                   labelsize=_THEME["font_sizes"]["tick"], length=0)
    _rotate_xticklabels(ax)
    _source(ax, source)
    _finalize_layout(ax)
    return ax


def missing_matrix(
    df: pd.DataFrame,
    *,
    title: str | None = None,
    subtitle: str | None = None,
    source: str | None = None,
    ax=None,
    **kwargs,
):
    """Draw a nullity matrix — one row per record, dark cells mark missing values.

    Useful for spotting whether missingness is scattered or clustered.
    Returns the ``Axes``; the input is not mutated.
    """
    _require_dataframe(df)
    if df.shape[1] == 0:
        raise ValueError("no columns to plot")
    ax = _new_ax(ax)
    present, missing = "#e8e8e8", _base_color()
    sns.heatmap(df.isna(), cbar=False, cmap=[present, missing],
                yticklabels=False, ax=ax, **kwargs)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title is None:
        title = "Missing-value locations across the dataset"
    _titles(ax, title, subtitle)
    ax.set_xlabel("column", fontsize=_THEME["font_sizes"]["label"],
                  color=_THEME["text_color"])
    ax.tick_params(colors=_THEME["text_color"],
                   labelsize=_THEME["font_sizes"]["tick"], length=0)
    _rotate_xticklabels(ax)
    _source(ax, source)
    _finalize_layout(ax)
    return ax


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
    """Draw a scatter-matrix of the numeric columns and return the seaborn grid.

    This is a multi-panel figure, so it returns the seaborn ``PairGrid``
    (use ``grid.figure`` for the ``Figure``) rather than a single ``Axes``.
    Restrict the columns with ``columns`` and colour by ``hue``. Large frames
    auto-sample for responsiveness; pass ``sample=`` for an explicit
    reproducible subset (``random_state``). The input is not mutated.
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
