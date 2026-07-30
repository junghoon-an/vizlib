"""Heatmap-shaped charts: ``correlation_heatmap`` and ``missing_matrix``."""

from __future__ import annotations

import pandas as pd
import seaborn as sns

from .._coerce import _numeric_frame
from .._validate import _require_dataframe
from .axes import _rotate_xticklabels
from .chrome import _new_ax, _source, _titles
from .colors import _base_color, _luminance
from .data import _upper_triangle_mask
from .options import Captions
from .theme import _THEME


def _finish_matrix(ax, captions: Captions) -> None:
    """Shared chrome for the heatmaps: no border box, title, ticks, source."""
    for spine in ax.spines.values():
        spine.set_visible(False)
    _titles(ax, captions)
    ax.tick_params(colors=_THEME["text_color"],
                   labelsize=_THEME["font_sizes"]["tick"], length=0)
    _rotate_xticklabels(ax)
    _source(ax, captions)


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
    """Correlation heatmap of the numeric columns, done right.

    The redundant upper triangle is masked, cells are annotated to two
    decimals, and the diverging scale is centred at 0 and fixed to ``[-1, 1]``
    so colours compare across datasets. Needs at least two numeric columns.
    Returns the ``Axes``; the input is not mutated.
    """
    _require_dataframe(df)
    numeric = _numeric_frame(df)
    if numeric.shape[1] < 2:
        raise ValueError("need at least two numeric columns for a correlation heatmap")
    corr = numeric.corr(method=method)
    ax = _new_ax(ax)
    kwargs.setdefault("cmap", "vlag")
    kwargs.setdefault("linewidths", 0.5)
    sns.heatmap(corr, mask=_upper_triangle_mask(corr), annot=annot, fmt=".2f",
                vmin=-1, vmax=1, center=0, square=True,
                cbar_kws={"shrink": 0.8, "label": "correlation"}, ax=ax, **kwargs)
    if title is None:
        title = f"Correlation among numeric columns ({method})"
    _finish_matrix(ax, Captions(title, subtitle, source))
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
    """Nullity matrix — one row per record, dark cells mark missing values.

    Useful for spotting whether missingness is scattered or clustered.
    Returns the ``Axes``; the input is not mutated.
    """
    _require_dataframe(df)
    if df.shape[1] == 0:
        raise ValueError("no columns to plot")
    ax = _new_ax(ax)
    # "present" cells: a light panel on light themes, a lighter-than-navy panel
    # on a dark theme so the matrix reads on either background.
    bg = _THEME.get("background")
    present = "#33335A" if (bg and _luminance(bg) < 0.4) else "#e8e8e8"
    sns.heatmap(df.isna(), cbar=False, cmap=[present, _base_color()],
                yticklabels=False, ax=ax, **kwargs)
    if title is None:
        title = "Missing-value locations across the dataset"
    ax.set_xlabel("column", fontsize=_THEME["font_sizes"]["label"],
                  color=_THEME["text_color"])
    _finish_matrix(ax, Captions(title, subtitle, source))
    return ax
