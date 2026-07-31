"""Distribution charts of a single numeric Series: ``hist`` and ``distribution``."""

from __future__ import annotations

import pandas as pd
import seaborn as sns

from .chrome import _finish, _new_ax
from .colors import _base_color
from .data import _numeric_values
from .options import Captions


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
    """Histogram of a numeric Series — the graphical twin of ``histogram``.

    Non-numeric entries and missing values are dropped (never in place) and
    the count axis starts at zero. Set ``kde=True`` for a density overlay.
    Returns the ``Axes``.
    """
    values = _numeric_values(series)
    ax = _new_ax(ax)
    kwargs.setdefault("color", _base_color())
    sns.histplot(x=values, bins=bins, kde=kde, ax=ax, **kwargs)
    ax.set_ylim(bottom=0)  # honest, zero-based count axis
    name = series.name if series.name is not None else "value"
    if title is None:
        title = f"Distribution of {name}"
    return _finish(ax, Captions(title, subtitle, source),
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
    return _finish(ax, Captions(title, subtitle, source),
                   xlabel=str(name), ylabel="count", grid_axis="y")
