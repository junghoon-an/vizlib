"""Matplotlib/seaborn-backed plots for vizlib — opinionated EDA figures.

This module is the graphical counterpart to :mod:`vizlib.core`. The core
functions work with pandas alone; importing *this* module pulls in
matplotlib and seaborn, which are an optional extra::

    pip install "vizlib[plot]"

It is deliberately kept out of ``vizlib/__init__.py``'s import chain, so a
bare ``import vizlib`` stays fast and pandas-only. Reach the plotting API
explicitly with ``import vizlib.plots`` or ``from vizlib import plots``.

Every function here follows the same contract as the core: it never
mutates its input, and it *returns* the matplotlib ``Axes`` it drew on
(or a ``Figure``/seaborn grid for multi-panel plots) rather than calling
``plt.show()``. Pass ``ax=`` to compose plots into an existing figure. The
defaults bake in sound data-visualization practice — colorblind-safe
palettes, honest zero baselines, sorted categories, de-cluttered axes and
always-labelled titles — so a good chart is the path of least resistance.
"""

from __future__ import annotations

import pandas as pd

try:  # matplotlib + seaborn are the optional "plot" extra
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "vizlib.plots requires matplotlib and seaborn. "
        'Install them with: pip install "vizlib[plot]"'
    ) from exc

from .core import _require_dataframe, _require_series

__all__ = [
    "set_theme",
    "bar",
    "hist",
    "distribution",
    "box",
    "scatter",
    "line",
    "correlation_heatmap",
    "missing_bar",
    "missing_matrix",
    "pairplot",
]

# Shared, mutable defaults. Updated by set_theme(); read by every plot so a
# good-looking figure needs no configuration and stays consistent.
_THEME: dict = {
    "palette": "colorblind",  # colorblind-safe qualitative default
    "context": "notebook",
    "style": "whitegrid",
    "figsize": (8, 5),
    "dpi": 110,
}


def set_theme(
    *,
    palette: str = "colorblind",
    context: str = "notebook",
    style: str = "whitegrid",
    figsize: tuple[float, float] = (8, 5),
    dpi: int = 110,
) -> None:
    """Configure the look shared by every plot in this module.

    Updates the module defaults and applies seaborn's global context/style
    so future figures match. Returns ``None``; call it once (optionally)
    before plotting. Plots already look good without calling this.
    """
    _THEME.update(
        palette=palette, context=context, style=style, figsize=figsize, dpi=dpi
    )
    sns.set_theme(context=context, style=style, palette=palette)


def bar(
    data,
    column: str | None = None,
    *,
    top: int = 15,
    sort: bool = True,
    ax: "Axes | None" = None,
    **kwargs,
) -> "Axes":
    """Draw a value-counts bar chart — the graphical twin of ``value_counts_bar``.

    Accepts a Series or a DataFrame plus a ``column`` name. Keeps the busiest
    ``top`` categories and folds the rest into a single ``"Other"`` bar,
    sorts by frequency (descending) unless the column is an ordered
    categorical or ``sort=False``, and anchors the count axis at zero.
    Returns the ``Axes``; the input is never mutated.
    """
    series = _resolve_column(data, column)
    ordered = isinstance(series.dtype, pd.CategoricalDtype) and series.dtype.ordered
    counts = series.value_counts(dropna=True)
    if counts.empty:
        raise ValueError("no data to plot")

    if ordered:
        counts = counts.reindex(list(series.cat.categories)).dropna()
    else:
        if top is not None and len(counts) > top:
            other = counts.iloc[top:].sum()
            counts = pd.concat([counts.iloc[:top], pd.Series({"Other": other})])
        if sort:
            counts = counts.sort_values(ascending=False)

    ax = _new_ax(ax)
    # Horizontal bars read well with long labels; one colour, because the
    # category is already encoded by position. Reverse so the biggest is on top.
    plot = counts.iloc[::-1]
    kwargs.setdefault("color", _base_color())
    ax.barh([str(i) for i in plot.index], plot.to_numpy(), **kwargs)
    ax.set_xlim(left=0)  # honest, zero-based magnitude axis
    name = series.name if series.name is not None else "value"
    return _finish(ax, title=f"Value counts: {name}", xlabel="count", ylabel=str(name),
                   grid_axis="x")


def hist(
    series: pd.Series,
    *,
    bins="auto",
    kde: bool = False,
    ax: "Axes | None" = None,
    **kwargs,
) -> "Axes":
    """Plot a histogram of a numeric Series — the graphical twin of ``histogram``.

    Non-numeric entries and missing values are dropped (never in place).
    Set ``kde=True`` to overlay a kernel-density estimate. Returns the
    ``Axes``.
    """
    values = _numeric_values(series)
    ax = _new_ax(ax)
    kwargs.setdefault("color", _base_color())
    sns.histplot(x=values, bins=bins, kde=kde, ax=ax, **kwargs)
    name = series.name if series.name is not None else "value"
    return _finish(ax, title=f"Distribution of {name}", xlabel=str(name),
                   ylabel="count")


def distribution(
    series: pd.Series, *, ax: "Axes | None" = None, **kwargs
) -> "Axes":
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
    name = series.name if series.name is not None else "value"
    return _finish(ax, title=f"Distribution of {name}", xlabel=str(name),
                   ylabel="count")


def box(
    df: pd.DataFrame,
    column: str | None = None,
    *,
    by: str | None = None,
    ax: "Axes | None" = None,
    **kwargs,
) -> "Axes":
    """Draw a boxplot for spread and outlier inspection.

    With no ``column`` every numeric column becomes a box. Give ``column``
    for a single distribution, and add ``by`` to split it across the levels
    of a categorical column. Rows with missing values in the plotted
    columns are dropped. Returns the ``Axes``; the input is untouched.
    """
    _require_dataframe(df)
    ax = _new_ax(ax)
    kwargs.setdefault("color", _base_color())  # one colour; position carries the group
    rotate = False

    if column is None:
        numeric = df.select_dtypes(include="number")
        if numeric.empty:
            raise ValueError("no numeric columns to plot")
        sns.boxplot(data=numeric, ax=ax, **kwargs)
        title, xlabel, ylabel = "Distribution by column", "", "value"
        rotate = numeric.shape[1] > 4
    elif by is None:
        values = _numeric_values(df[column])
        sns.boxplot(y=values, ax=ax, **kwargs)
        title, xlabel, ylabel = f"Distribution of {column}", "", str(column)
    else:
        sub = df[[column, by]].dropna()
        if sub.empty:
            raise ValueError("no rows left after dropping missing values")
        sns.boxplot(data=sub, x=by, y=column, ax=ax, **kwargs)
        title, xlabel, ylabel = f"{column} by {by}", str(by), str(column)
        rotate = True

    ax = _finish(ax, title=title, xlabel=xlabel, ylabel=ylabel, grid_axis="y")
    if rotate:
        _rotate_xticklabels(ax)
    return ax


def scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    hue: str | None = None,
    reg: bool = False,
    ax: "Axes | None" = None,
    **kwargs,
) -> "Axes":
    """Plot the relationship between two numeric columns.

    Optionally colour points by ``hue`` and overlay a single regression
    line with ``reg=True``. Rows missing any plotted value are dropped
    (never in place). A legend appears only when ``hue`` is given. Returns
    the ``Axes``.
    """
    _require_dataframe(df)
    cols = [x, y] + ([hue] if hue else [])
    _require_columns(df, cols)
    sub = df[cols].dropna()
    if sub.empty:
        raise ValueError("no rows left after dropping missing values")

    ax = _new_ax(ax)
    palette = _THEME["palette"] if hue else None
    sns.scatterplot(data=sub, x=x, y=y, hue=hue, palette=palette, ax=ax, **kwargs)
    if reg:
        sns.regplot(data=sub, x=x, y=y, ax=ax, scatter=False, color=_base_color())
    return _finish(ax, title=f"{y} vs {x}", xlabel=str(x), ylabel=str(y),
                   grid_axis="both")


def line(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    hue: str | None = None,
    ax: "Axes | None" = None,
    **kwargs,
) -> "Axes":
    """Draw a line plot for ordered or time-series data.

    Rows are sorted by ``x`` and rows missing a plotted value are dropped
    (without mutating the caller's frame). Group into multiple lines with
    ``hue``; a legend appears only then. Returns the ``Axes``.
    """
    _require_dataframe(df)
    cols = [x, y] + ([hue] if hue else [])
    _require_columns(df, cols)
    sub = df[cols].dropna().sort_values(x)
    if sub.empty:
        raise ValueError("no rows left after dropping missing values")

    ax = _new_ax(ax)
    palette = _THEME["palette"] if hue else None
    color = None if hue else _base_color()
    sns.lineplot(data=sub, x=x, y=y, hue=hue, palette=palette, color=color,
                 ax=ax, **kwargs)
    ax = _finish(ax, title=f"{y} over {x}", xlabel=str(x), ylabel=str(y),
                 grid_axis="both")
    _rotate_xticklabels(ax)
    return ax


def correlation_heatmap(
    df: pd.DataFrame,
    *,
    method: str = "pearson",
    annot: bool = True,
    ax: "Axes | None" = None,
    **kwargs,
) -> "Axes":
    """Draw a correlation heatmap of the numeric columns, done right.

    The redundant upper triangle is masked, cells are annotated, and the
    diverging colour scale is centred at 0 and fixed to ``[-1, 1]`` so the
    colours are comparable across datasets. Needs at least two numeric
    columns. Returns the ``Axes``; the input is not mutated.
    """
    _require_dataframe(df)
    numeric = df.select_dtypes(include="number")
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
    ax.set_title(f"Correlation ({method})")
    _rotate_xticklabels(ax)
    ax.figure.tight_layout()
    return ax


def missing_bar(
    df: pd.DataFrame, *, ax: "Axes | None" = None, **kwargs
) -> "Axes":
    """Bar chart of the percentage of missing values per column, largest first.

    The graphical twin of :func:`vizlib.core.missing_values`. The axis runs
    a full 0–100 % so magnitudes read honestly. Returns the ``Axes``; the
    input is untouched.
    """
    _require_dataframe(df)
    if df.shape[1] == 0:
        raise ValueError("no columns to plot")
    n = len(df)
    pct = (df.isna().sum() / n * 100) if n else df.isna().sum() * 0.0
    pct = pct.sort_values(ascending=False)

    ax = _new_ax(ax)
    kwargs.setdefault("color", _base_color())
    plot = pct.iloc[::-1]  # biggest on top for a horizontal bar
    ax.barh([str(i) for i in plot.index], plot.to_numpy(), **kwargs)
    ax.set_xlim(0, 100)
    return _finish(ax, title="Missing values by column", xlabel="% missing",
                   ylabel="column", grid_axis="x")


def missing_matrix(
    df: pd.DataFrame, *, ax: "Axes | None" = None, **kwargs
) -> "Axes":
    """Draw a nullity matrix — one row per record, dark cells mark missing values.

    Useful for spotting whether missingness is scattered or clustered.
    Returns the ``Axes``; the input is not mutated.
    """
    _require_dataframe(df)
    if df.shape[1] == 0:
        raise ValueError("no columns to plot")
    present, missing = "#e8e8e8", _base_color()
    sns.heatmap(df.isna(), cbar=False, cmap=[present, missing],
                yticklabels=False, ax=_new_ax(ax) if ax is None else ax,
                **kwargs)
    ax = plt.gca() if ax is None else ax
    ax.set_title("Missing-data matrix (dark = missing)")
    ax.set_xlabel("column")
    _rotate_xticklabels(ax)
    ax.figure.tight_layout()
    return ax


def pairplot(
    df: pd.DataFrame,
    *,
    hue: str | None = None,
    columns: list[str] | None = None,
    **kwargs,
):
    """Draw a scatter-matrix of the numeric columns and return the seaborn grid.

    This is a multi-panel figure, so it returns the seaborn ``PairGrid``
    (use ``grid.figure`` for the ``Figure``) rather than a single ``Axes``.
    Restrict the columns with ``columns`` and colour by ``hue``. The input
    is not mutated.
    """
    _require_dataframe(df)
    if columns is not None:
        _require_columns(df, columns)
        variables = columns
    else:
        variables = list(df.select_dtypes(include="number").columns)
    if len(variables) < 2:
        raise ValueError("need at least two numeric columns for a pairplot")

    kwargs.setdefault("corner", True)
    kwargs.setdefault("diag_kind", "kde")
    palette = _THEME["palette"] if hue else None
    grid = sns.pairplot(df, vars=variables, hue=hue, palette=palette, **kwargs)
    grid.figure.tight_layout()
    return grid


# --- internal helpers -------------------------------------------------------

def _new_ax(ax: "Axes | None") -> "Axes":
    """Return ``ax`` or a fresh one sized from the shared theme."""
    if ax is None:
        _, ax = plt.subplots(figsize=_THEME["figsize"], dpi=_THEME["dpi"])
    return ax


def _base_color():
    """The first colour of the active colorblind-safe palette."""
    return sns.color_palette(_THEME["palette"])[0]


def _style_axes(ax: "Axes", *, grid_axis: str = "y") -> "Axes":
    """Apply the shared low-chartjunk styling: no top/right spines, light grid."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)
    if grid_axis in ("x", "y", "both"):
        ax.grid(False)
        ax.grid(True, axis=grid_axis, alpha=0.3, linewidth=0.6)
    return ax


def _finish(ax, *, title=None, xlabel=None, ylabel=None, grid_axis="y") -> "Axes":
    """Label, style and tidy an Axes, then return it."""
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    _style_axes(ax, grid_axis=grid_axis)
    ax.figure.tight_layout()
    return ax


def _rotate_xticklabels(ax: "Axes", angle: int = 45) -> None:
    """Rotate x tick labels so long/dense labels don't overlap."""
    for label in ax.get_xticklabels():
        label.set_rotation(angle)
        label.set_ha("right")


def _resolve_column(data, column: str | None) -> pd.Series:
    """Return the Series to plot from a Series or a (DataFrame, column) pair."""
    if isinstance(data, pd.DataFrame):
        if column is None:
            raise ValueError("column is required when data is a DataFrame")
        _require_columns(data, [column])
        return data[column]
    _require_series(data)
    return data


def _require_columns(df: pd.DataFrame, columns) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"column(s) not found in DataFrame: {missing}")


def _numeric_values(series: pd.Series) -> pd.Series:
    """Coerce a Series to numeric and drop NaN, mirroring ``core.histogram``."""
    _require_series(series)
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        raise ValueError("no numeric values to plot")
    return values


def _upper_triangle_mask(corr: pd.DataFrame) -> pd.DataFrame:
    """Boolean mask hiding the redundant upper triangle (incl. diagonal).

    Built with pandas alone so this module's only hard imports stay
    matplotlib, seaborn and pandas.
    """
    mask = pd.DataFrame(False, index=corr.index, columns=corr.columns)
    for i in range(len(corr)):
        for j in range(i, len(corr)):
            mask.iat[i, j] = True
    return mask
