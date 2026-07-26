"""Matplotlib/seaborn-backed plots for vizlib — opinionated EDA figures.

This module is the graphical counterpart to :mod:`vizlib.core`. The core
functions work with pandas alone; importing *this* module pulls in
matplotlib and seaborn (installed by default as of 0.3.0). It is kept out
of ``vizlib/__init__.py``'s import chain, so a bare ``import vizlib`` stays
fast — reach the plotting API explicitly with ``import vizlib.plots`` or
``from vizlib import plots``.

Every function follows the same contract as the core: it never mutates its
input, and it *returns* the matplotlib ``Axes`` it drew on (or a
``Figure``/seaborn grid for multi-panel plots) rather than calling
``plt.show()``. Pass ``ax=`` to compose into an existing figure.

The defaults follow the Evergreen & Emery Data Visualization Checklist: a
left-justified descriptive title, an optional muted subtitle and source
caption, a readable font hierarchy, dark high-contrast text, muted
gridlines, direct data labels on bars (with the redundant value axis
hidden), frequency/median-ordered categories, honest zero-based magnitude
axes, and a colorblind- and grayscale-legible palette. The mechanical
guidelines are enforced automatically; the interpretive ones are exposed as
hooks — ``title``, ``subtitle``, ``source`` and ``highlight`` — with neutral
defaults. Override ``title``/``subtitle`` with your actual finding; vizlib
never fabricates a takeaway the data doesn't support.
"""

from __future__ import annotations

import pandas as pd

try:  # matplotlib + seaborn ship as regular dependencies
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
except ImportError as exc:  # pragma: no cover - exercised only without the deps
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
# checklist-compliant figure needs no configuration and stays consistent.
_DEFAULTS: dict = {
    "palette": "colorblind",   # colorblind- and luminance-separated
    "context": "notebook",
    "style": "whitegrid",
    "figsize": (8, 5),
    "dpi": 110,
    "accent": "#1a5fb4",       # action color for highlighted marks
    "muted": "#b6b6b6",        # de-emphasis gray
    "text_color": "#1a1a1a",   # near-black, high contrast on white
    "grid_color": "#dcdcdc",   # faint gray gridlines
    "font_sizes": {"title": 15, "subtitle": 12, "label": 11, "tick": 10, "source": 8},
}

_THEME: dict = {**_DEFAULTS, "font_sizes": dict(_DEFAULTS["font_sizes"])}


def set_theme(
    *,
    palette: str | None = None,
    context: str | None = None,
    style: str | None = None,
    figsize: tuple[float, float] | None = None,
    dpi: int | None = None,
    accent: str | None = None,
    muted: str | None = None,
    text_color: str | None = None,
    grid_color: str | None = None,
    title_size: float | None = None,
    subtitle_size: float | None = None,
    label_size: float | None = None,
    tick_size: float | None = None,
    source_size: float | None = None,
) -> None:
    """Configure the look shared by every plot in this module.

    All parameters are optional and default to the current look, so pass
    only what you want to change. ``accent`` is the action color used by
    ``highlight=``; ``muted`` de-emphasizes everything else. The default
    ``colorblind`` palette and the diverging ``vlag`` correlation map are
    chosen to separate by luminance too, so patterns survive black-and-white
    printing. Returns ``None``; plots already look good without calling this.
    """
    scalar = {
        "palette": palette, "context": context, "style": style,
        "figsize": figsize, "dpi": dpi, "accent": accent, "muted": muted,
        "text_color": text_color, "grid_color": grid_color,
    }
    for key, value in scalar.items():
        if value is not None:
            _THEME[key] = value
    sizes = {
        "title": title_size, "subtitle": subtitle_size, "label": label_size,
        "tick": tick_size, "source": source_size,
    }
    for key, value in sizes.items():
        if value is not None:
            _THEME["font_sizes"][key] = value
    sns.set_theme(
        context=_THEME["context"], style=_THEME["style"], palette=_THEME["palette"]
    )


def bar(
    data,
    column: str | None = None,
    *,
    top: int = 15,
    sort: bool = True,
    highlight=None,
    value_labels: bool = True,
    precision: int = 0,
    title: str | None = None,
    subtitle: str | None = None,
    source: str | None = None,
    ax: "Axes | None" = None,
    **kwargs,
) -> "Axes":
    """Draw a value-counts bar chart — the graphical twin of ``value_counts_bar``.

    Keeps the busiest ``top`` categories, folds the rest into ``"Other"``,
    sorts by frequency (descending) unless the column is an ordered
    categorical or ``sort=False``, and anchors the count axis at zero. By
    default each bar is labelled directly (``value_labels=True``) and the
    now-redundant value axis and tick marks are hidden. ``highlight`` (a
    label or list of labels) paints the chosen bars in the accent color and
    the rest muted. ``precision`` sets the label decimals. Returns the
    ``Axes``; the input is never mutated. The default ``title`` is a neutral
    placeholder — override it with your finding.
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
    plot = counts.iloc[::-1]  # biggest on top for a horizontal bar
    kwargs.setdefault("color", _bar_colors(plot.index, highlight))
    container = ax.barh([str(i) for i in plot.index], plot.to_numpy(), **kwargs)
    biggest = float(plot.max())
    ax.set_xlim(0, biggest * 1.18 if biggest else 1)  # zero-based, room for labels

    name = series.name if series.name is not None else "value"
    grid_axis, xlabel = "x", "count"
    if value_labels:
        ax.bar_label(
            container, fmt=f"%.{precision}f", padding=3,
            fontsize=_THEME["font_sizes"]["label"], color=_THEME["text_color"],
        )
        ax.xaxis.set_visible(False)                 # value axis is redundant now
        ax.spines["bottom"].set_visible(False)
        grid_axis, xlabel = None, None
    ax.tick_params(length=0)  # no tick marks on bar-type charts
    if title is None:
        title = f"Record count by {name}"
    return _finish(ax, title=title, subtitle=subtitle, source=source,
                   xlabel=xlabel, ylabel=str(name), grid_axis=grid_axis)


def hist(
    series: pd.Series,
    *,
    bins="auto",
    kde: bool = False,
    title: str | None = None,
    subtitle: str | None = None,
    source: str | None = None,
    ax: "Axes | None" = None,
    **kwargs,
) -> "Axes":
    """Plot a histogram of a numeric Series — the graphical twin of ``histogram``.

    Non-numeric entries and missing values are dropped (never in place) and
    the count axis starts at zero. Set ``kde=True`` for a density overlay.
    ``title``/``subtitle``/``source`` add checklist-style captions. Returns
    the ``Axes``.
    """
    values = _numeric_values(series)
    ax = _new_ax(ax)
    kwargs.setdefault("color", _base_color())
    sns.histplot(x=values, bins=bins, kde=kde, ax=ax, **kwargs)
    ax.set_ylim(bottom=0)  # honest, zero-based count axis
    name = series.name if series.name is not None else "value"
    if title is None:
        title = f"Distribution of {name}"
    return _finish(ax, title=title, subtitle=subtitle, source=source,
                   xlabel=str(name), ylabel="count", grid_axis="y")


def distribution(
    series: pd.Series,
    *,
    title: str | None = None,
    subtitle: str | None = None,
    source: str | None = None,
    ax: "Axes | None" = None,
    **kwargs,
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
    ax.set_ylim(bottom=0)
    name = series.name if series.name is not None else "value"
    if title is None:
        title = f"Distribution of {name}"
    return _finish(ax, title=title, subtitle=subtitle, source=source,
                   xlabel=str(name), ylabel="count", grid_axis="y")


def box(
    df: pd.DataFrame,
    column: str | None = None,
    *,
    by: str | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    source: str | None = None,
    ax: "Axes | None" = None,
    **kwargs,
) -> "Axes":
    """Draw a boxplot for spread and outlier inspection.

    With no ``column`` every numeric column becomes a box. Give ``column``
    for a single distribution, and add ``by`` to split it across the levels
    of a categorical column — unordered groups are ordered by median (an
    ordered categorical keeps its natural order). Rows missing a plotted
    value are dropped. Returns the ``Axes``; the input is untouched.
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
        default_title, xlabel, ylabel = "Value spread across numeric columns", "", "value"
        rotate = numeric.shape[1] > 4
    elif by is None:
        values = _numeric_values(df[column])
        sns.boxplot(y=values, ax=ax, **kwargs)
        default_title, xlabel, ylabel = f"Value spread for {column}", "", str(column)
    else:
        sub = df[[column, by]].dropna()
        if sub.empty:
            raise ValueError("no rows left after dropping missing values")
        by_series = sub[by]
        if isinstance(by_series.dtype, pd.CategoricalDtype) and by_series.dtype.ordered:
            order = [c for c in by_series.cat.categories if c in set(by_series)]
        else:  # unordered: order boxes by group median
            order = sub.groupby(by)[column].median().sort_values().index.tolist()
        sns.boxplot(data=sub, x=by, y=column, order=order, ax=ax, **kwargs)
        default_title, xlabel, ylabel = f"{column} across {by} groups", str(by), str(column)
        rotate = True

    ax = _finish(ax, title=title or default_title, subtitle=subtitle,
                 source=source, xlabel=xlabel, ylabel=ylabel, grid_axis="y")
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
    title: str | None = None,
    subtitle: str | None = None,
    source: str | None = None,
    ax: "Axes | None" = None,
    **kwargs,
) -> "Axes":
    """Plot the relationship between two numeric columns.

    Optionally colour points by ``hue`` and overlay a single accent-colored
    regression line with ``reg=True``. Rows missing any plotted value are
    dropped (never in place). When ``hue`` is given a clean, frameless legend
    is drawn (direct labelling isn't practical for a point cloud). Returns
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
    if not hue:
        kwargs.setdefault("color", _base_color())
    sns.scatterplot(data=sub, x=x, y=y, hue=hue, palette=palette, ax=ax, **kwargs)
    if reg:
        sns.regplot(data=sub, x=x, y=y, ax=ax, scatter=False, color=_THEME["accent"])
    if hue:
        ax.legend(frameon=False, title=str(hue),
                  fontsize=_THEME["font_sizes"]["tick"],
                  title_fontsize=_THEME["font_sizes"]["label"])
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
    title: str | None = None,
    subtitle: str | None = None,
    source: str | None = None,
    ax: "Axes | None" = None,
    **kwargs,
) -> "Axes":
    """Draw a line plot for ordered or time-series data.

    Rows are sorted by ``x`` and rows missing a plotted value are dropped
    (without mutating the caller's frame). With ``hue`` each line is labelled
    directly at its right end and the legend is suppressed. Tick marks are
    kept, since they demarcate points along the axis. Returns the ``Axes``.
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
    if hue:
        handles, labels = ax.get_legend_handles_labels()
        color_by = {lab: h.get_color() for h, lab in zip(handles, labels)}
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        for label, group in sub.groupby(hue):
            xl, yl = group[x].iloc[-1], group[y].iloc[-1]
            ax.text(xl, yl, f"  {label}", va="center", ha="left",
                    fontsize=_THEME["font_sizes"]["label"],
                    color=color_by.get(str(label), _THEME["text_color"]))
        ax.margins(x=0.15)  # room for the right-end labels
    if title is None:
        title = f"{y} over {x}"
    ax = _finish(ax, title=title, subtitle=subtitle, source=source,
                 xlabel=str(x), ylabel=str(y), grid_axis="both")
    _rotate_xticklabels(ax)
    return ax


def correlation_heatmap(
    df: pd.DataFrame,
    *,
    method: str = "pearson",
    annot: bool = True,
    title: str | None = None,
    subtitle: str | None = None,
    source: str | None = None,
    ax: "Axes | None" = None,
    **kwargs,
) -> "Axes":
    """Draw a correlation heatmap of the numeric columns, done right.

    The redundant upper triangle is masked, cells are annotated to two
    decimals, and the diverging colour scale is centred at 0 and fixed to
    ``[-1, 1]`` so colours are comparable across datasets. Needs at least two
    numeric columns. Returns the ``Axes``; the input is not mutated.
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
    for spine in ax.spines.values():
        spine.set_visible(False)  # no surrounding border box
    if title is None:
        title = f"Correlation among numeric columns ({method})"
    _titles(ax, title, subtitle)
    ax.tick_params(colors=_THEME["text_color"],
                   labelsize=_THEME["font_sizes"]["tick"], length=0)
    _rotate_xticklabels(ax)
    _source(ax, source)
    ax.figure.tight_layout()
    return ax


def missing_bar(
    df: pd.DataFrame,
    *,
    highlight=None,
    value_labels: bool = True,
    precision: int = 1,
    title: str | None = None,
    subtitle: str | None = None,
    source: str | None = None,
    ax: "Axes | None" = None,
    **kwargs,
) -> "Axes":
    """Bar chart of the percentage of missing values per column, largest first.

    The graphical twin of :func:`vizlib.core.missing_values`. The axis runs a
    full 0–100 % so magnitudes read honestly; by default each bar is labelled
    directly and the value axis/tick marks are hidden. ``highlight`` accents
    the chosen column(s); ``precision`` sets the percent decimals. Returns the
    ``Axes``; the input is untouched.
    """
    _require_dataframe(df)
    if df.shape[1] == 0:
        raise ValueError("no columns to plot")
    n = len(df)
    pct = (df.isna().sum() / n * 100) if n else df.isna().sum() * 0.0
    pct = pct.sort_values(ascending=False)

    ax = _new_ax(ax)
    plot = pct.iloc[::-1]  # biggest on top
    kwargs.setdefault("color", _bar_colors(plot.index, highlight))
    container = ax.barh([str(i) for i in plot.index], plot.to_numpy(), **kwargs)
    ax.set_xlim(0, 100)

    grid_axis, xlabel = "x", "% missing"
    if value_labels:
        ax.bar_label(
            container, labels=[f"{v:.{precision}f}%" for v in plot.to_numpy()],
            padding=3, fontsize=_THEME["font_sizes"]["label"],
            color=_THEME["text_color"],
        )
        ax.xaxis.set_visible(False)
        ax.spines["bottom"].set_visible(False)
        grid_axis, xlabel = None, None
    ax.tick_params(length=0)
    if title is None:
        title = "Share of missing values by column"
    return _finish(ax, title=title, subtitle=subtitle, source=source,
                   xlabel=xlabel, ylabel="column", grid_axis=grid_axis)


def missing_matrix(
    df: pd.DataFrame,
    *,
    title: str | None = None,
    subtitle: str | None = None,
    source: str | None = None,
    ax: "Axes | None" = None,
    **kwargs,
) -> "Axes":
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
    ax.figure.tight_layout()
    return ax


def pairplot(
    df: pd.DataFrame,
    *,
    hue: str | None = None,
    columns: list[str] | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    source: str | None = None,
    **kwargs,
):
    """Draw a scatter-matrix of the numeric columns and return the seaborn grid.

    This is a multi-panel figure, so it returns the seaborn ``PairGrid``
    (use ``grid.figure`` for the ``Figure``) rather than a single ``Axes``.
    Restrict the columns with ``columns`` and colour by ``hue``. The input is
    not mutated.
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


# --- internal helpers -------------------------------------------------------

def _new_ax(ax: "Axes | None") -> "Axes":
    """Return ``ax`` or a fresh one sized from the shared theme."""
    if ax is None:
        _, ax = plt.subplots(figsize=_THEME["figsize"], dpi=_THEME["dpi"])
    return ax


def _base_color():
    """The first colour of the active colorblind-safe palette."""
    return sns.color_palette(_THEME["palette"])[0]


def _bar_colors(index, highlight):
    """Per-bar colours: accent for highlighted labels, muted for the rest.

    Returns the single base colour when ``highlight`` is ``None``.
    """
    if highlight is None:
        return _base_color()
    keys = {highlight} if isinstance(highlight, str) else set(highlight)
    keys |= {str(k) for k in keys}
    accent, muted = _THEME["accent"], _THEME["muted"]
    return [accent if (idx in keys or str(idx) in keys) else muted for idx in index]


def _titles(ax: "Axes", title: str | None = None, subtitle: str | None = None) -> None:
    """Set a left-justified title and, under it, a muted left-aligned subtitle."""
    sizes, tc, muted = _THEME["font_sizes"], _THEME["text_color"], _THEME["muted"]
    if subtitle:
        ax.set_title(title or "", loc="left", fontsize=sizes["title"],
                     fontweight="bold", color=tc, pad=24)
        ax.text(0.0, 1.015, subtitle, transform=ax.transAxes, ha="left",
                va="bottom", fontsize=sizes["subtitle"], color=muted)
    elif title:
        ax.set_title(title, loc="left", fontsize=sizes["title"],
                     fontweight="bold", color=tc, pad=10)


def _source(ax: "Axes", text: str | None = None) -> None:
    """Render a small, muted source caption in the lower-left, below the axes."""
    if not text:
        return
    ax.text(0.0, -0.14, text, transform=ax.transAxes, ha="left", va="top",
            fontsize=_THEME["font_sizes"]["source"], color=_THEME["muted"])


def _style_axes(ax: "Axes", *, grid_axis: str | None = "y") -> "Axes":
    """Apply the shared low-chartjunk styling: no top/right spines, faint grid."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(False)
    if grid_axis in ("x", "y", "both"):
        ax.grid(True, axis=grid_axis, color=_THEME["grid_color"], linewidth=0.6)
    tc = _THEME["text_color"]
    for spine in ax.spines.values():
        spine.set_edgecolor(tc)
    ax.tick_params(colors=tc, labelsize=_THEME["font_sizes"]["tick"])
    return ax


def _finish(ax, *, title=None, subtitle=None, source=None, xlabel=None,
            ylabel=None, grid_axis="y") -> "Axes":
    """Add titles/labels/source, apply the theme, tidy, and return the Axes."""
    _titles(ax, title, subtitle)
    sizes, tc = _THEME["font_sizes"], _THEME["text_color"]
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=sizes["label"], color=tc)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=sizes["label"], color=tc)
    _style_axes(ax, grid_axis=grid_axis)
    _source(ax, source)
    ax.figure.tight_layout()
    return ax


def _rotate_xticklabels(ax: "Axes", angle: int = 45, *,
                        max_len: int = 6, max_labels: int = 8) -> None:
    """Rotate x tick labels only when they are long or dense enough to overlap."""
    labels = [t.get_text() for t in ax.get_xticklabels()]
    if not labels:
        return
    longest = max((len(s) for s in labels), default=0)
    if longest > max_len or len(labels) > max_labels:
        for label in ax.get_xticklabels():
            label.set_rotation(angle)
            label.set_horizontalalignment("right")


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
