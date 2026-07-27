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
    import numpy as np
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.axes import Axes
    from matplotlib.colors import to_rgb
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch, Polygon
except ImportError as exc:  # pragma: no cover - exercised only without the deps
    raise ImportError(
        "vizlib.plots requires matplotlib and seaborn. "
        'Install them with: pip install "vizlib[plot]"'
    ) from exc

from .core import (
    _coerce_numeric,
    _looks_datetime,
    _numeric_frame,
    _require_dataframe,
    _require_series,
)

# Row counts above which scatter/pairplot auto-sample (deterministically) to
# stay responsive. Override or disable with an explicit ``sample=``.
_AUTO_SAMPLE_SCATTER = 20_000
_AUTO_SAMPLE_PAIRPLOT = 2_000

# Vivid qualitative palette for the "infographic" preset (Section A). It is
# NOT fully colorblind-safe (red/green, red/orange adjacencies) — that is the
# documented tradeoff of the look; ``colorblind`` stays the default.
_VIVID_PALETTE = [
    "#EE3524", "#27AAE1", "#FDB913", "#F58220", "#22B573",
    "#92278F", "#17A398", "#1B3A5B", "#EC008C",
]
# Sequential traffic-light scale for ordered low -> medium -> high data.
_TRAFFIC_LIGHT = ["#27AAE1", "#FDB913", "#EE3524"]

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
    "donut",
]

# Shared, mutable defaults. Updated by set_theme(); read by every plot so a
# checklist-compliant figure needs no configuration and stays consistent.
_DEFAULTS: dict = {
    "style_preset": "default",
    "palette": "colorblind",   # colorblind- and luminance-separated
    "context": "notebook",
    "style": "whitegrid",
    "figsize": (8, 5),
    "dpi": 110,
    "accent": "#1a5fb4",       # action color for highlighted marks
    "muted": "#b6b6b6",        # de-emphasis gray
    "text_color": "#1a1a1a",   # near-black, high contrast on white
    "grid_color": "#dcdcdc",   # faint gray gridlines
    "background": None,        # None -> matplotlib default (transparent)
    "linewidth": None,         # None -> matplotlib default line width
    "hide_all_spines": False,  # default hides only top/right
    "show_grid": True,
    "bold_labels": False,      # bold, on-data value labels
    "swatch_legend": False,    # colored-swatch legends
    "font_sizes": {"title": 15, "subtitle": 12, "label": 11, "tick": 10, "source": 8},
}

# Preset overlays applied on top of _DEFAULTS by set_theme(style_preset=...).
_PRESETS: dict = {
    "default": {},
    "infographic": {
        "palette": list(_VIVID_PALETTE),
        "accent": "#EE3524",
        "text_color": "#1A1A1A",
        "grid_color": "#ECECEC",
        "background": "#FFFFFF",
        "linewidth": 2.6,
        "hide_all_spines": True,
        "show_grid": False,
        "bold_labels": True,
        "swatch_legend": True,
        "font_sizes": {"title": 18, "subtitle": 13, "label": 13, "tick": 11, "source": 9},
    },
}

_THEME: dict = {**_DEFAULTS, "font_sizes": dict(_DEFAULTS["font_sizes"])}


def set_theme(
    *,
    style_preset: str | None = None,
    palette=None,
    context: str | None = None,
    style: str | None = None,
    figsize: tuple[float, float] | None = None,
    dpi: int | None = None,
    accent: str | None = None,
    muted: str | None = None,
    text_color: str | None = None,
    grid_color: str | None = None,
    background: str | None = None,
    linewidth: float | None = None,
    title_size: float | None = None,
    subtitle_size: float | None = None,
    label_size: float | None = None,
    tick_size: float | None = None,
    source_size: float | None = None,
) -> None:
    """Configure the look shared by every plot in this module.

    Pass ``style_preset`` to switch the whole look at once:

    - ``"default"`` (the out-of-the-box look): the colorblind- and
      grayscale-legible, low-chartjunk checklist style.
    - ``"infographic"``: a bold, vivid dashboard look — saturated palette,
      large bold on-data labels, borderless white-background chrome, thicker
      lines, gradient area fills and swatch legends. **This palette is not
      fully colorblind-safe** (it uses red/green and red/orange adjacencies);
      prefer the default for analytical work and keep this for presentation.

    Setting a preset resets the theme to that preset's look; any other
    argument you pass is then applied on top, so every knob stays
    overridable. Called with no ``style_preset`` it just updates the
    individual values you provide. Returns ``None``.
    """
    if style_preset is not None:
        if style_preset not in _PRESETS:
            raise ValueError(
                f"unknown style_preset {style_preset!r}; "
                f"choose from {sorted(_PRESETS)}"
            )
        base = {**_DEFAULTS, "font_sizes": dict(_DEFAULTS["font_sizes"])}
        overlay = _PRESETS[style_preset]
        base.update({k: v for k, v in overlay.items() if k != "font_sizes"})
        if "font_sizes" in overlay:
            base["font_sizes"] = dict(overlay["font_sizes"])
        base["style_preset"] = style_preset
        _THEME.clear()
        _THEME.update(base)

    scalar = {
        "palette": palette, "context": context, "style": style,
        "figsize": figsize, "dpi": dpi, "accent": accent, "muted": muted,
        "text_color": text_color, "grid_color": grid_color,
        "background": background, "linewidth": linewidth,
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
    as_percent: bool = False,
    fmt: str | None = None,
    label_padding: int = 5,
    max_label_chars: int | None = None,
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
    the rest muted. Set ``as_percent=True`` to plot each category's share of
    the total; ``precision`` sets the decimals and ``fmt`` overrides the label
    format string. Labels always sit a fixed ``label_padding`` (points) past
    each bar's tip — clear of the y-axis category labels — and the infographic
    preset only makes them larger and bold. vizlib-owned figures reserve the
    left margin against the active fonts automatically, so long category names
    are not clipped; ``max_label_chars`` optionally ellipsizes very long names
    as a last resort (off by default — reserving space is preferred). Returns
    the ``Axes``; the input is never mutated. The default ``title`` is a
    neutral placeholder — override it with your finding.
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

    if as_percent:
        total = counts.sum()
        counts = counts / total * 100 if total else counts * 0.0

    plot = counts.iloc[::-1]  # biggest on top for a horizontal bar
    ax = _new_ax(ax, min_height=0.4 * len(plot) + 1)  # tall enough per row
    kwargs.setdefault("color", _bar_colors(plot.index, highlight))
    if _THEME.get("linewidth"):
        kwargs.setdefault("edgecolor", "white")
    container = ax.barh([str(i) for i in plot.index], plot.to_numpy(), **kwargs)
    _ellipsize_yticklabels(ax, max_label_chars)
    biggest = float(plot.max())
    # zero-based, with right headroom so edge labels never clip
    ax.set_xlim(0, biggest * 1.18 if biggest else 1)

    name = series.name if series.name is not None else "value"
    grid_axis, xlabel = "x", ("% of total" if as_percent else "count")
    if value_labels:
        default_fmt = f"%.{precision}f%%" if as_percent else f"%.{precision}f"
        _draw_value_labels(ax, container, fmt=fmt or default_fmt,
                           padding=label_padding)
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
        numeric = _numeric_frame(df)
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
    sample: int | None = None,
    random_state: int = 0,
    annotations=None,
    title: str | None = None,
    subtitle: str | None = None,
    source: str | None = None,
    ax: "Axes | None" = None,
    **kwargs,
) -> "Axes":
    """Plot the relationship between two numeric columns.

    ``x`` and ``y`` are coerced to numeric (currency/thousands/``%`` stripped),
    so numeric-looking string columns just work. Optionally colour points by
    ``hue`` and overlay a single accent-colored regression line with
    ``reg=True``. Rows missing any plotted value are dropped (never in place).
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
        sns.regplot(data=sub, x=x, y=y, ax=ax, scatter=False, color=_THEME["accent"])
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
    ax: "Axes | None" = None,
    **kwargs,
) -> "Axes":
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


def missing_bar(
    df: pd.DataFrame,
    *,
    highlight=None,
    value_labels: bool = True,
    precision: int = 1,
    fmt: str | None = None,
    label_padding: int = 5,
    max_label_chars: int | None = None,
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
    the chosen column(s); ``precision`` sets the percent decimals and ``fmt``
    overrides the label format. Labels sit a fixed ``label_padding`` (points)
    past each bar's tip — clear of the column names on the left — and the
    infographic preset only makes them larger and bold. The left margin is
    reserved against the active fonts automatically so long column names are
    not clipped; ``max_label_chars`` optionally ellipsizes very long names as a
    last resort (off by default). Returns the ``Axes``; the input is untouched.
    """
    _require_dataframe(df)
    if df.shape[1] == 0:
        raise ValueError("no columns to plot")
    n = len(df)
    pct = (df.isna().sum() / n * 100) if n else df.isna().sum() * 0.0
    pct = pct.sort_values(ascending=False)

    plot = pct.iloc[::-1]  # biggest on top
    ax = _new_ax(ax, min_height=0.4 * len(plot) + 1)  # tall enough per row
    kwargs.setdefault("color", _bar_colors(plot.index, highlight))
    if _THEME.get("linewidth"):
        kwargs.setdefault("edgecolor", "white")
    container = ax.barh([str(i) for i in plot.index], plot.to_numpy(), **kwargs)
    _ellipsize_yticklabels(ax, max_label_chars)

    grid_axis, xlabel = "x", "% missing"
    if value_labels:
        labels = [(fmt % v) if fmt else f"{v:.{precision}f}%" for v in plot.to_numpy()]
        _draw_value_labels(ax, container, labels=labels, padding=label_padding)
        # right headroom so edge labels clear the axis; scale hidden anyway
        ax.set_xlim(0, max(float(plot.max()) * 1.12, 1.0))
        ax.xaxis.set_visible(False)
        ax.spines["bottom"].set_visible(False)
        grid_axis, xlabel = None, None
    else:
        ax.set_xlim(0, 100)  # honest 0–100 scale when the axis is shown
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
    ax: "Axes | None" = None,
    **kwargs,
) -> "Axes":
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
    wedges, _texts, autotexts = ax.pie(
        values, colors=colors, explode=explode_arr, counterclock=False,
        autopct=lambda p: f"{p:.0f}%", pctdistance=0.78,
        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2),
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
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"),
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


# --- internal helpers -------------------------------------------------------

def _new_ax(ax: "Axes | None", *, min_height: float | None = None) -> "Axes":
    """Return ``ax`` or a fresh, vizlib-owned figure sized from the theme.

    A freshly-created figure uses matplotlib's *constrained* layout engine so
    the axes automatically reserve room for tick labels rendered at the
    currently active theme's fonts — and re-measure at draw time. That makes
    the margins self-correct whenever the style changes (e.g. the larger,
    bolder ``infographic`` fonts), instead of assuming one fixed size. The
    figure is tagged as vizlib-owned so :func:`_finalize_layout` knows it may
    manage the layout; when the caller passes their own ``ax`` we leave their
    figure untouched.

    ``min_height`` grows a freshly-created figure so horizontal-bar rows stay
    tall enough for their labels; it is ignored when the caller passes ``ax``.
    """
    if ax is None:
        width, height = _THEME["figsize"]
        if min_height is not None:
            height = max(height, min_height)
        fig, ax = plt.subplots(figsize=(width, height), dpi=_THEME["dpi"],
                               constrained_layout=True)
        fig._vizlib_owned = True  # we created it -> we may manage its layout
    return ax


def _finalize_layout(ax: "Axes") -> None:
    """Resolve the figure layout without hijacking a caller-supplied one.

    vizlib-owned figures (see :func:`_new_ax`) already run a constrained
    layout engine that reserves space measured against the current theme's
    fonts, so there is nothing to do — calling ``tight_layout`` on top would
    fight that engine and warn. When the caller supplied their own ``ax`` we
    deliberately leave their figure's layout alone (honouring ``ax=`` means
    not overriding their composition). This helper centralises that contract
    so every plot treats layout the same way.
    """
    return None


def _base_color():
    """The first colour of the active colorblind-safe palette."""
    return sns.color_palette(_THEME["palette"])[0]


def _hue_palette(values):
    """A palette sized to the number of hue levels (avoids seaborn warnings)."""
    n = max(int(pd.Series(values).nunique()), 1)
    return sns.color_palette(_THEME["palette"], n)


def _bar_colors(index, highlight):
    """Per-bar colours.

    With ``highlight`` (a label or list), highlighted bars use the accent
    colour and the rest are muted. Otherwise the default preset returns a
    single base colour, while the infographic preset cycles the vivid palette
    so bars read as a colourful dashboard.
    """
    if highlight is not None:
        keys = {highlight} if isinstance(highlight, str) else set(highlight)
        keys |= {str(k) for k in keys}
        accent, muted = _THEME["accent"], _THEME["muted"]
        return [accent if (idx in keys or str(idx) in keys) else muted
                for idx in index]
    if _THEME.get("bold_labels"):  # infographic preset -> colourful bars
        pal = sns.color_palette(_THEME["palette"], max(len(index), 1))
        return [pal[i % len(pal)] for i in range(len(index))]
    return _base_color()


def _luminance(rgba) -> float:
    """Relative luminance of an RGBA/tuple colour in [0, 1]."""
    r, g, b = to_rgb(rgba[:3] if len(rgba) >= 3 else rgba)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _draw_value_labels(ax, container, *, labels=None, fmt="%.0f", padding=5) -> None:
    """Label horizontal bars at a constant offset past each bar's tip.

    Preset-agnostic placement: always ``label_type="edge"`` with a fixed
    ``padding`` (in points), so every label sits the same distance from its
    bar tip and clear of the left y-axis column labels — never over them. The
    infographic preset only changes the weight and size (dark high-contrast
    text either way). The enlarged size is trimmed when there are many bars so
    adjacent-row labels can't collide.
    """
    bold = _THEME.get("bold_labels")
    n_bars = len(container.patches)
    factor = 1.45 if bold else 1.0
    if n_bars > 15:                       # many rows -> keep labels from touching
        factor = min(factor, 1.1)
    fs = _THEME["font_sizes"]["label"] * factor
    label_kw = {"labels": labels} if labels is not None else {"fmt": fmt}
    ax.bar_label(
        container, label_type="edge", padding=padding, fontsize=fs,
        fontweight=("bold" if bold else "normal"),
        color=_THEME["text_color"], **label_kw,
    )


def _ellipsize_yticklabels(ax, max_chars: int | None) -> None:
    """Optionally shorten long y-tick labels to ``max_chars`` with an ellipsis.

    Off by default (``max_chars is None``): the constrained layout already
    reserves room for full labels, so reserving space is preferred over
    truncating. When set, each label longer than ``max_chars`` keeps its
    leading characters and gains a trailing ``…``. The tick positions are
    fixed by the categorical bars, so re-setting the label text is safe.
    """
    if max_chars is None or max_chars < 1:
        return
    new = []
    for tick in ax.get_yticklabels():
        s = tick.get_text()
        new.append(s if len(s) <= max_chars
                   else (s[: max_chars - 1] + "…" if max_chars > 1 else "…"))
    ax.set_yticks(ax.get_yticks())  # pin positions so set_yticklabels won't warn
    ax.set_yticklabels(new)


def _swatch_legend(ax, labels, colors, *, title=None, loc="best"):
    """Draw a frameless legend as a row of colored swatches with labels."""
    handles = [Patch(facecolor=c, edgecolor="none", label=str(lab))
               for lab, c in zip(labels, colors)]
    return ax.legend(
        handles=handles, frameon=False, loc=loc, title=title,
        fontsize=_THEME["font_sizes"]["tick"],
        title_fontsize=_THEME["font_sizes"]["label"],
        handlelength=1.1, handleheight=1.1, borderaxespad=0.4,
    )


def _to_num(seq):
    """Numeric view of a sequence for gradient extents / nearest-point math.

    Datetimes (and date-like strings) map through ``date2num`` so callout
    ``x`` values can be given as strings against a datetime axis.
    """
    arr = pd.Series(list(seq))
    if pd.api.types.is_datetime64_any_dtype(arr):
        return mdates.date2num(arr.to_numpy())
    num = pd.to_numeric(arr, errors="coerce")
    if num.notna().any():
        return num.to_numpy(dtype=float)
    return mdates.date2num(pd.to_datetime(arr, errors="coerce").to_numpy())


def _stack_colors(cats):
    """Colours for stacked-area bands: traffic-light for low/med/high, else palette."""
    low, med, high = {"low", "l"}, {"medium", "med", "m"}, {"high", "h"}
    names = [str(c).strip().lower() for c in cats]
    rank = {**{k: 0 for k in low}, **{k: 1 for k in med}, **{k: 2 for k in high}}
    if all(n in rank for n in names):
        return [_TRAFFIC_LIGHT[rank[n]] for n in names]
    return list(sns.color_palette(_THEME["palette"], len(cats)))


def _gradient_fill(ax, xnum, y, color, *, baseline=0.0, alpha=0.85) -> None:
    """Fill under a curve with a vertical gradient from ``color`` to transparent.

    Builds a gradient image and clips it to the polygon between the line and
    ``baseline`` — mirrors the reference's gradient area panels.
    """
    xnum = np.asarray(xnum, dtype=float)
    y = np.asarray(y, dtype=float)
    if xnum.size == 0:
        return
    rgb = to_rgb(color)
    ramp = np.empty((256, 1, 4))
    ramp[:, :, :3] = rgb
    ramp[:, :, 3] = np.linspace(0.0, alpha, 256)[:, None]  # transparent -> solid
    xmin, xmax = float(xnum.min()), float(xnum.max())
    ymin, ymax = float(min(baseline, y.min())), float(max(baseline, y.max()))
    image = ax.imshow(ramp, aspect="auto", origin="lower",
                      extent=[xmin, xmax, ymin, ymax], zorder=1)
    verts = [(xmin, baseline), *zip(xnum, y), (xmax, baseline)]
    clip = Polygon(verts, closed=True, facecolor="none", edgecolor="none")
    ax.add_patch(clip)
    image.set_clip_path(clip)


def _draw_callouts(ax, annotations, points_x=None, points_y=None) -> None:
    """Attach leader-line event callouts to points.

    Each item is ``(x, text)`` (y is read from the nearest plotted point) or
    ``(x, y, text)``. Rendered with a thin leader line and a small rounded box.
    """
    if not annotations:
        return
    px = _to_num(points_x) if points_x is not None else None
    accent, muted, tc = _THEME["accent"], _THEME["muted"], _THEME["text_color"]
    for item in annotations:
        if len(item) == 3:
            xv, yv, text = item
        elif len(item) == 2 and px is not None:
            xv, text = item
            idx = int(np.nanargmin(np.abs(px - _to_num([xv])[0])))
            xv, yv = list(points_x)[idx], list(points_y)[idx]
        else:
            raise ValueError("each annotation must be (x, text) or (x, y, text)")
        ax.annotate(
            str(text), xy=(xv, yv), xytext=(0, 34), textcoords="offset points",
            ha="center", va="bottom", fontsize=_THEME["font_sizes"]["tick"],
            color=tc, zorder=6,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=accent, lw=1.2),
            arrowprops=dict(arrowstyle="-", color=muted, lw=1.0),
        )


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
    """Apply the active preset's chrome: spines, background, grid, tick fonts.

    The default preset hides the top/right spines and keeps a faint grid; the
    infographic preset hides every spine (the chart bleeds), paints a white
    background and drops the grid.
    """
    sides = (("top", "right", "bottom", "left")
             if _THEME.get("hide_all_spines") else ("top", "right"))
    for side in sides:
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(False)
    if _THEME.get("show_grid", True) and grid_axis in ("x", "y", "both"):
        ax.grid(True, axis=grid_axis, color=_THEME["grid_color"], linewidth=0.6)
    background = _THEME.get("background")
    if background:
        ax.set_facecolor(background)
        ax.figure.set_facecolor(background)
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
    _finalize_layout(ax)
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
    """Coerce a Series to numeric and drop NaN.

    Strips currency symbols, thousands separators and stray ``%`` first (via
    the shared ``core._coerce_numeric``), so a numeric-looking string column
    plots without manual cleaning. Raises when nothing parses.
    """
    _require_series(series)
    values = _coerce_numeric(series).dropna()
    if values.empty:
        raise ValueError("no numeric values to plot")
    return values


def _maybe_sample(frame: pd.DataFrame, sample, auto_threshold: int, random_state: int):
    """Return a reproducible row subset, honouring an explicit or auto cap."""
    n = sample if sample is not None else (
        auto_threshold if len(frame) > auto_threshold else None
    )
    if n is not None and n < len(frame):
        return frame.sample(n=n, random_state=random_state)
    return frame


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
