"""Shared horizontal-bar renderer for ``bar`` and ``missing_bar``.

The two public charts differ only in how they turn a DataFrame/Series into a
plotted Series and its value-label strings; everything downstream — colours,
direct labels, the hidden redundant value axis and the worst-case margin
reservation — is identical and lives here.
"""

from __future__ import annotations

from .chrome import _finish, _new_ax
from .colors import _bar_colors
from .labels import _draw_value_labels, _ellipsize_yticklabels
from .margins import _reserve_hbar_margins
from .theme import _THEME


def _hbar(plot, *, highlight, value_strings, xlim, xlabel, ylabel, title,
          subtitle, source, label_padding, max_label_chars, ax, **kwargs):
    """Draw ``plot`` (biggest on top) as a horizontal bar chart and return the Axes.

    ``value_strings`` is the list of on-data labels, or ``None`` to keep the
    value axis visible instead. Reserves explicit worst-case margins so the
    axes never move on a style switch.
    """
    ax = _new_ax(ax, min_height=0.4 * len(plot) + 1, constrained=False)
    kwargs.setdefault("color", _bar_colors(plot.index, highlight))
    if _THEME.get("linewidth"):
        kwargs.setdefault("edgecolor", "white")
    container = ax.barh([str(i) for i in plot.index], plot.to_numpy(), **kwargs)
    _ellipsize_yticklabels(ax, max_label_chars)
    ax.set_xlim(*xlim)

    grid_axis = "x"
    if value_strings is not None:
        _draw_value_labels(ax, container, labels=value_strings, padding=label_padding)
        ax.xaxis.set_visible(False)              # value axis is redundant now
        ax.spines["bottom"].set_visible(False)
        grid_axis, xlabel = None, None
    ax.tick_params(length=0)                     # no tick marks on bar charts

    ax = _finish(ax, title=title, subtitle=subtitle, source=source,
                 xlabel=xlabel, ylabel=ylabel, grid_axis=grid_axis)
    _reserve_hbar_margins(ax, value_strings or [], n_bars=len(plot),
                          max_label_chars=max_label_chars)
    return ax
