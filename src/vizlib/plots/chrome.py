"""Figure creation, titles/captions and the shared ``_finish`` orchestration.

These build the vizlib-owned figure and place the left-justified title, muted
subtitle and source caption. Axis appearance lives in :mod:`vizlib.plots.axes`.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from .axes import _style_axes
from .options import Captions
from .theme import _THEME


def _new_ax(ax: "Axes | None", *, min_height: float | None = None,
            constrained: bool = True) -> "Axes":
    """Return ``ax`` or a fresh, vizlib-owned figure sized from the theme.

    A freshly-created figure uses matplotlib's *constrained* layout so the
    axes reserve room for tick labels at the active fonts and re-measure at
    draw time — margins self-correct on a style change.

    Horizontal-bar charts pass ``constrained=False``: constrained layout only
    measures the *active* artists, so it can't reserve room for an inactive
    preset's larger fonts. Those charts reserve explicit worst-case margins
    via :func:`vizlib.plots.margins._reserve_hbar_margins` (tagged
    ``_vizlib_manual_margins``) so the axes never move on a switch.

    The figure is tagged vizlib-owned so the layout helpers know they may
    manage it; a caller-supplied ``ax`` is left untouched. ``min_height`` grows
    a freshly-created figure so bar rows stay tall enough for their labels.
    """
    if ax is None:
        width, height = _THEME["figsize"]
        if min_height is not None:
            height = max(height, min_height)
        fig, ax = plt.subplots(figsize=(width, height), dpi=_THEME["dpi"],
                               constrained_layout=constrained)
        fig._vizlib_owned = True  # we created it -> we may manage its layout
        if not constrained:
            fig._vizlib_manual_margins = True
    return ax


def _titles(ax: "Axes", captions: Captions) -> None:
    """Set a left-justified title and, under it, a muted left-aligned subtitle."""
    title, subtitle = captions.title, captions.subtitle
    sizes, tc, muted = _THEME["font_sizes"], _THEME["text_color"], _THEME["muted"]
    if subtitle:
        ax.set_title(title or "", loc="left", fontsize=sizes["title"],
                     fontweight="bold", color=tc, pad=24)
        ax.text(0.0, 1.015, subtitle, transform=ax.transAxes, ha="left",
                va="bottom", fontsize=sizes["subtitle"], color=muted)
    elif title:
        ax.set_title(title, loc="left", fontsize=sizes["title"],
                     fontweight="bold", color=tc, pad=10)


def _source(ax: "Axes", captions: Captions) -> None:
    """Render a small, muted source caption in the lower-left, below the axes."""
    if not captions.source:
        return
    ax.text(0.0, -0.14, captions.source, transform=ax.transAxes, ha="left",
            va="top", fontsize=_THEME["font_sizes"]["source"],
            color=_THEME["muted"])


def _finish(ax, captions: Captions, *, xlabel=None, ylabel=None,
            grid_axis="y") -> "Axes":
    """Add captions/labels and apply the preset chrome; return the Axes.

    vizlib-owned figures run a constrained (or explicitly-reserved) layout, so
    there is nothing to finalise here; a caller-supplied figure's layout is
    deliberately left untouched.
    """
    _titles(ax, captions)
    sizes, tc = _THEME["font_sizes"], _THEME["text_color"]
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=sizes["label"], color=tc)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=sizes["label"], color=tc)
    _style_axes(ax, grid_axis=grid_axis)
    _source(ax, captions)
    return ax
