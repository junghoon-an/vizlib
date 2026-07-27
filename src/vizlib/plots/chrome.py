"""Figure creation, titles/captions, axis chrome and layout finalisation.

These helpers are shared by every chart: they build the vizlib-owned figure,
apply the active preset's spines/grid/background/tick fonts, and place the
left-justified title, muted subtitle and source caption.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from .theme import _THEME


def _new_ax(ax: "Axes | None", *, min_height: float | None = None,
            constrained: bool = True) -> "Axes":
    """Return ``ax`` or a fresh, vizlib-owned figure sized from the theme.

    By default a freshly-created figure uses matplotlib's *constrained* layout
    engine so the axes automatically reserve room for tick labels rendered at
    the active theme's fonts — and re-measure at draw time. That makes the
    margins self-correct whenever the style changes.

    Horizontal-bar charts pass ``constrained=False``: constrained layout only
    ever measures the *active* artists, so it cannot reserve room for an
    inactive preset's larger fonts. Those charts instead reserve explicit,
    worst-case margins with :func:`vizlib.plots.margins._reserve_hbar_margins`
    (tagged ``_vizlib_manual_margins``) so the axes never move on a switch.

    The figure is tagged vizlib-owned so the layout helpers know they may
    manage it; when the caller passes their own ``ax`` we leave it untouched.
    ``min_height`` grows a freshly-created figure so horizontal-bar rows stay
    tall enough for their labels; it is ignored when the caller passes ``ax``.
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
