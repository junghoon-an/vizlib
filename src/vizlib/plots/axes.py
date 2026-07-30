"""Axis appearance: preset chrome (spines/grid/background/ticks) and rotation."""

from __future__ import annotations

from matplotlib.axes import Axes

from .theme import _THEME


def _style_axes(ax: "Axes", *, grid_axis: str | None = "y") -> "Axes":
    """Apply the active preset's chrome: spines, background, grid, tick fonts.

    The default preset hides top/right spines and keeps a faint grid; the
    dashboard presets hide every spine (the chart bleeds) and paint the
    background.
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


def _rotate_xticklabels(ax: "Axes", angle: int = 45, *,
                        max_len: int = 6, max_labels: int = 8) -> None:
    """Rotate x tick labels only when long or dense enough to overlap."""
    labels = [t.get_text() for t in ax.get_xticklabels()]
    if not labels:
        return
    longest = max((len(s) for s in labels), default=0)
    if longest > max_len or len(labels) > max_labels:
        for label in ax.get_xticklabels():
            label.set_rotation(angle)
            label.set_horizontalalignment("right")
