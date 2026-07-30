"""Numeric-axis helpers, gradient area fills and leader-line callouts."""

from __future__ import annotations

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.patches import Polygon

from .theme import _THEME, _surface_color


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


def _gradient_fill(ax, xnum, y, color, *, baseline=0.0, alpha=0.85) -> None:
    """Fill under a curve with a vertical gradient from ``color`` to transparent.

    Builds a gradient image and clips it to the polygon between the line and
    ``baseline`` — mirrors the reference's gradient area panels.
    """
    xnum = np.asarray(xnum, dtype=float)
    y = np.asarray(y, dtype=float)
    if xnum.size == 0:
        return
    ramp = np.empty((256, 1, 4))
    ramp[:, :, :3] = to_rgb(color)
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

    Each item is ``(x, text)`` (y read from the nearest plotted point) or
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
            bbox=dict(boxstyle="round,pad=0.35", fc=_surface_color(),
                      ec=accent, lw=1.2),
            arrowprops=dict(arrowstyle="-", color=muted, lw=1.0),
        )
