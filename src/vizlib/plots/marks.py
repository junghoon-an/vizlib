"""Colour, direct-label, legend, gradient and callout helpers.

The visual building blocks the charts share: palette selection, per-bar
colours, on-data value labels, swatch legends, gradient area fills and
leader-line callouts.
"""

from __future__ import annotations

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch, Polygon

from .theme import _THEME, _TRAFFIC_LIGHT


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

    Off by default (``max_chars is None``): the layout already reserves room
    for full labels, so reserving space is preferred over truncating. When
    set, each label longer than ``max_chars`` keeps its leading characters and
    gains a trailing ``…``. The tick positions are fixed by the categorical
    bars, so re-setting the label text is safe.
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
