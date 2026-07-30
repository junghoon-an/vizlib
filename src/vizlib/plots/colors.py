"""Colour selection: palette picks, per-bar colours and luminance."""

from __future__ import annotations

import pandas as pd
import seaborn as sns
from matplotlib.colors import to_rgb

from .palettes import _TRAFFIC_LIGHT
from .theme import _THEME


def _base_color():
    """The first colour of the active palette."""
    return sns.color_palette(_THEME["palette"])[0]


def _hue_palette(values):
    """A palette sized to the number of hue levels (avoids seaborn warnings)."""
    n = max(int(pd.Series(values).nunique()), 1)
    return sns.color_palette(_THEME["palette"], n)


def _bar_colors(index, highlight):
    """Per-bar colours.

    With ``highlight`` (a label or list) the chosen bars use the accent colour
    and the rest are muted. Otherwise the default preset returns a single base
    colour, while the infographic/neon presets cycle the palette so bars read
    as a colourful dashboard.
    """
    if highlight is not None:
        keys = {highlight} if isinstance(highlight, str) else set(highlight)
        keys |= {str(k) for k in keys}
        accent, muted = _THEME["accent"], _THEME["muted"]
        return [accent if (idx in keys or str(idx) in keys) else muted
                for idx in index]
    if _THEME.get("bold_labels"):  # dashboard presets -> colourful bars
        pal = sns.color_palette(_THEME["palette"], max(len(index), 1))
        return [pal[i % len(pal)] for i in range(len(index))]
    return _base_color()


def _luminance(rgba) -> float:
    """Relative luminance of an RGBA/tuple colour in [0, 1]."""
    r, g, b = to_rgb(rgba[:3] if len(rgba) >= 3 else rgba)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _stack_colors(cats):
    """Colours for stacked bands: traffic-light for low/med/high, else palette."""
    low, med, high = {"low", "l"}, {"medium", "med", "m"}, {"high", "h"}
    names = [str(c).strip().lower() for c in cats]
    rank = {**{k: 0 for k in low}, **{k: 1 for k in med}, **{k: 2 for k in high}}
    if all(n in rank for n in names):
        return [_TRAFFIC_LIGHT[rank[n]] for n in names]
    return list(sns.color_palette(_THEME["palette"], len(cats)))
