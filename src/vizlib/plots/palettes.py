"""Palette constants and preset configuration data (no logic).

The single source of the style *data* the theme layer resolves: qualitative
palettes and the ``_DEFAULTS``/``_PRESETS`` dicts. ``theme.py`` turns these
into the live, mutable ``_THEME``.
"""

from __future__ import annotations

# Vivid qualitative palette for the "infographic" preset. NOT fully
# colorblind-safe (red/green, red/orange adjacencies) — a documented tradeoff.
_VIVID_PALETTE = [
    "#EE3524", "#27AAE1", "#FDB913", "#F58220", "#22B573",
    "#92278F", "#17A398", "#1B3A5B", "#EC008C",
]
# Sequential traffic-light scale for ordered low -> medium -> high data.
_TRAFFIC_LIGHT = ["#27AAE1", "#FDB913", "#EE3524"]

# Bright neon palette for the "neon" preset — a dark-navy dashboard look. Like
# ``infographic`` it is NOT colorblind-safe; it is a presentation style.
_NEON_PALETTE = [
    "#FF4F8B", "#37C7E6", "#4CE0B3", "#9B8CFF", "#FFE45E",
    "#F58220", "#C86DD7", "#2ED1A2", "#5B8DEF",
]

# Shared defaults. Read by every plot so a checklist-compliant figure needs no
# configuration; overlaid per preset by _PRESETS.
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
    "neon": {
        "palette": list(_NEON_PALETTE),
        "accent": "#FF4F8B",
        "muted": "#7A7AA8",
        "text_color": "#EDEDF7",      # near-white, high contrast on navy
        "grid_color": "#3A3A63",      # faint light gridlines on the dark panel
        "background": "#1B1B3E",      # deep indigo dashboard background
        "linewidth": 2.6,
        "hide_all_spines": True,
        "show_grid": True,            # thin gridlines like the reference
        "bold_labels": True,
        "swatch_legend": True,
        "font_sizes": {"title": 18, "subtitle": 13, "label": 13, "tick": 11, "source": 9},
    },
}
