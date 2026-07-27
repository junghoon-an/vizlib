"""Shared theme state and the ``set_theme`` entry point.

The mutable ``_THEME`` dict is the single source of truth every plot reads
from. ``set_theme`` only ever mutates it *in place* (``clear``/``update``),
so the other submodules can ``from .theme import _THEME`` and all observe the
same live configuration.
"""

from __future__ import annotations

import seaborn as sns

# Vivid qualitative palette for the "infographic" preset. It is NOT fully
# colorblind-safe (red/green, red/orange adjacencies) — that is the documented
# tradeoff of the look; ``colorblind`` stays the default.
_VIVID_PALETTE = [
    "#EE3524", "#27AAE1", "#FDB913", "#F58220", "#22B573",
    "#92278F", "#17A398", "#1B3A5B", "#EC008C",
]
# Sequential traffic-light scale for ordered low -> medium -> high data.
_TRAFFIC_LIGHT = ["#27AAE1", "#FDB913", "#EE3524"]

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


def _resolved_theme(preset_name: str) -> dict:
    """The fully-merged theme dict for a preset (defaults + its overlay)."""
    base = {**_DEFAULTS, "font_sizes": dict(_DEFAULTS["font_sizes"])}
    overlay = _PRESETS[preset_name]
    base.update({k: v for k, v in overlay.items() if k != "font_sizes"})
    if "font_sizes" in overlay:
        base["font_sizes"] = dict(overlay["font_sizes"])
    return base


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
        base = _resolved_theme(style_preset)
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
