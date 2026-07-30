"""Live theme state and the ``set_theme`` entry point.

The mutable ``_THEME`` dict is the single source of truth every plot reads
from. ``set_theme`` only ever mutates it *in place* (``clear``/``update``),
so the submodules can ``from .theme import _THEME`` and all observe the same
live configuration. The style *data* lives in :mod:`vizlib.plots.palettes`.
"""

from __future__ import annotations

import seaborn as sns

from .palettes import _DEFAULTS, _PRESETS

_THEME: dict = {**_DEFAULTS, "font_sizes": dict(_DEFAULTS["font_sizes"])}


def _surface_color():
    """Background for small label boxes / separators.

    The active theme's background when one is set (e.g. the dark ``neon``
    navy), otherwise plain white — so callout boxes stay legible on either.
    """
    return _THEME.get("background") or "white"


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
    font_sizes: dict | None = None,
) -> None:
    """Configure the look shared by every plot.

    Pass ``style_preset`` to switch the whole look at once:

    - ``"default"``: the colorblind- and grayscale-legible checklist style.
    - ``"infographic"``: a bold vivid dashboard on white. **Not colorblind-safe.**
    - ``"neon"``: the same bold chrome on a dark navy background with a neon
      palette and light text. **Not colorblind-safe.**

    Setting a preset resets the theme to that preset; any other argument you
    pass is then applied on top, so every knob stays overridable. ``font_sizes``
    is a partial mapping of ``{"title"|"subtitle"|"label"|"tick"|"source":
    points}`` merged over the active sizes. Called with no ``style_preset`` it
    just updates the values you provide. Returns ``None``.
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
    if font_sizes:
        _THEME["font_sizes"].update(font_sizes)
    sns.set_theme(
        context=_THEME["context"], style=_THEME["style"], palette=_THEME["palette"]
    )
