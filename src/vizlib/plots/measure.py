"""Renderer-independent text measurement for worst-case margin reservation.

Widths are taken off a :class:`~matplotlib.textpath.TextPath`, so no renderer
or active-figure state is touched — that is what lets us reserve space for an
*inactive* preset's larger fonts without disturbing the active render.
"""

from __future__ import annotations

from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath

from .palettes import _PRESETS
from .theme import _THEME, _resolved_theme


def _text_width_px(s: str, *, size: float, weight, dpi: float) -> float:
    """Rendered width of ``s`` in pixels at a given font size/weight."""
    if not s:
        return 0.0
    tp = TextPath((0, 0), str(s), prop=FontProperties(size=size, weight=weight))
    return tp.get_extents().width * dpi / 72.0


def _label_factor(bold: bool, n_bars: int) -> float:
    """Value-label size multiplier — mirror of the logic in _draw_value_labels."""
    factor = 1.45 if bold else 1.0
    if n_bars > 15:
        factor = min(factor, 1.1)
    return factor


def _hbar_font_regimes(n_bars: int):
    """``(tick_regimes, label_regimes)`` across every preset and the active theme.

    Each regime is a ``(size, weight)`` pair. Enumerating *all* presets — not
    just the active ``_THEME`` — is the whole point: the reserved margin is a
    worst-case superset, so the axes stay put when the user switches styles.
    """
    themes = [_resolved_theme(name) for name in _PRESETS]
    themes.append(_THEME)  # also fit any custom overrides on the active theme
    tick, label = set(), set()
    for th in themes:
        sizes = th["font_sizes"]
        tick.add((sizes["tick"], "normal"))  # tick labels are never bold
        bold = bool(th.get("bold_labels"))
        label.add((sizes["label"] * _label_factor(bold, n_bars),
                   "bold" if bold else "normal"))
    return sorted(tick), sorted(label)


def _widest_px(labels, regimes, dpi) -> float:
    """Max rendered width (px) of any label over any font regime."""
    return max((_text_width_px(s, size=sz, weight=w, dpi=dpi)
                for s in labels for sz, w in regimes), default=0.0)
