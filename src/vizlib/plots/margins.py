"""Explicit worst-case margin reservation for horizontal-bar charts.

Constrained layout only ever measures the *active* artists, so it cannot
reserve room for an inactive preset's larger fonts. These helpers instead
measure the widest column name and value label under *every* preset's fonts
(via :class:`~matplotlib.textpath.TextPath`, independent of the active
render) and reserve the maximum with ``subplots_adjust`` — so the axes never
move on a style switch and the names can never overlap the bars.
"""

from __future__ import annotations

from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath

from .theme import _PRESETS, _THEME, _resolved_theme

# Gaps are in points (font-size-independent); caps bound how much of the figure
# the names / value labels may claim before we ellipsize instead.
_HBAR_TICK_PAD_PTS = 8      # gap between the widest column name and the axes
_HBAR_LABEL_PAD_PTS = 8     # gap kept past the widest value label
_HBAR_LEFT_CAP = 0.40       # names never claim more than this fraction of width
_HBAR_RIGHT_CAP = 0.32      # nor value labels this much on the right


def _text_width_px(s: str, *, size: float, weight, dpi: float) -> float:
    """Rendered width of ``s`` in pixels at a given font size/weight.

    Measured off a :class:`~matplotlib.textpath.TextPath`, so no renderer or
    active-figure state is touched — the width is independent of whichever
    style is currently active. That is what lets us reserve space for an
    *inactive* preset's larger fonts without disturbing the active render.
    """
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


def _fit_ytick_labels(ax, labels, avail_px, tick_regimes, dpi, max_label_chars):
    """Ellipsize y labels so the widest fits ``avail_px`` at the largest font.

    Width grows monotonically with the character budget, so the largest budget
    that still fits is found with a binary search (a linear scan would rebuild
    a TextPath for every dropped character of a very long name).
    """
    max_size = max(sz for sz, _ in tick_regimes)
    longest = max((len(s) for s in labels), default=0)
    hi = min(max_label_chars, longest) if max_label_chars else longest

    def _trunc(k):
        return [s if len(s) <= k else (s[: k - 1] + "…" if k > 1 else "…")
                for s in labels]

    lo, best = 1, 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if _widest_px(_trunc(mid), [(max_size, "normal")], dpi) <= avail_px:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    chosen = _trunc(best)
    ax.set_yticks(ax.get_yticks())  # pin positions so set_yticklabels won't warn
    ax.set_yticklabels(chosen)
    return chosen


def _reserve_hbar_margins(ax, value_labels, *, n_bars, max_label_chars=None) -> None:
    """Reserve worst-case left margin (names) and right headroom (value labels).

    Only manages vizlib-owned horizontal-bar figures; a caller-supplied ``ax``
    is left untouched (honouring ``ax=``). Widths are measured against the
    largest font across *all* presets, so the axes position is identical no
    matter which style is active — the bars never shift on a theme switch and
    can never be overlapped by the y-axis column names. A pathologically long
    name that would exceed :data:`_HBAR_LEFT_CAP` is ellipsized rather than
    crushing the plot.
    """
    fig = ax.figure
    if not getattr(fig, "_vizlib_owned", False):
        return  # caller supplied ax: do not manage their margins
    dpi = fig.dpi
    fig_w_px = fig.get_figwidth() * dpi
    tick_regimes, label_regimes = _hbar_font_regimes(n_bars)

    y_labels = [t.get_text() for t in ax.get_yticklabels()]
    tick_pad = _HBAR_TICK_PAD_PTS * dpi / 72.0
    left = (_widest_px(y_labels, tick_regimes, dpi) + tick_pad) / fig_w_px
    if left > _HBAR_LEFT_CAP:  # guardrail: ellipsize instead of a crushed plot
        avail = _HBAR_LEFT_CAP * fig_w_px - tick_pad
        y_labels = _fit_ytick_labels(ax, y_labels, avail, tick_regimes, dpi,
                                      max_label_chars)
        left = (_widest_px(y_labels, tick_regimes, dpi) + tick_pad) / fig_w_px
    left = min(left, _HBAR_LEFT_CAP)

    right_reserve = 0.0
    if value_labels:
        label_pad = _HBAR_LABEL_PAD_PTS * dpi / 72.0
        right_reserve = min(
            (_widest_px(value_labels, label_regimes, dpi) + label_pad) / fig_w_px,
            _HBAR_RIGHT_CAP,
        )
    right = 1.0 - right_reserve
    if right - left < 0.2:  # always keep a usable plotting band
        right = min(left + 0.2, 0.98)

    # One-shot tidy fixes top/bottom (title, source) at the active fonts — run
    # after any ellipsis so labels already fit — then override left/right with
    # the cross-preset worst case. tight_layout leaves an inert placeholder
    # engine, so subplots_adjust sticks at draw time.
    fig.tight_layout()
    fig.subplots_adjust(left=left, right=right)
