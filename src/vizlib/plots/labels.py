"""On-data value labels, y-tick ellipsis and swatch legends."""

from __future__ import annotations

from matplotlib.patches import Patch

from .theme import _THEME


def _draw_value_labels(ax, container, *, labels=None, fmt="%.0f", padding=5) -> None:
    """Label horizontal bars at a constant offset past each bar's tip.

    Always ``label_type="edge"`` with a fixed ``padding`` (points), so every
    label sits the same distance from its bar tip and clear of the left column
    labels. The bold presets only change weight and size; the enlarged size is
    trimmed when there are many bars so adjacent-row labels can't collide.
    """
    bold = _THEME.get("bold_labels")
    factor = 1.45 if bold else 1.0
    if len(container.patches) > 15:       # many rows -> keep labels from touching
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
    for full labels, so reserving space is preferred over truncating.
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
    legend = ax.legend(
        handles=handles, frameon=False, loc=loc, title=title,
        fontsize=_THEME["font_sizes"]["tick"],
        title_fontsize=_THEME["font_sizes"]["label"],
        handlelength=1.1, handleheight=1.1, borderaxespad=0.4,
    )
    tc = _THEME["text_color"]  # keep labels legible on a dark background
    for text in legend.get_texts():
        text.set_color(tc)
    if legend.get_title() is not None:
        legend.get_title().set_color(tc)
    return legend
