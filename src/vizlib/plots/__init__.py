"""Matplotlib/seaborn-backed plots for vizlib — opinionated EDA figures.

This package is the graphical counterpart to :mod:`vizlib.core`. The core
functions work with pandas alone; importing *this* package pulls in
matplotlib and seaborn (installed by default as of 0.3.0). It is kept out
of ``vizlib/__init__.py``'s import chain, so a bare ``import vizlib`` stays
fast — reach the plotting API explicitly with ``import vizlib.plots`` or
``from vizlib import plots``.

Every function follows the same contract as the core: it never mutates its
input, and it *returns* the matplotlib ``Axes`` it drew on (or a
``Figure``/seaborn grid for multi-panel plots) rather than calling
``plt.show()``. Pass ``ax=`` to compose into an existing figure.

The defaults follow the Evergreen & Emery Data Visualization Checklist: a
left-justified descriptive title, an optional muted subtitle and source
caption, a readable font hierarchy, dark high-contrast text, muted
gridlines, direct data labels on bars (with the redundant value axis
hidden), frequency/median-ordered categories, honest zero-based magnitude
axes, and a colorblind- and grayscale-legible palette. The mechanical
guidelines are enforced automatically; the interpretive ones are exposed as
hooks — ``title``, ``subtitle``, ``source`` and ``highlight`` — with neutral
defaults. Override ``title``/``subtitle`` with your actual finding; vizlib
never fabricates a takeaway the data doesn't support.

The implementation is split across small submodules (``theme``, ``chrome``,
``margins``, ``marks``, ``data`` and one module per chart family); this file
re-exports the public API so the whole layer is still reached as
``vizlib.plots``.
"""

from __future__ import annotations

try:  # matplotlib + seaborn ship as regular dependencies
    import matplotlib  # noqa: F401
    import seaborn  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised only without the deps
    raise ImportError(
        "vizlib.plots requires matplotlib and seaborn. "
        'Install them with: pip install "vizlib[plot]"'
    ) from exc

from .bars import bar, missing_bar
from .distribution import box, distribution, hist
from .donut import donut
from .matrix import correlation_heatmap, missing_matrix, pairplot
from .relational import line, scatter
from .theme import set_theme

# Internal names re-exported for tests and advanced tuning (see the theme and
# margins submodules). They stay importable as ``vizlib.plots._NAME`` so the
# refactor into submodules is invisible to callers.
from .margins import _HBAR_LEFT_CAP, _HBAR_RIGHT_CAP, _text_width_px  # noqa: F401
from .theme import (  # noqa: F401
    _DEFAULTS,
    _NEON_PALETTE,
    _PRESETS,
    _THEME,
    _TRAFFIC_LIGHT,
    _VIVID_PALETTE,
    _resolved_theme,
)

__all__ = [
    "set_theme",
    "bar",
    "hist",
    "distribution",
    "box",
    "scatter",
    "line",
    "correlation_heatmap",
    "missing_bar",
    "missing_matrix",
    "pairplot",
    "donut",
]
