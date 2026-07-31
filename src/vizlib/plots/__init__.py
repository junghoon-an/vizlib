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

The implementation is split across small, single-responsibility submodules
(configuration in ``palettes``/``theme``; shared visuals in ``colors``,
``labels``, ``annotate``, ``chrome``, ``axes``, ``measure`` and ``margins``;
one module per chart family); this file re-exports the public API so the whole
layer is still reached as ``vizlib.plots``.
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

from .bars import bar
from .box import box
from .distribution import distribution, hist
from .heatmap import correlation_heatmap, missing_matrix
from .line import line
from .missing_bar import missing_bar
from .options import Captions, ValueLabels
from .pairplot import pairplot
from .scatter import scatter
from .theme import set_theme

# Internal names re-exported for tests and advanced tuning. They stay
# importable as ``vizlib.plots._NAME`` so the split into submodules is
# invisible to callers.
from .margins import _HBAR_LEFT_CAP, _HBAR_RIGHT_CAP  # noqa: F401
from .measure import _text_width_px  # noqa: F401
from .palettes import (  # noqa: F401
    _DEFAULTS,
    _NEON_PALETTE,
    _PRESETS,
    _TRAFFIC_LIGHT,
    _VIVID_PALETTE,
)
from .theme import _THEME, _resolved_theme  # noqa: F401

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
    "Captions",
    "ValueLabels",
]
