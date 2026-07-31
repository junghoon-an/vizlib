"""Plain helper functions shared across the plotting-layer test modules.

Not a test module itself (no ``test_`` prefix), so pytest won't collect it;
the sibling test files import from it. The Agg backend is already selected by
this directory's ``conftest.py``, which pytest imports first.
"""

import pandas as pd
from matplotlib.axes import Axes

from vizlib import plots


def _assert_axes(obj):
    assert isinstance(obj, Axes)


def _many_cat_series(n_cats=20):
    """A categorical Series with many distinct-count categories for ``bar``.

    ``bar`` draws the same horizontal-bar renderer ``missing_bar`` used, so
    these Series drive the shared layout/margin logic.
    """
    data = []
    for i in range(n_cats):
        data += [f"cat_{i:02d}"] * (i + 1)  # distinct counts -> bars of varied length
    return pd.Series(data)


def _long_name_series(n_cats=7):
    """Long category names — the case that used to overflow the left margin."""
    data = []
    for i in range(n_cats):
        name = f"a_really_long_descriptive_category_name_number_{i:02d}"
        data += [name] * (i + 2)
    return pd.Series(data)


def _label_gaps(ax):
    """Pixel gap between each value label and its bar's tip (drawn order)."""
    ax.figure.canvas.draw()
    gaps = []
    for text, patch in zip(ax.texts, ax.patches):
        tip_px = ax.transData.transform((patch.get_width(), 0))[0]
        gaps.append(text.get_window_extent().x0 - tip_px)
    return gaps


def _bboxes_overlap(a, b, tol=0.5):
    return not (a.x1 <= b.x0 + tol or b.x1 <= a.x0 + tol
                or a.y1 <= b.y0 + tol or b.y1 <= a.y0 + tol)


def _widest_ytick_px_under_infographic(labels, dpi):
    """Widest y-tick label in px measured under the infographic tick font."""
    info = plots._resolved_theme("infographic")["font_sizes"]["tick"]
    return max(plots._text_width_px(s, size=info, weight="normal", dpi=dpi)
               for s in labels)
