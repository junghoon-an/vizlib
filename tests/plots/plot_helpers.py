"""Plain helper functions shared across the plotting-layer test modules.

Not a test module itself (no ``test_`` prefix), so pytest won't collect it;
the sibling test files import from it. The Agg backend is already selected by
this directory's ``conftest.py``, which pytest imports first.
"""

import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from vizlib import plots


def _assert_axes(obj):
    assert isinstance(obj, Axes)


def _missing_df(n_cols=6, n_rows=60):
    rng = np.random.default_rng(0)
    cols = {}
    for i in range(n_cols):
        p_missing = min(0.05 + i * 0.03, 0.6)  # varied but always valid
        cols[f"col_{i:02d}"] = rng.choice([1.0, np.nan], n_rows,
                                          p=[1 - p_missing, p_missing])
    return pd.DataFrame(cols)


def _long_name_df(n_cols=7, n_rows=80):
    """Long column names + short bars — the case that used to overflow."""
    rng = np.random.default_rng(1)
    cols = {}
    for i in range(n_cols):
        name = f"a_really_long_descriptive_column_name_number_{i:02d}"
        col = rng.choice([1.0, np.nan], n_rows,
                         p=[1 - min(0.03 + i * 0.02, 0.5), min(0.03 + i * 0.02, 0.5)])
        cols[name] = col
    return pd.DataFrame(cols)


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
