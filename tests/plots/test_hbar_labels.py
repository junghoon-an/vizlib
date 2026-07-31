"""Horizontal-bar value-label placement (regression: labels over the y-axis).

``bar`` drives the shared horizontal-bar renderer these tests exercise.
"""

import pandas as pd
import pytest
from matplotlib.axes import Axes

from plot_helpers import _label_gaps, _long_name_series, _many_cat_series

from vizlib import plots


@pytest.mark.parametrize("preset", ["infographic", "default"])
def test_bar_labels_edge_placed_and_clear_of_yaxis(preset):
    plots.set_theme(style_preset=preset)
    ax = plots.bar(_long_name_series())
    ax.figure.canvas.draw()
    axis0_px = ax.transData.transform((0, 0))[0]  # the value-axis (x=0) in pixels
    for text, patch in zip(ax.texts, ax.patches):
        # edge placement: label anchored at the bar tip, not its centre
        assert text.xy[0] == pytest.approx(patch.get_width())
        # and drawn to the right of the axis, never over the left column names
        assert text.get_window_extent().x0 >= axis0_px


@pytest.mark.parametrize("preset", ["infographic", "default"])
def test_bar_label_gap_is_uniform(preset):
    plots.set_theme(style_preset=preset)
    gaps = _label_gaps(plots.bar(_long_name_series()))
    assert max(gaps) - min(gaps) < 1.0  # same fixed offset for every label


def test_bar_has_right_headroom():
    plots.set_theme(style_preset="infographic")
    ax = plots.bar(_long_name_series())
    assert ax.get_xlim()[1] > max(p.get_width() for p in ax.patches)


def test_bar_labels_do_not_overlap_vertically():
    plots.set_theme(style_preset="infographic")
    ax = plots.bar(_many_cat_series(n_cats=20), top=100)
    ax.figure.canvas.draw()
    boxes = sorted((t.get_window_extent() for t in ax.texts), key=lambda b: b.y0)
    for lower, upper in zip(boxes, boxes[1:]):
        assert lower.y1 <= upper.y0 + 0.5  # adjacent labels don't intersect


def test_bar_labels_edge_placed_under_preset(df):
    plots.set_theme(style_preset="infographic")
    ax = plots.bar(df, "city")
    for text, patch in zip(ax.texts, ax.patches):
        assert text.xy[0] == pytest.approx(patch.get_width())  # at tip, not centre


def test_bar_under_preset_no_mutation():
    s = _long_name_series()
    before = s.copy()
    plots.set_theme(style_preset="infographic")
    assert isinstance(plots.bar(s), Axes)
    pd.testing.assert_series_equal(s, before)
