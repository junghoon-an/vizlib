"""Horizontal-bar value-label placement (regression: labels over the y-axis)."""

import pytest
from matplotlib.axes import Axes

from plot_helpers import _label_gaps, _missing_df

from vizlib import plots


@pytest.mark.parametrize("preset", ["infographic", "default"])
def test_missing_bar_labels_edge_placed_and_clear_of_yaxis(preset):
    plots.set_theme(style_preset=preset)
    ax = plots.missing_bar(_missing_df())
    ax.figure.canvas.draw()
    axis0_px = ax.transData.transform((0, 0))[0]  # the value-axis (x=0) in pixels
    for text, patch in zip(ax.texts, ax.patches):
        # edge placement: label anchored at the bar tip, not its centre
        assert text.xy[0] == pytest.approx(patch.get_width())
        # and drawn to the right of the axis, never over the left column names
        assert text.get_window_extent().x0 >= axis0_px


@pytest.mark.parametrize("preset", ["infographic", "default"])
def test_missing_bar_label_gap_is_uniform(preset):
    plots.set_theme(style_preset=preset)
    gaps = _label_gaps(plots.missing_bar(_missing_df()))
    assert max(gaps) - min(gaps) < 1.0  # same fixed offset for every label


def test_missing_bar_has_right_headroom():
    plots.set_theme(style_preset="infographic")
    ax = plots.missing_bar(_missing_df())
    assert ax.get_xlim()[1] > max(p.get_width() for p in ax.patches)


def test_missing_bar_labels_do_not_overlap_vertically():
    plots.set_theme(style_preset="infographic")
    ax = plots.missing_bar(_missing_df(n_cols=20))
    ax.figure.canvas.draw()
    boxes = sorted((t.get_window_extent() for t in ax.texts), key=lambda b: b.y0)
    for lower, upper in zip(boxes, boxes[1:]):
        assert lower.y1 <= upper.y0 + 0.5  # adjacent labels don't intersect


def test_bar_horizontal_labels_edge_placed_under_preset(df):
    plots.set_theme(style_preset="infographic")
    ax = plots.bar(df, "city")
    for text, patch in zip(ax.texts, ax.patches):
        assert text.xy[0] == pytest.approx(patch.get_width())  # at tip, not centre


def test_missing_bar_under_preset_no_mutation():
    import pandas as pd

    mdf = _missing_df()
    before = mdf.copy()
    plots.set_theme(style_preset="infographic")
    assert isinstance(plots.missing_bar(mdf), Axes)
    pd.testing.assert_frame_equal(mdf, before)
