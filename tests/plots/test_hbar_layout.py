"""Auto-reserving layout: enlarged fonts must not spill into the plot."""

import numpy as np
import pandas as pd
import pytest

from plot_helpers import _bboxes_overlap, _long_name_df, _missing_df

from vizlib import plots


@pytest.mark.parametrize("preset", ["default", "infographic"])
def test_ticklabels_stay_in_left_margin(preset):
    """Column names must not spill rightward into the plotting area."""
    plots.set_theme(style_preset=preset)
    ax = plots.missing_bar(_long_name_df())
    ax.figure.canvas.draw()
    left = ax.get_window_extent().x0
    for label in ax.get_yticklabels():
        assert label.get_window_extent().x1 <= left + 1.0


@pytest.mark.parametrize("preset", ["default", "infographic"])
def test_value_labels_do_not_overlap_ticklabels(preset):
    plots.set_theme(style_preset=preset)
    ax = plots.missing_bar(_long_name_df())
    ax.figure.canvas.draw()
    ticks = [t.get_window_extent() for t in ax.get_yticklabels()]
    values = [t.get_window_extent() for t in ax.texts]
    for vb in values:
        assert not any(_bboxes_overlap(vb, tb) for tb in ticks)


@pytest.mark.parametrize("preset", ["default", "infographic"])
def test_value_labels_do_not_overlap_each_other(preset):
    plots.set_theme(style_preset=preset)
    ax = plots.missing_bar(_missing_df(n_cols=20))
    ax.figure.canvas.draw()
    boxes = sorted((t.get_window_extent() for t in ax.texts), key=lambda b: b.y0)
    for lower, upper in zip(boxes, boxes[1:]):
        assert lower.y1 <= upper.y0 + 0.5


@pytest.mark.parametrize("preset", ["default", "infographic"])
def test_missing_bar_headroom_no_clipping(preset):
    plots.set_theme(style_preset=preset)
    ax = plots.missing_bar(_long_name_df())
    assert ax.get_xlim()[1] > max(p.get_width() for p in ax.patches)


def test_layout_is_style_switch_stable():
    """Both styles lay out with no overlap — resolved per call, not at import."""
    def _no_overlap(ax):
        ax.figure.canvas.draw()
        left = ax.get_window_extent().x0
        assert all(t.get_window_extent().x1 <= left + 1.0
                   for t in ax.get_yticklabels())
        ticks = [t.get_window_extent() for t in ax.get_yticklabels()]
        for vb in (t.get_window_extent() for t in ax.texts):
            assert not any(_bboxes_overlap(vb, tb) for tb in ticks)

    plots.set_theme(style_preset="infographic")
    _no_overlap(plots.missing_bar(_long_name_df()))
    plots.set_theme(style_preset="default")
    _no_overlap(plots.missing_bar(_long_name_df()))


def test_ax_supplied_layout_not_hijacked(df):
    """Passing ``ax=`` must leave the caller's figure layout untouched."""
    import matplotlib.pyplot as plt

    plots.set_theme(style_preset="infographic")
    fig, ax = plt.subplots()
    engine_before = fig.get_layout_engine()
    out = plots.missing_bar(df, ax=ax)
    assert out is ax
    assert fig.get_layout_engine() is engine_before  # not overridden
    assert not getattr(fig, "_vizlib_owned", False)


def test_non_hbar_owned_figure_uses_constrained_layout(df):
    from matplotlib.layout_engine import ConstrainedLayoutEngine

    ax = plots.hist(df["age"])  # non-hbar charts keep auto layout
    assert isinstance(ax.figure.get_layout_engine(), ConstrainedLayoutEngine)


def test_hbar_owned_figure_uses_manual_margins(df):
    from matplotlib.layout_engine import ConstrainedLayoutEngine

    # horizontal bars opt out of constrained layout and reserve margins by hand
    ax = plots.missing_bar(df)
    assert getattr(ax.figure, "_vizlib_manual_margins", False)
    assert not isinstance(ax.figure.get_layout_engine(), ConstrainedLayoutEngine)


def test_max_label_chars_ellipsizes():
    dfl = pd.DataFrame({
        "a_really_long_descriptive_column_name": np.r_[np.nan, np.ones(9)],
        "short": np.r_[np.nan, np.nan, np.ones(8)],
    })
    ax = plots.missing_bar(dfl, max_label_chars=12)
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert any(s.endswith("…") for s in labels)
    assert all(len(s) <= 12 for s in labels)
    assert "short" in labels  # short names left intact
