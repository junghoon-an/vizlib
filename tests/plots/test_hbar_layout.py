"""Auto-reserving layout: enlarged fonts must not spill into the plot.

``bar`` drives the shared horizontal-bar renderer. Layout-engine selection,
``ax=`` honouring and ellipsis live in ``test_hbar_engine.py``.
"""

import pytest

from plot_helpers import _bboxes_overlap, _long_name_series, _many_cat_series

from vizlib import plots


@pytest.mark.parametrize("preset", ["default", "infographic"])
def test_ticklabels_stay_in_left_margin(preset):
    """Category names must not spill rightward into the plotting area."""
    plots.set_theme(style_preset=preset)
    ax = plots.bar(_long_name_series())
    ax.figure.canvas.draw()
    left = ax.get_window_extent().x0
    for label in ax.get_yticklabels():
        assert label.get_window_extent().x1 <= left + 1.0


@pytest.mark.parametrize("preset", ["default", "infographic"])
def test_value_labels_do_not_overlap_ticklabels(preset):
    plots.set_theme(style_preset=preset)
    ax = plots.bar(_long_name_series())
    ax.figure.canvas.draw()
    ticks = [t.get_window_extent() for t in ax.get_yticklabels()]
    values = [t.get_window_extent() for t in ax.texts]
    for vb in values:
        assert not any(_bboxes_overlap(vb, tb) for tb in ticks)


@pytest.mark.parametrize("preset", ["default", "infographic"])
def test_value_labels_do_not_overlap_each_other(preset):
    plots.set_theme(style_preset=preset)
    ax = plots.bar(_many_cat_series(n_cats=20), top=100)
    ax.figure.canvas.draw()
    boxes = sorted((t.get_window_extent() for t in ax.texts), key=lambda b: b.y0)
    for lower, upper in zip(boxes, boxes[1:]):
        assert lower.y1 <= upper.y0 + 0.5


@pytest.mark.parametrize("preset", ["default", "infographic"])
def test_bar_headroom_no_clipping(preset):
    plots.set_theme(style_preset=preset)
    ax = plots.bar(_long_name_series())
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
    _no_overlap(plots.bar(_long_name_series()))
    plots.set_theme(style_preset="default")
    _no_overlap(plots.bar(_long_name_series()))
