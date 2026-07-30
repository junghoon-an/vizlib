"""Evergreen & Emery checklist behaviours baked into the default look."""

import inspect

import matplotlib.colors as mcolors
import pandas as pd

from plot_helpers import _assert_axes

from vizlib import plots


def test_title_is_left_justified_and_subtitle_present(df):
    ax = plots.bar(df, "city", title="SF leads by volume", subtitle="n = 6")
    assert ax.get_title(loc="left") == "SF leads by volume"
    assert ax.get_title(loc="center") == ""  # placed left, not centered
    assert any("n = 6" in t.get_text() for t in ax.texts)  # subtitle rendered


def test_bar_labels_each_bar_and_hides_value_axis(df):
    ax = plots.bar(df, "city")
    assert len(ax.texts) == len(ax.patches)  # one direct label per bar
    assert not ax.xaxis.get_visible()  # redundant value axis hidden


def test_missing_bar_labels_each_bar(df):
    ax = plots.missing_bar(df)
    assert len(ax.texts) == len(ax.patches)
    assert not ax.xaxis.get_visible()


def test_value_labels_false_keeps_axis_and_no_labels(df):
    ax = plots.bar(df["city"], labels=plots.ValueLabels(show=False))
    assert ax.xaxis.get_visible()
    assert len(ax.texts) == 0


def test_highlight_uses_accent_and_muted(df):
    ax = plots.bar(df["city"], highlight="SF")
    accent = tuple(round(c, 4) for c in mcolors.to_rgba(plots._THEME["accent"]))
    muted = tuple(round(c, 4) for c in mcolors.to_rgba(plots._THEME["muted"]))
    faces = [tuple(round(c, 4) for c in p.get_facecolor()) for p in ax.patches]
    assert faces.count(accent) == 1  # only SF highlighted
    assert muted in faces  # the rest de-emphasized


def test_bar_has_no_tick_marks(df):
    ax = plots.bar(df["city"], labels=plots.ValueLabels(show=False))
    ticklines = ax.get_xticklines() + ax.get_yticklines()
    assert ticklines and all(t.get_markersize() == 0 for t in ticklines)


def test_line_keeps_tick_marks(df):
    ax = plots.line(df, "age", "height")
    ticklines = ax.get_xticklines() + ax.get_yticklines()
    assert any(t.get_markersize() > 0 for t in ticklines)


def test_box_by_orders_unordered_groups_by_median():
    d = pd.DataFrame(
        {"val": [1, 2, 10, 11, 5, 6], "grp": ["a", "a", "b", "b", "c", "c"]}
    )
    ax = plots.box(d, "val", by="grp")
    order = [t.get_text() for t in ax.get_xticklabels()]
    assert order == ["a", "c", "b"]  # medians 1.5 < 5.5 < 10.5


def test_new_params_are_keyword_only():
    sig = inspect.signature(plots.bar)
    for name in ("highlight", "labels", "as_percent", "max_label_chars",
                 "title", "subtitle", "source"):
        assert sig.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    # existing positional call (data, column) still works unchanged
    _assert_axes(plots.bar(pd.Series(["a", "a", "b"])))
