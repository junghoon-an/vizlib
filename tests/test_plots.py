"""Tests for the optional plotting layer.

The non-interactive Agg backend is selected *before* importing the plots
module so nothing tries to open a window.
"""

import matplotlib

matplotlib.use("Agg")

import inspect  # noqa: E402

import matplotlib.colors as mcolors  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

from vizlib import plots  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    """Close figures and reset the theme so preset tests don't leak state."""
    import matplotlib.pyplot as plt

    yield
    plt.close("all")
    plots.set_theme(style_preset="default")


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "city": ["SF", "SF", "LA", "NYC", "LA", "SF", None],
            "age": [21, 34, 29, 41, 25, 38, 30],
            "height": [170, 165, 180, 175, 160, 185, 172],
            "group": ["a", "b", "a", "b", "a", "b", "a"],
        }
    )


def _assert_axes(obj):
    assert isinstance(obj, Axes)


def test_bar_returns_axes_and_no_mutation(df):
    before = df.copy()
    _assert_axes(plots.bar(df, "city"))
    _assert_axes(plots.bar(df["city"]))
    pd.testing.assert_frame_equal(df, before)


def test_bar_top_and_other_bucket():
    s = pd.Series(list("aaabbbcccdddeee") + ["f", "g", "h"])
    ax = plots.bar(s, top=3)
    labels = {t.get_text() for t in ax.get_yticklabels()}
    assert "Other" in labels
    # top=3 busiest categories plus the Other bucket => 4 bars
    assert len(ax.patches) == 4


def test_bar_sort_false_keeps_all(df):
    ax = plots.bar(df["city"], sort=False, top=100)
    assert len(ax.patches) == df["city"].nunique()


def test_bar_requires_column_for_dataframe(df):
    with pytest.raises(ValueError):
        plots.bar(df)


def test_bar_empty_raises():
    with pytest.raises(ValueError):
        plots.bar(pd.Series([], dtype=object))


def test_hist_returns_axes(df):
    _assert_axes(plots.hist(df["age"], bins=5, kde=True))


def test_hist_raises_on_non_numeric():
    with pytest.raises(ValueError):
        plots.hist(pd.Series(["x", "y"]))


def test_distribution_returns_axes(df):
    _assert_axes(plots.distribution(df["age"]))


def test_box_variants(df):
    _assert_axes(plots.box(df))  # all numeric columns
    _assert_axes(plots.box(df, "age"))  # single column
    _assert_axes(plots.box(df, "age", by="group"))  # grouped


def test_box_no_numeric_raises():
    with pytest.raises(ValueError):
        plots.box(pd.DataFrame({"a": ["x", "y"]}))


def test_scatter_returns_axes(df):
    _assert_axes(plots.scatter(df, "age", "height"))
    _assert_axes(plots.scatter(df, "age", "height", hue="group", reg=True))


def test_scatter_missing_column_raises(df):
    with pytest.raises(KeyError):
        plots.scatter(df, "age", "nope")


def test_line_returns_axes(df):
    _assert_axes(plots.line(df, "age", "height"))
    _assert_axes(plots.line(df, "age", "height", hue="group"))


def test_correlation_heatmap(df):
    _assert_axes(plots.correlation_heatmap(df))


def test_correlation_heatmap_needs_two_numeric():
    with pytest.raises(ValueError):
        plots.correlation_heatmap(pd.DataFrame({"a": [1, 2, 3]}))


def test_missing_plots(df):
    _assert_axes(plots.missing_bar(df))
    _assert_axes(plots.missing_matrix(df))


def test_pairplot_returns_grid(df):
    grid = plots.pairplot(df)
    from matplotlib.figure import Figure

    assert isinstance(grid.figure, Figure)


def test_ax_is_honored(df):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    out = plots.hist(df["age"], ax=ax)
    assert out is ax


# --- checklist behaviors ---------------------------------------------------


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
    ax = plots.bar(df["city"], value_labels=False)
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
    ax = plots.bar(df["city"], value_labels=False)
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
    for name in ("highlight", "value_labels", "precision", "title", "subtitle", "source"):
        assert sig.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    # existing positional call (data, column) still works unchanged
    _assert_axes(plots.bar(pd.Series(["a", "a", "b"])))


# --- infographic preset ----------------------------------------------------


def test_preset_switch_changes_and_restores(df):
    plots.set_theme(style_preset="infographic")
    assert plots._THEME["palette"] == plots._VIVID_PALETTE
    assert plots._THEME["background"] == "#FFFFFF"
    assert plots._THEME["bold_labels"] is True
    assert isinstance(plots.bar(df, "city"), Axes)  # still returns an Axes
    plots.set_theme(style_preset="default")
    assert plots._THEME["palette"] == "colorblind"
    assert plots._THEME["background"] is None
    assert isinstance(plots.hist(df["age"]), Axes)


def test_unknown_preset_raises():
    with pytest.raises(ValueError):
        plots.set_theme(style_preset="nope")


def test_infographic_bold_labels_with_contrast(df):
    plots.set_theme(style_preset="infographic")
    ax = plots.bar(df, "city")
    assert len(ax.texts) == len(ax.patches)  # one label per bar
    assert all(t.get_fontweight() == "bold" for t in ax.texts)
    allowed = {"white", (1.0, 1.0, 1.0, 1.0)}
    for t in ax.texts:  # contrasting: white on dark, else the dark text color
        col = t.get_color()
        assert col in allowed or col == plots._THEME["text_color"]


def test_default_labels_not_bold(df):
    ax = plots.bar(df, "city")  # default preset
    assert ax.texts and all(t.get_fontweight() != "bold" for t in ax.texts)


def test_bar_as_percent(df):
    ax = plots.bar(df, "city", as_percent=True)
    assert any("%" in t.get_text() for t in ax.texts)


def test_line_area_adds_gradient_image(df):
    ax = plots.line(df, "age", "height", area=True)
    assert len(ax.images) >= 1  # gradient fill image


def test_stacked_area_uses_traffic_light_and_swatch_legend():
    import matplotlib.colors as mcolors

    n = 20
    sdf = pd.DataFrame({
        "t": list(range(n)) * 3,
        "v": [1.0] * (n * 3),
        "sev": ["low"] * n + ["medium"] * n + ["high"] * n,
    })
    ax = plots.line(sdf, "t", "v", hue="sev", area=True, stack=True)
    assert ax.get_legend() is not None  # swatch legend
    faces = [tuple(round(c, 3) for c in coll.get_facecolor()[0])
             for coll in ax.collections if len(coll.get_facecolor())]
    for hexc in plots._TRAFFIC_LIGHT:
        want = tuple(round(c, 3) for c in mcolors.to_rgba(hexc))
        assert any(all(abs(f[i] - want[i]) < 0.02 for i in range(3)) for f in faces)


def test_callouts_add_annotations(df):
    import matplotlib.text as mtext

    anns = [(df["age"].iloc[3], "event A"), (df["age"].iloc[5], "event B")]
    ax = plots.line(df, "age", "height", annotations=anns)
    drawn = [c for c in ax.get_children() if isinstance(c, mtext.Annotation)]
    assert len(drawn) == 2
    assert all(a.arrow_patch is not None for a in drawn)


def test_donut_renders_center_and_wedges(df):
    ax = plots.donut(df["city"], center_text="Cities")
    assert isinstance(ax, Axes)
    assert len(ax.patches) >= 2  # wedges
    assert any(t.get_text() == "Cities" for t in ax.texts)  # center caption


def test_infographic_ax_honored_and_no_mutation(df):
    import matplotlib.pyplot as plt

    plots.set_theme(style_preset="infographic")
    before = df.copy()
    fig, ax = plt.subplots()
    out = plots.bar(df, "city", ax=ax)
    assert out is ax
    plots.line(df, "age", "height", area=True)
    plots.donut(df["group"])
    pd.testing.assert_frame_equal(df, before)


# --- horizontal-bar label placement (regression: labels over y-axis) --------


def _missing_df(n_cols=6, n_rows=60):
    rng = np.random.default_rng(0)
    cols = {}
    for i in range(n_cols):
        p_missing = min(0.05 + i * 0.03, 0.6)  # varied but always valid
        cols[f"col_{i:02d}"] = rng.choice([1.0, np.nan], n_rows,
                                          p=[1 - p_missing, p_missing])
    return pd.DataFrame(cols)


def _label_gaps(ax):
    """Pixel gap between each value label and its bar's tip (drawn order)."""
    ax.figure.canvas.draw()
    gaps = []
    for text, patch in zip(ax.texts, ax.patches):
        tip_px = ax.transData.transform((patch.get_width(), 0))[0]
        gaps.append(text.get_window_extent().x0 - tip_px)
    return gaps


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
    mdf = _missing_df()
    before = mdf.copy()
    plots.set_theme(style_preset="infographic")
    assert isinstance(plots.missing_bar(mdf), Axes)
    pd.testing.assert_frame_equal(mdf, before)


# --- auto-reserving layout (regression: enlarged fonts spill into the plot) --

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


def _bboxes_overlap(a, b, tol=0.5):
    return not (a.x1 <= b.x0 + tol or b.x1 <= a.x0 + tol
                or a.y1 <= b.y0 + tol or b.y1 <= a.y0 + tol)


@pytest.mark.parametrize("preset", ["default", "infographic"])
def test_ticklabels_stay_in_left_margin(preset):
    """Column names must not spill rightward into the plotting area.

    Proves the left margin is reserved against the *active* style's fonts:
    the layout self-corrects for the larger, bolder infographic ticks.
    """
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
    """Set infographic, plot, measure; switch to default, plot, measure.

    Both must lay out with no overlap — because the layout is resolved per
    call against the fonts current at that call, not fixed at import time.
    """
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


def test_owned_figure_uses_constrained_layout(df):
    from matplotlib.layout_engine import ConstrainedLayoutEngine

    ax = plots.missing_bar(df)
    assert isinstance(ax.figure.get_layout_engine(), ConstrainedLayoutEngine)


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


def test_plots_never_mutate_input(df):
    before = df.copy()
    plots.bar(df, "city", highlight="SF", title="t", subtitle="s", source="src")
    plots.hist(df["age"], title="t")
    plots.distribution(df["age"])
    plots.box(df, "age", by="group")
    plots.scatter(df, "age", "height", hue="group", reg=True)
    plots.line(df, "age", "height", hue="group")
    plots.correlation_heatmap(df, subtitle="s")
    plots.missing_bar(df, highlight="city")
    plots.missing_matrix(df)
    plots.pairplot(df, columns=["age", "height"])
    pd.testing.assert_frame_equal(df, before)
