"""Basic contract tests: every plot returns an Axes and never mutates input."""

import pandas as pd
import pytest
from matplotlib.axes import Axes

from plot_helpers import _assert_axes

from vizlib import plots


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
