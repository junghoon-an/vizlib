"""Tests for the optional plotting layer.

The non-interactive Agg backend is selected *before* importing the plots
module so nothing tries to open a window.
"""

import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

from vizlib import plots  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    """Close every figure a test opens so the suite doesn't leak them."""
    import matplotlib.pyplot as plt

    yield
    plt.close("all")


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


def test_plots_never_mutate_input(df):
    before = df.copy()
    plots.bar(df, "city")
    plots.hist(df["age"])
    plots.box(df, "age", by="group")
    plots.scatter(df, "age", "height", hue="group", reg=True)
    plots.line(df, "age", "height")
    plots.correlation_heatmap(df)
    plots.missing_bar(df)
    plots.missing_matrix(df)
    pd.testing.assert_frame_equal(df, before)
