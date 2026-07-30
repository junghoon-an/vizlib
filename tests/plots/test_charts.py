"""Contract tests for the relational and matrix charts, plus cross-cutting ones.

Every plot returns an Axes/grid, honours ``ax=`` and never mutates its input.
"""

import pandas as pd
import pytest

from plot_helpers import _assert_axes

from vizlib import plots


def test_scatter_returns_axes(df):
    _assert_axes(plots.scatter(df, "age", "height"))
    _assert_axes(plots.scatter(df, "age", "height", hue="group", reg=True))


def test_scatter_missing_column_raises(df):
    with pytest.raises(KeyError):
        plots.scatter(df, "age", "nope")


def test_scatter_reg_draws_line_without_confidence_band(df):
    """reg=True adds a trend line but no shaded confidence band (ci=None).

    The regplot band renders as a filled ``PolyCollection``; the scatter
    points are ``PathCollection``s and the trend itself is a ``Line2D``.
    """
    from matplotlib.collections import PolyCollection

    before = df.copy()
    ax = plots.scatter(df, "age", "height", reg=True)
    assert not [c for c in ax.collections if isinstance(c, PolyCollection)]
    assert ax.lines  # the regression trend line is still drawn
    pd.testing.assert_frame_equal(df, before)


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
    from matplotlib.figure import Figure

    grid = plots.pairplot(df)
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
