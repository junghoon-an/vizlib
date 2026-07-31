"""Contract tests for the categorical and distribution charts.

Every plot returns an Axes and never mutates its input. Relational, matrix and
cross-cutting contract tests live in ``test_charts.py``.
"""

import pandas as pd
import pytest

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
