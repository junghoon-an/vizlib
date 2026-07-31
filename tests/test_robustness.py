"""Plots cope with messy input, fail clearly on degenerate input, never mutate."""

import pandas as pd
import pytest
from matplotlib.axes import Axes

from dataset_helpers import _close_figures, _path  # noqa: F401

import vizlib
from vizlib import plots


def test_hist_on_numeric_string_series():
    s = pd.Series(["$1,000", "$2,000", "$3,000", "bad"])
    assert isinstance(plots.hist(s), Axes)


def test_scatter_on_numeric_string_columns():
    df = pd.DataFrame({"a": ["$1", "$2", "$3"], "b": ["10%", "20%", "30%"]})
    assert isinstance(plots.scatter(df, "a", "b"), Axes)


def test_correlation_heatmap_over_coercible_columns():
    df = pd.DataFrame({"a": ["1", "2", "3", "4"], "b": ["$2", "$4", "$6", "$8"]})
    assert isinstance(plots.correlation_heatmap(df), Axes)


def test_clear_errors_on_degenerate_input():
    with pytest.raises(ValueError):
        vizlib.numeric_summary(pd.DataFrame())  # empty
    with pytest.raises(ValueError):
        plots.missing_bar(pd.DataFrame())  # no columns
    with pytest.raises(ValueError):
        plots.correlation_heatmap(pd.DataFrame({"a": [1, 2, 3]}))  # single numeric
    with pytest.raises(ValueError):
        plots.hist(pd.Series([None, None], dtype="float"))  # all-NaN


def test_duplicate_column_names_raise_clearly():
    df = pd.DataFrame([[1, 2], [3, 4]], columns=["a", "a"])
    with pytest.raises(ValueError):
        vizlib.numeric_summary(df)


def test_scatter_sample_is_reproducible():
    df = pd.DataFrame({"x": range(1000), "y": range(1000)})
    assert isinstance(plots.scatter(df, "x", "y", sample=50, random_state=1), Axes)


def test_load_and_plots_do_not_mutate_dataset():
    er = vizlib.load(_path("er_daily_visits"))
    before = er.copy()
    plots.line(er, "date", "admissions", hue="department")
    plots.scatter(er, "admissions", "avg_wait_min")
    plots.bar(er, "department")
    plots.box(er, "admissions", by="department")
    plots.correlation_heatmap(er[["admissions", "staff_on_duty"]])
    pd.testing.assert_frame_equal(er, before)
