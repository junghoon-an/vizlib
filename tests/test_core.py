"""Tests for the pandas-only core API. These must pass with no plotting libs."""

import pandas as pd
import pytest

import vizlib


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "city": ["SF", "SF", "LA", "NYC", None],
            "age": [21, 34, 29, 41, 25],
        }
    )


def test_summarize_shape_and_columns(df):
    out = vizlib.summarize(df)
    assert list(out.columns) == ["dtype", "non_null", "nulls", "null_pct", "unique"]
    assert out.loc["city", "nulls"] == 1
    assert out.loc["age", "non_null"] == 5


def test_missing_values_only_missing(df):
    out = vizlib.missing_values(df)
    assert list(out.index) == ["city"]
    assert out.loc["city", "nulls"] == 1
    # only_missing=False lists every column
    assert set(vizlib.missing_values(df, only_missing=False).index) == {"city", "age"}


def test_numeric_summary(df):
    out = vizlib.numeric_summary(df)
    assert "age" in out.index
    assert out.loc["age", "count"] == 5


def test_numeric_summary_raises_without_numeric():
    with pytest.raises(ValueError):
        vizlib.numeric_summary(pd.DataFrame({"a": ["x", "y"]}))


def test_ascii_charts_return_strings(df):
    assert isinstance(vizlib.value_counts_bar(df["city"]), str)
    assert isinstance(vizlib.histogram(df["age"], bins=4), str)


def test_histogram_raises_without_numeric():
    with pytest.raises(ValueError):
        vizlib.histogram(pd.Series(["a", "b"]))


def test_type_guards():
    with pytest.raises(TypeError):
        vizlib.summarize([1, 2, 3])
    with pytest.raises(TypeError):
        vizlib.histogram([1, 2, 3])


def test_core_does_not_mutate(df):
    before = df.copy()
    vizlib.summarize(df)
    vizlib.missing_values(df)
    vizlib.value_counts_bar(df["city"])
    pd.testing.assert_frame_equal(df, before)
