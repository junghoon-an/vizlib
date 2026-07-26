"""Tests for vizlib.load — cleaning arbitrary CSVs (pandas-only)."""

import pandas as pd
import pytest

import vizlib


def _write(tmp_path, name, text, encoding="utf-8"):
    p = tmp_path / name
    p.write_bytes(text.encode(encoding))
    return str(p)


def test_na_tokens_become_nan(tmp_path):
    csv = "a,b\n1,x\nNA,y\nN/A,z\n?,w\n,v\nnull,u\nunknown,t\n"
    df = vizlib.load(_write(tmp_path, "na.csv", csv))
    # only "1" is a real value in column a -> the other six tokens are NaN
    assert df["a"].isna().sum() == 6
    assert pd.api.types.is_numeric_dtype(df["a"])


def test_na_tokens_are_case_insensitive(tmp_path):
    csv = "a\nNull\nNONE\nUnknown\n5\n"
    df = vizlib.load(_write(tmp_path, "na2.csv", csv))
    assert df["a"].isna().sum() == 3


def test_currency_thousands_percent_stripped(tmp_path):
    csv = 'price,rate,id\n"$1,234.50",12%,A001\n"$2,000.00",8%,A002\n'
    df = vizlib.load(_write(tmp_path, "money.csv", csv))
    assert pd.api.types.is_numeric_dtype(df["price"])
    assert df["price"].tolist() == [1234.5, 2000.0]
    assert pd.api.types.is_numeric_dtype(df["rate"])
    # an ID-like text column is left alone
    assert not pd.api.types.is_numeric_dtype(df["id"])


def test_dates_parsed(tmp_path):
    csv = "d,v\n2024-01-01,1\n2024-01-02,2\n2024-01-03,3\n"
    df = vizlib.load(_write(tmp_path, "d.csv", csv))
    assert pd.api.types.is_datetime64_any_dtype(df["d"])


def test_encoding_fallback_to_latin1(tmp_path):
    # 0xE9 ("é" in latin-1) is invalid UTF-8, forcing the fallback path
    csv = "name,val\nJos\xe9,1\nRen\xe9,2\n"
    df = vizlib.load(_write(tmp_path, "lat.csv", csv, encoding="latin-1"))
    assert len(df) == 2 and "name" in df.columns


def test_numeric_false_leaves_strings(tmp_path):
    csv = 'price\n"$1,000"\n"$2,000"\n'
    df = vizlib.load(_write(tmp_path, "m.csv", csv), numeric=False)
    assert not pd.api.types.is_numeric_dtype(df["price"])


def test_numeric_list_coerces_only_listed(tmp_path):
    csv = 'a,b\n"$1,000","$2,000"\n"$3,000","$4,000"\n'
    df = vizlib.load(_write(tmp_path, "m2.csv", csv), numeric=["a"])
    assert pd.api.types.is_numeric_dtype(df["a"])
    assert not pd.api.types.is_numeric_dtype(df["b"])


def test_parse_dates_false(tmp_path):
    csv = "d,v\n2024-01-01,1\n2024-01-02,2\n"
    df = vizlib.load(_write(tmp_path, "d2.csv", csv), parse_dates=False)
    assert not pd.api.types.is_datetime64_any_dtype(df["d"])


def test_extra_na_values_merge(tmp_path):
    csv = "a\nMISSING\n5\n7\n"
    df = vizlib.load(_write(tmp_path, "x.csv", csv), na_values=["MISSING"])
    assert df["a"].isna().sum() == 1


def test_sample_is_reproducible(tmp_path):
    csv = "x\n" + "\n".join(str(i) for i in range(100)) + "\n"
    path = _write(tmp_path, "big.csv", csv)
    a = vizlib.load(path, sample=10, random_state=3)
    b = vizlib.load(path, sample=10, random_state=3)
    assert len(a) == 10
    pd.testing.assert_frame_equal(a, b)
