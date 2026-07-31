"""The demo datasets exist and load with the expected dtypes and NaNs."""

import os

import pandas as pd

from dataset_helpers import FILES, REPO, _close_figures, _path  # noqa: F401

import vizlib


def test_all_five_datasets_present():
    for name in FILES:
        assert os.path.exists(_path(name)), f"missing {name}.csv"


def test_no_generator_script_committed():
    for dirpath, dirs, files in os.walk(REPO):
        if ".git" in dirpath:
            continue
        for fn in files:
            assert "generate" not in fn.lower() or not fn.endswith(".py"), (
                f"unexpected generator script committed: {fn}"
            )


def test_datasets_load_with_expected_types_and_nans():
    er = vizlib.load(_path("er_daily_visits"))
    assert {"date", "department", "admissions", "billed_amount",
            "avg_wait_min"}.issubset(er.columns)
    assert pd.api.types.is_datetime64_any_dtype(er["date"])
    assert pd.api.types.is_numeric_dtype(er["billed_amount"])  # "$..," stripped
    assert er["avg_wait_min"].isna().any()  # "N/A" tokens -> NaN

    intake = vizlib.load(_path("patient_intake"))
    assert intake["age"].isna().any()
    assert pd.api.types.is_numeric_dtype(intake["treatment_cost_usd"])
    assert pd.api.types.is_datetime64_any_dtype(intake["diagnosis_date"])

    claims = vizlib.load(_path("hospital_claims"))
    assert pd.api.types.is_numeric_dtype(claims["total_charges_usd"])
    assert claims["length_of_stay_days"].isna().any()  # "unknown" -> NaN

    mon = vizlib.load(_path("patient_monitoring"))
    assert pd.api.types.is_datetime64_any_dtype(mon["timestamp"])
    assert mon["heart_rate_bpm"].isna().any()  # "NA" -> NaN
