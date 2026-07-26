"""Tests for the demo datasets and end-to-end plotting over them."""

import matplotlib

matplotlib.use("Agg")

import os  # noqa: E402

import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

import vizlib  # noqa: E402
from vizlib import plots  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "datasets")
REPO = os.path.join(os.path.dirname(__file__), "..")
FILES = [
    "er_daily_visits",
    "patient_intake",
    "patient_vitals",
    "hospital_claims",
    "patient_monitoring",
]


@pytest.fixture(autouse=True)
def _close_figures():
    import matplotlib.pyplot as plt

    yield
    plt.close("all")


def _path(name):
    return os.path.join(DATA, f"{name}.csv")


# --- datasets exist & load --------------------------------------------------


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


# --- end-to-end demos (one per dataset) ------------------------------------


def test_er_demo():
    er = vizlib.load(_path("er_daily_visits"))
    assert isinstance(plots.line(er, "date", "admissions", hue="department"), Axes)
    assert isinstance(plots.scatter(er, "admissions", "avg_wait_min"), Axes)
    assert isinstance(
        plots.correlation_heatmap(er[["admissions", "avg_wait_min", "staff_on_duty"]]),
        Axes,
    )
    assert isinstance(plots.bar(er, "department"), Axes)
    assert isinstance(plots.box(er, "admissions", by="department"), Axes)


def test_intake_demo():
    intake = vizlib.load(_path("patient_intake"))
    assert isinstance(vizlib.summarize(intake), pd.DataFrame)
    assert isinstance(vizlib.missing_values(intake), pd.DataFrame)
    assert isinstance(plots.missing_bar(intake), Axes)
    assert isinstance(plots.missing_matrix(intake), Axes)
    assert isinstance(plots.bar(intake, "city", top=8), Axes)
    intake["stage"] = pd.Categorical(intake["stage"], ["I", "II", "III", "IV"], ordered=True)
    assert isinstance(plots.box(intake, "treatment_cost_usd", by="stage"), Axes)
    assert isinstance(plots.hist(intake["age"]), Axes)


def test_vitals_demo():
    vitals = vizlib.load(_path("patient_vitals"))
    grid = plots.pairplot(vitals, hue="risk_group", columns=["bmi", "glucose"])
    assert grid.figure is not None
    assert isinstance(plots.correlation_heatmap(vitals), Axes)
    assert isinstance(
        plots.scatter(vitals, "bmi", "glucose", hue="risk_group", reg=True), Axes
    )
    assert isinstance(plots.distribution(vitals["glucose"]), Axes)
    assert isinstance(plots.box(vitals, "glucose", by="risk_group"), Axes)


def test_claims_demo():
    claims = vizlib.load(_path("hospital_claims"))
    assert isinstance(plots.hist(claims["total_charges_usd"]), Axes)
    assert isinstance(plots.scatter(claims, "total_charges_usd", "reimbursement_usd"), Axes)
    assert isinstance(plots.bar(claims, "diagnosis_category"), Axes)
    means = claims.groupby("admission_year", as_index=False)["total_charges_usd"].mean()
    assert isinstance(plots.line(means, "admission_year", "total_charges_usd"), Axes)
    assert isinstance(plots.missing_bar(claims), Axes)


def test_monitoring_demo():
    mon = vizlib.load(_path("patient_monitoring"))
    assert isinstance(plots.line(mon, "timestamp", "heart_rate_bpm", hue="patient_id"), Axes)
    assert isinstance(plots.box(mon, "heart_rate_bpm", by="patient_id"), Axes)
    assert isinstance(plots.distribution(mon["spo2_pct"]), Axes)
    assert isinstance(plots.missing_matrix(mon), Axes)
    assert isinstance(plots.scatter(mon, "heart_rate_bpm", "spo2_pct"), Axes)


# --- robustness on arbitrary / messy input ---------------------------------


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


# --- non-mutation -----------------------------------------------------------


def test_load_and_plots_do_not_mutate_dataset():
    er = vizlib.load(_path("er_daily_visits"))
    before = er.copy()
    plots.line(er, "date", "admissions", hue="department")
    plots.scatter(er, "admissions", "avg_wait_min")
    plots.bar(er, "department")
    plots.box(er, "admissions", by="department")
    plots.correlation_heatmap(er[["admissions", "staff_on_duty"]])
    pd.testing.assert_frame_equal(er, before)
