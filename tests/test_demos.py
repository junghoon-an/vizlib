"""End-to-end demo per dataset: the documented plots all render to an Axes."""

import pandas as pd
from matplotlib.axes import Axes

from dataset_helpers import _close_figures, _path  # noqa: F401

import vizlib
from vizlib import plots


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
