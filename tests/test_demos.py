"""End-to-end demo per dataset: the documented plots all render to an Axes."""

import pandas as pd
from matplotlib.axes import Axes

from dataset_helpers import _close_figures, _path  # noqa: F401

import vizlib
from vizlib import plots


def test_er_demo():
    er = vizlib.load(_path("er_daily_visits"))
    assert isinstance(plots.scatter(er, "admissions", "avg_wait_min"), Axes)
    assert isinstance(plots.bar(er, "department"), Axes)
    assert isinstance(plots.box(er, "admissions", by="department"), Axes)


def test_intake_demo():
    intake = vizlib.load(_path("patient_intake"))
    assert isinstance(vizlib.summarize(intake), pd.DataFrame)
    assert isinstance(vizlib.missing_values(intake), pd.DataFrame)
    assert isinstance(plots.bar(intake, "city", top=8), Axes)
    intake["stage"] = pd.Categorical(intake["stage"], ["I", "II", "III", "IV"], ordered=True)
    assert isinstance(plots.box(intake, "treatment_cost_usd", by="stage"), Axes)
    assert isinstance(plots.hist(intake["age"]), Axes)


def test_vitals_demo():
    vitals = vizlib.load(_path("patient_vitals"))
    assert isinstance(
        plots.scatter(vitals, "bmi", "glucose", hue="risk_group", reg=True), Axes
    )
    assert isinstance(plots.hist(vitals["glucose"]), Axes)
    assert isinstance(plots.box(vitals, "glucose", by="risk_group"), Axes)


def test_claims_demo():
    claims = vizlib.load(_path("hospital_claims"))
    assert isinstance(plots.hist(claims["total_charges_usd"]), Axes)
    assert isinstance(plots.scatter(claims, "total_charges_usd", "reimbursement_usd"), Axes)
    assert isinstance(plots.bar(claims, "diagnosis_category"), Axes)


def test_monitoring_demo():
    mon = vizlib.load(_path("patient_monitoring"))
    assert isinstance(plots.box(mon, "heart_rate_bpm", by="patient_id"), Axes)
    assert isinstance(plots.hist(mon["spo2_pct"]), Axes)
    assert isinstance(plots.scatter(mon, "heart_rate_bpm", "spo2_pct"), Axes)
