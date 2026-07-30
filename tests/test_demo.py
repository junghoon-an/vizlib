"""Regression guard: the exact README ``## Demo`` calls still run unchanged.

Mirrors the two plotting calls in the README demo verbatim so a signature
change that would break the demo fails here instead.
"""

from matplotlib.axes import Axes

from dataset_helpers import _close_figures, _path  # noqa: F401

import vizlib
from vizlib import plots


def test_readme_demo_calls_run_and_return_axes():
    df = vizlib.load(_path("patient_vitals"))
    subset = df[df["risk_group"].isin(["Healthy", "Diabetic"])]
    assert isinstance(
        plots.scatter(subset, "bmi", "glucose", hue="risk_group", reg=True), Axes
    )

    df = vizlib.load(_path("patient_intake"))
    assert isinstance(plots.box(df, "treatment_cost_usd", by="stage"), Axes)
