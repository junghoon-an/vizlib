"""Shared paths and fixtures for the demo-dataset tests.

Not a test module itself (no ``test_`` prefix). The sibling ``test_datasets``/
``test_demos``/``test_robustness`` modules import ``_path``/``FILES`` and the
``_close_figures`` autouse fixture from here. Selecting the Agg backend on
import keeps the plotting calls headless.
"""

import os

import matplotlib
import pytest

matplotlib.use("Agg")

DATA = os.path.join(os.path.dirname(__file__), "..", "datasets")
REPO = os.path.join(os.path.dirname(__file__), "..")
FILES = [
    "er_daily_visits",
    "patient_intake",
    "patient_vitals",
    "hospital_claims",
    "patient_monitoring",
]


def _path(name):
    return os.path.join(DATA, f"{name}.csv")


@pytest.fixture(autouse=True)
def _close_figures():
    import matplotlib.pyplot as plt

    yield
    plt.close("all")
