"""Shared fixtures for the plotting-layer tests.

The non-interactive Agg backend is selected here — before any test module
imports the plots package — so nothing tries to open a window. This conftest
is imported by pytest ahead of the test modules in this directory.
"""

import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from vizlib import plots  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    """Close figures and reset the theme so preset tests don't leak state."""
    import matplotlib.pyplot as plt

    yield
    plt.close("all")
    plots.set_theme(style_preset="default")


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "city": ["SF", "SF", "LA", "NYC", "LA", "SF", None],
            "age": [21, 34, 29, 41, 25, 38, 30],
            "height": [170, 165, 180, 175, 160, 185, 172],
            "group": ["a", "b", "a", "b", "a", "b", "a"],
        }
    )
