"""Layout-engine choice for hbar charts, ``ax=`` honouring and label ellipsis."""

import numpy as np
import pandas as pd

from vizlib import plots


def test_ax_supplied_layout_not_hijacked(df):
    """Passing ``ax=`` must leave the caller's figure layout untouched."""
    import matplotlib.pyplot as plt

    plots.set_theme(style_preset="infographic")
    fig, ax = plt.subplots()
    engine_before = fig.get_layout_engine()
    out = plots.missing_bar(df, ax=ax)
    assert out is ax
    assert fig.get_layout_engine() is engine_before  # not overridden
    assert not getattr(fig, "_vizlib_owned", False)


def test_non_hbar_owned_figure_uses_constrained_layout(df):
    from matplotlib.layout_engine import ConstrainedLayoutEngine

    ax = plots.hist(df["age"])  # non-hbar charts keep auto layout
    assert isinstance(ax.figure.get_layout_engine(), ConstrainedLayoutEngine)


def test_hbar_owned_figure_uses_manual_margins(df):
    from matplotlib.layout_engine import ConstrainedLayoutEngine

    # horizontal bars opt out of constrained layout and reserve margins by hand
    ax = plots.missing_bar(df)
    assert getattr(ax.figure, "_vizlib_manual_margins", False)
    assert not isinstance(ax.figure.get_layout_engine(), ConstrainedLayoutEngine)


def test_max_label_chars_ellipsizes():
    dfl = pd.DataFrame({
        "a_really_long_descriptive_column_name": np.r_[np.nan, np.ones(9)],
        "short": np.r_[np.nan, np.nan, np.ones(8)],
    })
    ax = plots.missing_bar(dfl, max_label_chars=12)
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert any(s.endswith("…") for s in labels)
    assert all(len(s) <= 12 for s in labels)
    assert "short" in labels  # short names left intact
