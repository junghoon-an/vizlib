"""Worst-case cross-preset margin reservation for horizontal bars.

The left margin and right value-label headroom are reserved for the largest
label footprint across *all* presets, so the axes position is identical no
matter which style is active — and never overlapped by the column names.
"""

import numpy as np
import pandas as pd
import pytest

from plot_helpers import _long_name_df, _widest_ytick_px_under_infographic

from vizlib import plots


def test_margin_reserved_for_largest_regime():
    """With the *smaller* default style active, the reserved left region
    already fits the *larger* infographic column names."""
    plots.set_theme(style_preset="default")
    ax = plots.missing_bar(_long_name_df())
    fig = ax.figure
    fig.canvas.draw()
    reserved_px = ax.get_position().x0 * fig.get_figwidth() * fig.dpi
    names = [t.get_text() for t in ax.get_yticklabels()]
    assert reserved_px >= _widest_ytick_px_under_infographic(names, fig.dpi)


def test_axes_position_stable_across_styles():
    df = _long_name_df()
    plots.set_theme(style_preset="default")
    x0_default = plots.missing_bar(df).get_position().x0
    plots.set_theme(style_preset="infographic")
    x0_info = plots.missing_bar(df).get_position().x0
    assert x0_default == pytest.approx(x0_info, abs=1e-6)


def test_switch_style_position_and_overlap_stable():
    """Infographic → plot → measure, then default → plot → measure: both the
    axes position and the no-overlap guarantee hold in one test."""
    df = _long_name_df()
    positions = []
    for preset in ("infographic", "default"):
        plots.set_theme(style_preset=preset)
        ax = plots.missing_bar(df)
        ax.figure.canvas.draw()
        positions.append(ax.get_position().x0)
        left = ax.get_window_extent().x0
        assert all(t.get_window_extent().x1 <= left + 1.0
                   for t in ax.get_yticklabels())
    assert positions[0] == pytest.approx(positions[1], abs=1e-6)


def test_guardrail_caps_margin_and_ellipsizes():
    """A pathologically long name hits the cap and ellipsizes instead of
    crushing the plotting area to nothing."""
    plots.set_theme(style_preset="infographic")
    df = pd.DataFrame({
        "z" * 200: np.r_[np.nan, np.ones(9)],
        "short": np.r_[np.nan, np.nan, np.ones(8)],
    })
    ax = plots.missing_bar(df)
    ax.figure.canvas.draw()
    x0 = ax.get_position().x0
    assert x0 <= plots._HBAR_LEFT_CAP + 1e-6           # margin capped
    assert ax.get_position().x1 - x0 >= 0.2            # usable plotting band
    assert any(t.get_text().endswith("…") for t in ax.get_yticklabels())


def test_bar_horizontal_path_stable_across_styles(df):
    """`bar`'s horizontal path gets the same worst-case reservation."""
    plots.set_theme(style_preset="default")
    x0_default = plots.bar(df, "city").get_position().x0
    plots.set_theme(style_preset="infographic")
    x0_info = plots.bar(df, "city").get_position().x0
    assert x0_default == pytest.approx(x0_info, abs=1e-6)
