"""The opt-in dark-navy "neon" dashboard preset."""

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes

from vizlib import plots


def _navy():
    return mcolors.to_rgba("#1B1B3E")


def test_neon_preset_sets_dark_navy_and_neon_palette(df):
    plots.set_theme(style_preset="neon")
    assert plots._THEME["palette"] == plots._NEON_PALETTE
    assert plots._THEME["background"] == "#1B1B3E"
    assert plots._THEME["bold_labels"] is True
    # light text so labels read on the dark panel
    assert mcolors.to_rgba(plots._THEME["text_color"])[0] > 0.8


def test_neon_restores_to_default():
    plots.set_theme(style_preset="neon")
    plots.set_theme(style_preset="default")
    assert plots._THEME["palette"] == "colorblind"
    assert plots._THEME["background"] is None


@pytest.mark.parametrize("make", [
    lambda df: plots.bar(df, "city"),
    lambda df: plots.missing_bar(df),
    lambda df: plots.scatter(df, "age", "height", hue="group"),
    lambda df: plots.line(df, "age", "height", area=True),
    lambda df: plots.donut(df["city"]),
    lambda df: plots.hist(df["age"]),
])
def test_neon_charts_return_axes_without_mutation(df, make):
    plots.set_theme(style_preset="neon")
    before = df.copy()
    assert isinstance(make(df), Axes)
    pd.testing.assert_frame_equal(df, before)


def test_neon_paints_dark_background():
    plots.set_theme(style_preset="neon")
    ax = plots.hist(pd.Series(np.arange(50.0)))
    assert ax.get_facecolor() == pytest.approx(_navy())
    assert ax.figure.get_facecolor() == pytest.approx(_navy())


def test_neon_swatch_legend_text_is_light():
    plots.set_theme(style_preset="neon")
    n = 12
    sdf = pd.DataFrame({"t": list(range(n)) * 3, "v": [1.0] * (n * 3),
                        "g": ["a"] * n + ["b"] * n + ["c"] * n})
    ax = plots.line(sdf, "t", "v", hue="g")
    legend = ax.get_legend()
    light = mcolors.to_rgba(plots._THEME["text_color"])
    assert all(mcolors.to_rgba(t.get_color()) == pytest.approx(light)
               for t in legend.get_texts())


def test_neon_callout_box_matches_dark_surface(df):
    import matplotlib.text as mtext

    plots.set_theme(style_preset="neon")
    ax = plots.line(df, "age", "height",
                    annotations=[(df["age"].iloc[3], "peak")])
    ann = [c for c in ax.get_children() if isinstance(c, mtext.Annotation)][0]
    assert ann.get_bbox_patch().get_facecolor() == pytest.approx(_navy())


def test_neon_donut_separators_match_dark_surface(df):
    plots.set_theme(style_preset="neon")
    ax = plots.donut(df["city"])
    wedges = [p for p in ax.patches]
    assert wedges and all(
        w.get_edgecolor() == pytest.approx(_navy()) for w in wedges
    )
