"""The opt-in "infographic" style preset."""

import pandas as pd
import pytest
from matplotlib.axes import Axes

from vizlib import plots


def test_preset_switch_changes_and_restores(df):
    plots.set_theme(style_preset="infographic")
    assert plots._THEME["palette"] == plots._VIVID_PALETTE
    assert plots._THEME["background"] == "#FFFFFF"
    assert plots._THEME["bold_labels"] is True
    assert isinstance(plots.bar(df, "city"), Axes)  # still returns an Axes
    plots.set_theme(style_preset="default")
    assert plots._THEME["palette"] == "colorblind"
    assert plots._THEME["background"] is None
    assert isinstance(plots.hist(df["age"]), Axes)


def test_unknown_preset_raises():
    with pytest.raises(ValueError):
        plots.set_theme(style_preset="nope")


def test_infographic_bold_labels_with_contrast(df):
    plots.set_theme(style_preset="infographic")
    ax = plots.bar(df, "city")
    assert len(ax.texts) == len(ax.patches)  # one label per bar
    assert all(t.get_fontweight() == "bold" for t in ax.texts)
    allowed = {"white", (1.0, 1.0, 1.0, 1.0)}
    for t in ax.texts:  # contrasting: white on dark, else the dark text color
        col = t.get_color()
        assert col in allowed or col == plots._THEME["text_color"]


def test_default_labels_not_bold(df):
    ax = plots.bar(df, "city")  # default preset
    assert ax.texts and all(t.get_fontweight() != "bold" for t in ax.texts)


def test_bar_as_percent(df):
    ax = plots.bar(df, "city", as_percent=True)
    assert any("%" in t.get_text() for t in ax.texts)


def test_scatter_callouts_add_annotations(df):
    import matplotlib.text as mtext

    plots.set_theme(style_preset="infographic")
    anns = [(df["age"].iloc[3], df["height"].iloc[3], "event A")]
    ax = plots.scatter(df, "age", "height", annotations=anns)
    drawn = [c for c in ax.get_children() if isinstance(c, mtext.Annotation)]
    assert len(drawn) == 1
    assert all(a.arrow_patch is not None for a in drawn)


def test_infographic_ax_honored_and_no_mutation(df):
    import matplotlib.pyplot as plt

    plots.set_theme(style_preset="infographic")
    before = df.copy()
    fig, ax = plt.subplots()
    out = plots.bar(df, "city", ax=ax)
    assert out is ax
    plots.scatter(df, "age", "height")
    pd.testing.assert_frame_equal(df, before)
