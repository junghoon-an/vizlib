"""The opt-in "infographic" style preset."""

import matplotlib.colors as mcolors
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


def test_line_area_adds_gradient_image(df):
    ax = plots.line(df, "age", "height", area=True)
    assert len(ax.images) >= 1  # gradient fill image


def test_stacked_area_uses_traffic_light_and_swatch_legend():
    n = 20
    sdf = pd.DataFrame({
        "t": list(range(n)) * 3,
        "v": [1.0] * (n * 3),
        "sev": ["low"] * n + ["medium"] * n + ["high"] * n,
    })
    ax = plots.line(sdf, "t", "v", hue="sev", area=True, stack=True)
    assert ax.get_legend() is not None  # swatch legend
    faces = [tuple(round(c, 3) for c in coll.get_facecolor()[0])
             for coll in ax.collections if len(coll.get_facecolor())]
    for hexc in plots._TRAFFIC_LIGHT:
        want = tuple(round(c, 3) for c in mcolors.to_rgba(hexc))
        assert any(all(abs(f[i] - want[i]) < 0.02 for i in range(3)) for f in faces)


def test_callouts_add_annotations(df):
    import matplotlib.text as mtext

    anns = [(df["age"].iloc[3], "event A"), (df["age"].iloc[5], "event B")]
    ax = plots.line(df, "age", "height", annotations=anns)
    drawn = [c for c in ax.get_children() if isinstance(c, mtext.Annotation)]
    assert len(drawn) == 2
    assert all(a.arrow_patch is not None for a in drawn)


def test_donut_renders_center_and_wedges(df):
    ax = plots.donut(df["city"], center_text="Cities")
    assert isinstance(ax, Axes)
    assert len(ax.patches) >= 2  # wedges
    assert any(t.get_text() == "Cities" for t in ax.texts)  # center caption


def test_infographic_ax_honored_and_no_mutation(df):
    import matplotlib.pyplot as plt

    plots.set_theme(style_preset="infographic")
    before = df.copy()
    fig, ax = plt.subplots()
    out = plots.bar(df, "city", ax=ax)
    assert out is ax
    plots.line(df, "age", "height", area=True)
    plots.donut(df["group"])
    pd.testing.assert_frame_equal(df, before)
