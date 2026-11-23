import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from cm1analysis import style  # noqa: E402


def test_wind_colormap_one_color_per_bin():
    cmap, norm = style.wind_colormap()
    assert cmap.N == len(style.WIND_BOUNDS) - 1 == norm.Ncmap
    bins = np.array(style.WIND_BOUNDS)
    centers = 0.5 * (bins[1:] + bins[:-1])
    assert list(norm(centers)) == list(range(cmap.N))


def test_wind_colormap_matches_legacy_thresholds():
    cmap, norm = style.wind_colormap()
    color = dict(zip(style.WIND_BOUNDS[:-1], style.WIND_COLORS, strict=True))
    assert color[26.0] == "yellow"  # severe threshold
    assert color[33.4] == "darkred"  # significant-severe threshold
    assert color[47.5] == "navy"  # what the legacy 21-color list actually drew
    assert np.allclose(cmap(norm(60.0)), cmap.get_over())


def test_diverging_colormap_is_symmetric():
    cmap, norm = style.diverging_colormap(vmax=10.0, step=2.0)
    assert norm.boundaries[0] == -10.0 and norm.boundaries[-1] == 10.0
    assert len(norm.boundaries) == 11
    with pytest.raises(ValueError):
        style.diverging_colormap(vmax=0.0)


def test_apply_fonts(monkeypatch):
    monkeypatch.setattr(mpl, "rcParams", mpl.RcParams(mpl.rcParams))
    fonts = style.apply_fonts({"title": 20})
    assert fonts["title"] == 20 and fonts["tick"] == style.FONT_DEFAULTS["tick"]
    assert mpl.rcParams["axes.titlesize"] == 20
    with pytest.raises(ValueError, match="unknown font keys"):
        style.apply_fonts({"titel": 20})


def test_save_figure_creates_directories(tmp_path):
    fig, _ = plt.subplots()
    out = style.save_figure(fig, tmp_path / "a" / "b" / "fig.png", dpi=50)
    assert out.exists() and out.stat().st_size > 0
    assert not plt.get_fignums()
