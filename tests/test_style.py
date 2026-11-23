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
    assert np.allclose(cmap(norm(60.0)), cmap.get_over())


def test_wind_thresholds_are_bin_edges():
    assert style.SEVERE_WIND in style.WIND_BOUNDS
    assert style.SIG_SEVERE_WIND == 33.5 and style.SIG_SEVERE_WIND in style.WIND_BOUNDS
    assert {style.SEVERE_WIND, style.SIG_SEVERE_WIND} <= set(style.WIND_TICKS)


def _lightness(rgba):
    """Relative luminance (0 dark .. 1 light)."""
    r, g, b = rgba[:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def test_wind_colormap_smooth_within_regimes_and_jumps_at_thresholds():
    colors = style.wind_colors()
    edges = list(style.WIND_BOUNDS)
    severe, sig = edges.index(style.SEVERE_WIND), edges.index(style.SIG_SEVERE_WIND)
    steps = [np.abs(np.subtract(colors[i + 1][:3], colors[i][:3])).sum() for i in range(len(colors) - 1)]
    # the two largest color changes between neighboring bins are exactly at the thresholds
    assert sorted(np.argsort(steps)[-2:]) == [severe - 1, sig - 1]
    # each regime runs one way in lightness (a smooth sequential ramp)
    for lo, hi in ((0, severe), (severe, sig), (sig, len(colors))):
        lum = [_lightness(c) for c in colors[lo:hi]]
        assert all(b < a for a, b in zip(lum, lum[1:], strict=False))


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
