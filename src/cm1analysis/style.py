"""Shared figure style: colormaps, font sizes and saving.

Kept in one place so every figure in a paper or review uses the same scales.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap

# Font sizes in points. cbar_* and contour_label are not rcParams; plot code
# passes them explicitly (fontsize=fonts["cbar_label"]).
FONT_DEFAULTS: dict[str, int] = {
    "title": 17,  # axes titles
    "suptitle": 15,  # figure titles
    "axis_label": 16,
    "tick": 10,
    "cbar_label": 16,
    "cbar_tick": 12,
    "legend": 14,
    "contour_label": 12,
}

DEFAULT_DPI = 200

# Wind speed thresholds (m/s): severe is 58 mph, significant severe 75 mph.
SEVERE_WIND = 26.0
SIG_SEVERE_WIND = 33.5

# The wind scale has three regimes, each a smooth ramp from one sequential
# colormap, with a jump in hue and lightness at each threshold: calm-to-strong
# sub-severe winds run pale green to dark blue, severe winds bright yellow to
# orange, significant-severe winds magenta to deep purple. Each entry is
# (bin edges, colormap, sampled range of that colormap).
WIND_REGIMES = (
    (tuple(np.arange(0.0, SEVERE_WIND + 0.1, 2.0)), "GnBu", (0.05, 0.9)),
    ((SEVERE_WIND, 28.0, 30.0, 32.0, SIG_SEVERE_WIND), "YlOrRd", (0.1, 0.55)),
    ((SIG_SEVERE_WIND, 35.0, 37.5, 40.0, 42.5, 45.0, 47.5, 50.0), "RdPu", (0.55, 1.0)),
)
WIND_BOUNDS = tuple(
    float(b) for i, (edges, _, _) in enumerate(WIND_REGIMES) for b in (edges if i == 0 else edges[1:])
)
WIND_TICKS = (0.0, 10.0, 20.0, SEVERE_WIND, SIG_SEVERE_WIND, 40.0, 50.0)


# Diverging scale for vertical velocity and other signed fields:
# strong negative (purple) -> blue -> white at zero -> red -> strong positive (pink).
DIVERGING_COLORS = ("#B976E6", "#00008B", "#1E90FF", "#FFFFFF", "#FF0000", "#8B0000", "#FF69B4")


def wind_colors() -> list[tuple[float, float, float, float]]:
    """RGBA color of each wind bin, lowest first (see WIND_REGIMES)."""
    colors = []
    for edges, name, (lo, hi) in WIND_REGIMES:
        colors += [colormaps[name](v) for v in np.linspace(lo, hi, len(edges) - 1)]
    return colors


def wind_colormap() -> tuple[ListedColormap, BoundaryNorm]:
    """Discrete wind speed colormap and norm (one color per bin, m/s).

    Winds above 50 m/s use the top color.
    """
    colors = wind_colors()
    cmap = ListedColormap(colors, name="cm1_wind").with_extremes(over=colors[-1])
    return cmap, BoundaryNorm(WIND_BOUNDS, cmap.N)


def diverging_colormap(vmax: float, step: float = 1.0) -> tuple[LinearSegmentedColormap, BoundaryNorm]:
    """Symmetric diverging colormap with discrete bins of width `step` from -vmax to vmax."""
    if vmax <= 0 or step <= 0:
        raise ValueError("vmax and step must be positive")
    cmap = LinearSegmentedColormap.from_list("cm1_diverging", DIVERGING_COLORS)
    nbins = round(2 * vmax / step)
    bounds = np.linspace(-vmax, vmax, nbins + 1)
    return cmap, BoundaryNorm(bounds, cmap.N)


def apply_fonts(overrides: Mapping[str, int] | None = None) -> dict[str, int]:
    """Set matplotlib rcParams from the font sizes and return the merged sizes."""
    fonts = dict(FONT_DEFAULTS)
    if overrides:
        unknown = set(overrides) - set(FONT_DEFAULTS)
        if unknown:
            raise ValueError(f"unknown font keys {sorted(unknown)}; expected {sorted(FONT_DEFAULTS)}")
        fonts.update(overrides)
    mpl.rcParams.update(
        {
            "font.size": fonts["tick"],
            "axes.titlesize": fonts["title"],
            "axes.labelsize": fonts["axis_label"],
            "xtick.labelsize": fonts["tick"],
            "ytick.labelsize": fonts["tick"],
            "legend.fontsize": fonts["legend"],
            "figure.titlesize": fonts["suptitle"],
        }
    )
    return fonts


def save_figure(fig, path: str | Path, dpi: int = DEFAULT_DPI) -> Path:
    """Save and close a figure, creating the directory if needed. Returns the path."""
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
