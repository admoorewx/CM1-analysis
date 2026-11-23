"""Shared figure style: colormaps, font sizes and saving.

Kept in one place so every figure in a paper or review uses the same scales.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib as mpl
import numpy as np
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

# Wind speed bins in m/s. 26.0 and 33.4 m/s are the severe (58 mph) and
# significant-severe (75 mph) wind thresholds, so they get their own edges.
WIND_BOUNDS = (
    0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5,
    26.0, 27.5, 30.0, 33.4, 35.0, 37.5, 40.0, 42.5, 45.0, 47.5, 50.0,
)  # fmt: skip
WIND_COLORS = (
    "black", "navy", "blue", "royalblue", "cornflowerblue", "dodgerblue",
    "darkturquoise", "mediumspringgreen", "aquamarine", "palegreen",
    "yellow", "gold", "orange", "darkred", "indianred", "lightcoral",
    "violet", "mediumorchid", "indigo", "navy",
)  # fmt: skip

# Diverging scale for vertical velocity and other signed fields:
# strong negative (purple) -> blue -> white at zero -> red -> strong positive (pink).
DIVERGING_COLORS = ("#B976E6", "#00008B", "#1E90FF", "#FFFFFF", "#FF0000", "#8B0000", "#FF69B4")


def wind_colormap() -> tuple[ListedColormap, BoundaryNorm]:
    """Discrete wind speed colormap and norm (one color per bin, m/s).

    The legacy scripts listed 21 colors for these 20 bins, so matplotlib
    skipped the second-to-last ("blue"); this is the 20 colors actually drawn.
    """
    cmap = ListedColormap(WIND_COLORS, name="cm1_wind").with_extremes(over=WIND_COLORS[-1])
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
