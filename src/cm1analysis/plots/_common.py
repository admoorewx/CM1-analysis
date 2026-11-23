"""Plan-view building blocks shared by the plot modules.

``FIELD_STYLES`` fixes the colormap, contour levels and label of every field
that is plotted in more than one figure, so a quantity looks the same
wherever it appears.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import BoundaryNorm, Colormap, LinearSegmentedColormap, ListedColormap

from ..style import WIND_BOUNDS, WIND_TICKS, wind_colormap

FILL_ALPHA = 0.5  # the legacy plan views drew filled contours at half opacity


@dataclass(frozen=True)
class FieldStyle:
    cmap: str | Colormap
    levels: Sequence[float]
    label: str
    norm: BoundaryNorm | None = None
    extend: str = "neither"
    alpha: float = FILL_ALPHA
    ticks: Sequence[float] | None = None  # colorbar ticks; None lets matplotlib choose


_WIND_CMAP, _WIND_NORM = wind_colormap()
# Opaque, so the regime colors (and the jumps at the severe thresholds) stay true.
WIND = FieldStyle(
    _WIND_CMAP,
    WIND_BOUNDS,
    "Wind speed (m s$^{-1}$)",
    norm=_WIND_NORM,
    extend="max",
    alpha=1.0,
    ticks=WIND_TICKS,
)

# Hydrometeor colors follow the cross-section convention:
# cloud water cyan, rain green, ice orange, snow blue, graupel purple, hail red.
SPECIES_COLORS = {
    "qc": "c",
    "qr": "green",
    "qi": "orange",
    "qs": "blue",
    "qg": "purple",
    "qhl": "red",
    "qh": "red",
}
SPECIES_NAMES = {
    "qc": "Cloud water",
    "qr": "Rain",
    "qi": "Cloud ice",
    "qs": "Snow",
    "qg": "Graupel",
    "qhl": "Hail",
    "qh": "Hail",
    "qv": "Water vapor",
}
HYDROMETEOR_LEVELS = (0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)  # g kg-1
# Vertical mass flux spans 1-200 g m-2 s-1 in a derecho, so the levels are quasi-logarithmic.
FLUX_LEVELS = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0)  # g m-2 s-1
DOWNDRAFT_LEVELS = (-25.0, -20.0, -15.0, -12.0, -10.0, -8.0, -6.0, -4.0, -2.0, -1.0)  # m s-1
FLUX_DOWN_LEVELS = (-50.0, -20.0, -5.0)  # dashed contours of downward flux

FIELD_STYLES: dict[str, FieldStyle] = {
    "cref": FieldStyle("gist_ncar", np.arange(0.0, 95.0, 0.5), "Reflectivity (dBZ)"),
    "cpc": FieldStyle("cubehelix_r", np.arange(0.0, 50.0, 1.0), "Cold pool intensity C (m s$^{-1}$)"),
    "wind": WIND,
    # Derecho cold pools reach about -15 K at the lowest level.
    "thpert": FieldStyle("bwr", np.arange(-15.0, 15.5, 0.5), r"$\theta'$ (K)", extend="both"),
    "coldpool_min": FieldStyle("Blues_r", np.arange(-20.0, 0.5, 0.5), r"Minimum $\theta'$ (K)", extend="min"),
    "sus": FieldStyle("CMRmap_r", np.arange(0.0, 100.0, 0.5), "Vertical velocity (m s$^{-1}$)", extend="max"),
    # Up to 1025 hPa so the mesohigh under the cold pool is not cut off.
    "pressure": FieldStyle("RdBu_r", np.arange(990.0, 1025.5, 0.5), "Pressure (hPa)", extend="both"),
    # Pressure perturbation: mesohighs reach about +10 hPa, mid-level lows about -9 hPa.
    "prspert": FieldStyle("RdBu_r", np.arange(-8.0, 8.25, 0.5), "Pressure perturbation (hPa)", extend="both"),
    # Strongest downdraft in the column. Bins are finer for weak downdrafts so the
    # common -2 to -10 m/s range is resolved; the top edge is below zero so still air stays white.
    "downdraft": FieldStyle(
        # Blues trimmed at the light end, so the weakest bin is still visibly blue.
        ListedColormap(colormaps["Blues"](np.linspace(0.98, 0.22, len(DOWNDRAFT_LEVELS) - 1))),
        DOWNDRAFT_LEVELS,
        "Minimum vertical velocity (m s$^{-1}$)",
        norm=BoundaryNorm(DOWNDRAFT_LEVELS, len(DOWNDRAFT_LEVELS) - 1),
        extend="min",
        alpha=1.0,
        ticks=DOWNDRAFT_LEVELS,
    ),
    "t2": FieldStyle("rainbow", np.arange(290.0, 315.5, 0.5), "Temperature (K)", extend="both"),
    "buoyancy": FieldStyle(
        # Lowest-level cold pool buoyancy reaches about -0.5 m s-2.
        "RdBu_r",
        np.round(np.arange(-0.5, 0.501, 0.025), 3),
        "Buoyancy (m s$^{-2}$)",
        extend="both",
    ),
    "qv": FieldStyle("YlGnBu", np.arange(0.0, 22.0, 1.0), "Mixing ratio (g kg$^{-1}$)", extend="max"),
    "cooling": FieldStyle(
        "plasma",
        # Top edge just below zero so cells with no cooling stay unfilled (white).
        np.round(np.arange(-0.02, -0.0005, 0.001), 4),
        "Cooling rate (K s$^{-1}$)",
        extend="min",
        alpha=1.0,
    ),
    # Differences (experiment minus reference)
    "cref_diff": FieldStyle(
        "coolwarm", np.arange(-50.0, 52.5, 2.5), r"$\Delta$ reflectivity (dBZ)", extend="both"
    ),
    "wind_diff": FieldStyle(
        "coolwarm", np.arange(-30.0, 31.0, 1.0), r"$\Delta$ wind speed (m s$^{-1}$)", extend="both"
    ),
    "thpert_diff": FieldStyle(
        "coolwarm", np.arange(-10.0, 10.25, 0.5), r"$\Delta\theta'$ (K)", extend="both"
    ),
}


def species_style(name: str) -> FieldStyle:
    """Sequential white-to-color style for one hydrometeor's mixing ratio (g/kg)."""
    if name == "qv":
        return FIELD_STYLES["qv"]
    color = SPECIES_COLORS.get(name, "gray")
    cmap = LinearSegmentedColormap.from_list(f"{name}_seq", ["white", color])
    return FieldStyle(
        cmap,
        HYDROMETEOR_LEVELS,
        "Mixing ratio (g kg$^{-1}$)",
        norm=BoundaryNorm(HYDROMETEOR_LEVELS, 256, extend="max"),
        extend="max",
        alpha=1.0,
    )


def flux_style(name: str) -> FieldStyle:
    """White-to-color style for one hydrometeor's upward mass flux (g m-2 s-1)."""
    color = SPECIES_COLORS.get(name, "gray")
    cmap = LinearSegmentedColormap.from_list(f"{name}_flux", ["white", color])
    return FieldStyle(
        cmap,
        FLUX_LEVELS,
        "Upward mass flux (g m$^{-2}$ s$^{-1}$)",
        norm=BoundaryNorm(FLUX_LEVELS, 256, extend="max"),
        extend="max",
        alpha=1.0,
        ticks=FLUX_LEVELS,
    )


def panel_grid(nrows: int, ncols: int, panel: tuple[float, float] = (5.0, 4.3)):
    """Figure with an nrows x ncols grid of plan-view axes (constrained layout)."""
    fig, axs = plt.subplots(
        nrows, ncols, figsize=(panel[0] * ncols, panel[1] * nrows), layout="constrained", squeeze=False
    )
    return fig, axs


def planview(ax, x: np.ndarray, y: np.ndarray, data: np.ndarray, style: FieldStyle):
    """Filled contours of a 2-D field on km axes with equal aspect. Returns the mappable."""
    kwargs = {"norm": style.norm} if style.norm is not None else {}
    mappable = ax.contourf(
        x, y, data, levels=style.levels, cmap=style.cmap, extend=style.extend, alpha=style.alpha, **kwargs
    )
    ax.set_aspect("equal")
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(y[0], y[-1])
    return mappable


def colorbar(fig, mappable, ax, style: FieldStyle | str, fonts: Mapping[str, int]):
    """Colorbar beside one axes, or spanning a list of axes, labeled (and ticked) from the style."""
    label, ticks = (style, None) if isinstance(style, str) else (style.label, style.ticks)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.9, pad=0.02, ticks=ticks)
    if ticks is not None:
        cbar.ax.set_yticklabels([f"{t:g}" for t in ticks])
    cbar.set_label(label, fontsize=fonts["cbar_label"])
    cbar.ax.tick_params(labelsize=fonts["cbar_tick"])
    return cbar


def km_axes(ax, fonts: Mapping[str, int], xlabel: bool = True, ylabel: bool = True) -> None:
    if xlabel:
        ax.set_xlabel("x (km)", fontsize=fonts["axis_label"])
    if ylabel:
        ax.set_ylabel("y (km)", fontsize=fonts["axis_label"])
    ax.tick_params(labelsize=fonts["tick"])


def time_label(seconds: float) -> str:
    """'t = 2:05 (7500 s)' for titles."""
    total = round(seconds)
    hours, rem = divmod(total, 3600)
    return f"t = {hours}:{rem // 60:02d} ({total} s)"


def mark_maximum(
    ax, x: np.ndarray, y: np.ndarray, field: np.ndarray, fonts: Mapping[str, int], units: str
) -> None:
    """Dot and legend entry at the location of a field's maximum."""
    if not np.isfinite(field).any():
        return
    j, i = np.unravel_index(np.nanargmax(field), field.shape)
    ax.scatter(x[i], y[j], s=20, color="k", zorder=5, label=f"Max: {field[j, i]:.1f} {units}")
    ax.legend(loc="lower center", fontsize=fonts["legend"])


def same_grid(a, b) -> None:
    """Raise unless two frames share a horizontal grid, so a difference field is meaningful."""
    if (
        a.x.shape != b.x.shape
        or a.y.shape != b.y.shape
        or not (np.allclose(a.x, b.x) and np.allclose(a.y, b.y))
    ):
        raise ValueError(
            f"grids differ ({a.cm1.path.name}: {a.y.size}x{a.x.size}, "
            f"{b.cm1.path.name}: {b.y.size}x{b.x.size}); differences need identical grids"
        )
