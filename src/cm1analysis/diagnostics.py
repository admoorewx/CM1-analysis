"""Derived fields computed from CM1 output.

Each derived quantity lists the sets of raw fields it can be computed from, in
order of preference (``RECIPES``). Plots name derived quantities in their
``requires`` just like raw fields, and the registry checks the recipes.

All functions take a ``window`` of (y slice, x slice) so plots of a
subdomain read only that part of the file.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .io import CM1File

G = 9.81  # m s-2
EPS_VIRTUAL = 0.61  # (Rv / Rd) - 1
RD = 287.04  # J kg-1 K-1, dry-air gas constant (CM1 value)
CP = 1005.7  # J kg-1 K-1 (CM1 value)
P00 = 100000.0  # Pa

Window = tuple[slice, slice]
FULL: Window = (slice(None), slice(None))

# Derived quantity -> alternative sets of raw fields, most preferred first.
RECIPES: dict[str, tuple[tuple[str, ...], ...]] = {
    "low_level_wind": (("u10", "v10"), ("uinterp", "vinterp"), ("u", "v")),
    "surface_pressure": (("psfc",), ("prs",)),
    "buoyancy": (("th", "thpert"),),
    "air_density": (("rho",), ("prs", "th")),
    "w_scalar": (("winterp",), ("w",)),
}


def recipe(cm1: CM1File, quantity: str) -> tuple[str, ...] | None:
    """First recipe for a derived quantity that the file can satisfy, or None."""
    for fields in RECIPES[quantity]:
        if cm1.has(*fields):
            return fields
    return None


def _require(cm1: CM1File, quantity: str) -> tuple[str, ...]:
    fields = recipe(cm1, quantity)
    if fields is None:
        options = " or ".join("+".join(r) for r in RECIPES[quantity])
        raise KeyError(f"{cm1.path.name} cannot provide {quantity}: needs {options}")
    return fields


# ------------------------------------------------------------------ wind
@dataclass(frozen=True)
class Wind:
    u: np.ndarray
    v: np.ndarray
    label: str  # e.g. "10-m" or "lowest-level (50 m)", for titles

    @property
    def speed(self) -> np.ndarray:
        return np.hypot(self.u, self.v)


def low_level_wind(cm1: CM1File, time: int, window: Window = FULL) -> Wind:
    """10-m wind if the file has it, otherwise the lowest model level on scalar points.

    The label says which, so a figure never calls a lowest-level wind "10-m".
    """
    ys, xs = window
    fields = _require(cm1, "low_level_wind")
    if fields == ("u10", "v10"):
        return Wind(cm1.read("u10", time, y=ys, x=xs), cm1.read("v10", time, y=ys, x=xs), "10-m")
    label = f"lowest-level ({cm1.z_km[0] * 1000:.0f} m)"
    if fields == ("uinterp", "vinterp"):
        u = cm1.read("uinterp", time, level=0, y=ys, x=xs)
        v = cm1.read("vinterp", time, level=0, y=ys, x=xs)
        return Wind(u, v, label)
    # Staggered u and v: average the two faces around each scalar point.
    u = cm1.read("u", time, level=0, y=ys, x=_widen(xs, cm1.x_km.size))
    v = cm1.read("v", time, level=0, y=_widen(ys, cm1.y_km.size), x=xs)
    return Wind(0.5 * (u[:, 1:] + u[:, :-1]), 0.5 * (v[1:, :] + v[:-1, :]), label)


def _widen(s: slice, n: int) -> slice:
    """Scalar-point slice -> the face slice that brackets it (one extra point)."""
    start, stop, step = s.indices(n)
    if step != 1:
        raise ValueError("strided windows are not supported on staggered fields")
    return slice(start, stop + 1)


# -------------------------------------------------------------- pressure
def surface_pressure(cm1: CM1File, time: int, window: Window = FULL) -> tuple[np.ndarray, str]:
    """Surface pressure in hPa and a label: psfc if written, else the lowest-level prs."""
    ys, xs = window
    if _require(cm1, "surface_pressure") == ("psfc",):
        return cm1.read("psfc", time, y=ys, x=xs) / 100.0, "Surface"
    return cm1.read("prs", time, level=0, y=ys, x=xs) / 100.0, "Lowest-level"


# -------------------------------------------------------------- buoyancy
def buoyancy(cm1: CM1File, time: int, level: int, window: Window = FULL) -> np.ndarray:
    """Buoyancy B = g (theta'/theta0 + 0.61 qv' - q_hydrometeors) at one level, m s-2.

    The base state is CM1's own (theta0 = th - thpert, qv' = qvpert), not an
    estimate from a reference column. Moisture terms are included when the
    file has them; water loading uses every hydrometeor species in the file.
    """
    ys, xs = window
    _require(cm1, "buoyancy")

    def read(name: str) -> np.ndarray:
        return cm1.read(name, time, level=level, y=ys, x=xs, dtype=np.float64)

    thpert = read("thpert")
    theta0 = read("th") - thpert
    b = thpert / theta0
    if cm1.has("qvpert"):
        b += EPS_VIRTUAL * read("qvpert")
    elif cm1.has("qv"):
        # No perturbation written: use the level mean as the base state.
        qv = read("qv")
        b += EPS_VIRTUAL * (qv - np.nanmean(qv))
    for name in cm1.hydrometeors:
        b -= read(name)
    return G * b


# ------------------------------------------------------- column reductions
def column_reduce(
    cm1: CM1File,
    name: str,
    time: int,
    reducer: Callable[[np.ndarray, np.ndarray], np.ndarray] = np.fmax,
    window: Window = FULL,
    top_km: float | None = None,
) -> np.ndarray:
    """Reduce a 3-D field over the column (default: maximum), one level at a time.

    Reading level by level keeps memory at one 2-D slice, which matters for
    full-domain 3-D fields. ``top_km`` stops at that height. NaNs are ignored
    (np.fmax / np.fmin).
    """
    ys, xs = window
    heights = cm1.zf_km if cm1.location(name) == "w" else cm1.z_km
    nlev = cm1.ds[name].shape[-3]
    if top_km is not None and heights is not None:
        nlev = min(nlev, int(np.searchsorted(heights, top_km, side="right")) or 1)
    result = cm1.read(name, time, level=0, y=ys, x=xs)
    for k in range(1, nlev):
        result = reducer(result, cm1.read(name, time, level=k, y=ys, x=xs))
    return result


# ------------------------------------------------- density, w, mass flux
def air_density(cm1: CM1File, time: int, level: int, window: Window = FULL) -> np.ndarray:
    """Air density (kg m-3) at one level: CM1's rho if written, else the ideal gas law.

    The ideal-gas form uses T = th (p/p00)^(Rd/cp), with the virtual
    temperature correction when qv is available.
    """
    ys, xs = window
    if _require(cm1, "air_density") == ("rho",):
        return cm1.read("rho", time, level=level, y=ys, x=xs, dtype=np.float64)
    p = cm1.read("prs", time, level=level, y=ys, x=xs, dtype=np.float64)
    temp = cm1.read("th", time, level=level, y=ys, x=xs, dtype=np.float64) * (p / P00) ** (RD / CP)
    if cm1.has("qv"):
        temp *= 1.0 + EPS_VIRTUAL * cm1.read("qv", time, level=level, y=ys, x=xs, dtype=np.float64)
    return p / (RD * temp)


def w_scalar(cm1: CM1File, time: int, level: int, window: Window = FULL) -> np.ndarray:
    """Vertical velocity on a scalar level: winterp if written, else the mean of the faces around it."""
    ys, xs = window
    if _require(cm1, "w_scalar") == ("winterp",):
        return cm1.read("winterp", time, level=level, y=ys, x=xs)
    faces = cm1.read("w", time, level=slice(level, level + 2), y=ys, x=xs)
    return 0.5 * (faces[0] + faces[1])


def column_mass_flux_extremes(
    cm1: CM1File,
    time: int,
    species: list[str],
    window: Window = FULL,
    top_km: float | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Column max and min of the vertical mass flux rho q w (g m-2 s-1) of each species.

    One pass over the levels: w and density are read once per level and shared
    by every species. This is transport by the air motion only; CM1 does not
    write fall speeds by default, so sedimentation is not included.
    """
    _require(cm1, "air_density")
    _require(cm1, "w_scalar")
    nlev = cm1.z_km.size
    if top_km is not None:
        nlev = max(1, int(np.searchsorted(cm1.z_km, top_km, side="right")))
    fmax: dict[str, np.ndarray] = {}
    fmin: dict[str, np.ndarray] = {}
    ys, xs = window
    for k in range(nlev):
        rho_w = 1000.0 * air_density(cm1, time, k, window) * w_scalar(cm1, time, k, window)
        for name in species:
            flux = rho_w * cm1.read(name, time, level=k, y=ys, x=xs)
            fmax[name] = flux if k == 0 else np.fmax(fmax[name], flux)
            fmin[name] = flux if k == 0 else np.fmin(fmin[name], flux)
    return fmax, fmin
