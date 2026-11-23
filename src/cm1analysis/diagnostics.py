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

    return buoyancy_from(cm1, read)


def buoyancy_from(cm1: CM1File, read: Callable[[str], np.ndarray]) -> np.ndarray:
    """Buoyancy from fields supplied by ``read(name)`` (any shape: a level, a section, ...).

    See :func:`buoyancy`. Without qvpert, the base-state vapor is the mean of
    what ``read`` returns along its last axis.
    """
    thpert = read("thpert")
    b = thpert / (read("th") - thpert)
    if cm1.has("qvpert"):
        b += EPS_VIRTUAL * read("qvpert")
    elif cm1.has("qv"):
        qv = read("qv")
        b += EPS_VIRTUAL * (qv - np.nanmean(qv, axis=-1, keepdims=True))
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

    def read(name: str) -> np.ndarray:
        return cm1.read(name, time, level=level, y=ys, x=xs, dtype=np.float64)

    return density_from(cm1, read)


def density_from(cm1: CM1File, read: Callable[[str], np.ndarray]) -> np.ndarray:
    """Air density from fields supplied by ``read(name)``; see :func:`air_density`."""
    if _require(cm1, "air_density") == ("rho",):
        return read("rho")
    p = read("prs")
    temp = read("th") * (p / P00) ** (RD / CP)
    if cm1.has("qv"):
        temp = temp * (1.0 + EPS_VIRTUAL * read("qv"))
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


# --------------------------------------------------------- base state
def base_state_profile(cm1: CM1File, total: str, pert: str, time: int = 0) -> np.ndarray:
    """CM1 base-state profile total - pert (e.g. th - thpert) from one corner column, float64.

    CM1's base state is horizontally uniform, so a single column is exact.
    Staggered fields (u - upert) use their own grid's corner column.
    """
    a = cm1.read(total, time, y=0, x=0, dtype=np.float64)
    b = cm1.read(pert, time, y=0, x=0, dtype=np.float64)
    return a - b


def freezing_level(cm1: CM1File, time: int = 0) -> int | None:
    """Index of the lowest scalar level where the base-state temperature is at or below 0 C."""
    if not cm1.has("th", "thpert", "prs", "prspert"):
        return None
    theta0 = base_state_profile(cm1, "th", "thpert", time)
    p0 = base_state_profile(cm1, "prs", "prspert", time)
    t0 = theta0 * (p0 / P00) ** (RD / CP)
    below = np.flatnonzero(t0 <= 273.15)
    return int(below[0]) if below.size else None


def cold_pool_c(
    cm1: CM1File, time: int, window: Window = FULL, filter_dbz: float | None = 10.0
) -> np.ndarray:
    """Cold pool intensity C = sqrt(2 * integral of max(-B, 0) dz), Squitieri and Gallus (2022) Eq. 1.

    Integration in each column runs up from the surface and stops at the first
    level where B >= 0 (the cold pool top) or at the base-state freezing
    level, whichever is lower. C is set to 0 where composite reflectivity is
    below ``filter_dbz`` (None disables the filter). Reads one level at a time.
    """
    _require(cm1, "buoyancy")
    zf_m = cm1.zf_km * 1000.0
    dz = np.diff(zf_m)
    ktop = freezing_level(cm1, time) or dz.size
    c_sq = None
    active = None
    for k in range(ktop):
        b = buoyancy(cm1, time, k, window)
        if c_sq is None:
            c_sq = np.zeros_like(b)
            active = np.ones(b.shape, dtype=bool)
        c_sq += 2.0 * dz[k] * np.where(active, np.maximum(-b, 0.0), 0.0)
        active &= b < 0.0
        if not active.any():
            break
    c = np.sqrt(c_sq)
    if filter_dbz is not None and cm1.has("cref"):
        ys, xs = window
        c[cm1.read("cref", time, y=ys, x=xs) < filter_dbz] = 0.0
    return c
