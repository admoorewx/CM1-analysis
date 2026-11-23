"""West-east vertical cross sections that follow the storm.

Placement (one pass over a run, cached and shared by every cross-section
plot):

- Row: by default the row through the bow apex, taken as the forward-most
  point of the largest surface cold pool (lowest-level theta' at or below
  ``cp_threshold_k``). ``auto_row: max_wind`` uses the row of the strongest
  low-level wind (smoothed over ``smooth_km``) instead, and ``y_km`` pins
  the row.
- Window: ``window_km`` wide, with ``trailing_fraction`` of it behind (west
  of) the gust front on that row, or the row's reflectivity peak before a
  cold pool exists. Systems are assumed to move toward +x.
- Storm motion c_x: the eastward speed of the system's leading edge, from a
  least-squares fit of its position over ``motion_window_s`` centered on each
  time (Mahoney et al. 2009 average the leading edge's displacement over 1 h).
  The leading edge is the gust front (``leading_edge: cold_pool``) or the
  forward-most composite reflectivity of at least ``edge_dbz``
  (``leading_edge: reflectivity``, their definition, with 40 dBZ).
  ``storm_u`` fixes c_x instead.

Each section is averaged over ``average_km`` across the line (Mahoney and
Lackmann 2011 average their sections over 40 km). Products such as fluxes
must be formed row by row before averaging, so :class:`Section` exposes both
the band of rows (:meth:`Section.band`) and its mean (:meth:`Section.field`).

Reading a few rows of a 3-D CM1 field costs nearly as much as reading the
whole field when the file stores one horizontal level per chunk, so the band
of rows comes at no extra cost, and section arrays are kept in a size-limited
cache shared by all plots in the process.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from . import diagnostics
from .io import CM1File

AUTO_ROWS = ("apex", "max_wind")
LEADING_EDGES = ("cold_pool", "reflectivity")
CACHE_BYTES = 2 * 1024**3
PLACEMENT_KEYS = (
    "y_km", "auto_row", "window_km", "trailing_fraction", "cp_threshold_k", "storm_u", "smooth_km",
    "motion_window_s", "leading_edge", "edge_dbz",
)  # fmt: skip


@dataclass(frozen=True)
class Placement:
    """Where the section for one output time sits, and the storm motion then."""

    path: Path
    time: int
    seconds: float
    j: int  # row index
    xs: slice  # window columns
    gust_x_km: float | None  # surface cold-pool edge on row j
    storm_u: float | None  # c_x, m/s toward +x
    storm_source: str  # how c_x was obtained, for titles


# ---------------------------------------------------------------- placement
def placements(files: Sequence[Path], options: Mapping[str, Any]) -> dict[tuple[Path, int], Placement]:
    """Section placement for every output time of a run, keyed by (file, time index)."""
    frozen = tuple((k, _freeze(options.get(k))) for k in PLACEMENT_KEYS)
    return _placements(tuple(Path(f) for f in files), frozen)


def _freeze(value):
    return tuple(value) if isinstance(value, list) else value


@lru_cache(maxsize=16)
def _placements(files: tuple[Path, ...], frozen: tuple) -> dict[tuple[Path, int], Placement]:
    opt = dict(frozen)
    if opt["auto_row"] not in AUTO_ROWS:
        raise ValueError(f"auto_row must be one of {AUTO_ROWS}, got {opt['auto_row']!r}")
    if opt["leading_edge"] not in LEADING_EDGES:
        raise ValueError(f"leading_edge must be one of {LEADING_EDGES}, got {opt['leading_edge']!r}")
    rows = []  # (path, time, seconds, j, xs, gust_x_km, edge_x_km)
    for path in files:
        with CM1File(path) as cm1:
            for t, seconds in enumerate(cm1.times):
                j = _row(cm1, t, opt)
                low = cm1.read("thpert", t, level=0, y=j)
                cref = cm1.read("cref", t, y=j)
                cold = np.flatnonzero(low <= opt["cp_threshold_k"])
                gust = int(cold[-1]) if cold.size else None
                anchor = gust if gust is not None else int(np.nanargmax(cref))
                xs = _window(cm1.x_km, anchor, opt["window_km"], opt["trailing_fraction"])
                gust_x = None if gust is None else float(cm1.x_km[gust])
                if opt["leading_edge"] == "cold_pool":
                    edge_x = gust_x
                else:
                    strong = np.flatnonzero(cref >= opt["edge_dbz"])
                    edge_x = float(cm1.x_km[strong[-1]]) if strong.size else None
                rows.append((path, t, float(seconds), j, xs, gust_x, edge_x))

    if opt["storm_u"] is not None:
        speeds, source = [float(opt["storm_u"])] * len(rows), "configured"
    else:
        speeds = storm_motion([(r[2], r[6]) for r in rows], opt["motion_window_s"])
        edge = "gust front" if opt["leading_edge"] == "cold_pool" else f"{opt['edge_dbz']:g} dBZ edge"
        source = f"{edge}, {opt['motion_window_s'] / 60:g}-min fit"
    return {
        (path, t): Placement(path, t, seconds, j, xs, gx, u, source if u is not None else "unknown")
        for (path, t, seconds, j, xs, gx, _), u in zip(rows, speeds, strict=True)
    }


def storm_motion(track: Sequence[tuple[float, float | None]], window_s: float) -> list[float | None]:
    """Leading-edge speed (m/s) at each time: slope of a line fit to positions within window_s.

    The window is centered on each time. If it holds fewer than two positions,
    the nearest neighbors on each side are used instead, so a run with sparse
    output still gets a (local) estimate. Times without a leading edge get None.
    """
    times = np.array([t for t, _ in track], dtype=float)
    xs = np.array([np.nan if x is None else x for _, x in track], dtype=float)
    valid = np.isfinite(xs)
    speeds: list[float | None] = []
    for n, t in enumerate(times):
        if not valid[n]:
            speeds.append(None)
            continue
        pick = valid & (np.abs(times - t) <= window_s / 2.0 + 1e-6)
        if pick.sum() < 2:
            pick = valid & (np.abs(np.arange(times.size) - n) <= 1)
        if pick.sum() < 2 or np.ptp(times[pick]) == 0:
            speeds.append(None)
            continue
        slope = np.polyfit(times[pick], xs[pick], 1)[0]  # km/s
        speeds.append(float(slope * 1000.0))
    return speeds


def _row(cm1: CM1File, t: int, opt: Mapping[str, Any]) -> int:
    if opt["y_km"] is not None:
        y = float(opt["y_km"])
        if not cm1.y_km[0] <= y <= cm1.y_km[-1]:
            raise ValueError(f"y_km {y:g} is outside the domain ({cm1.y_km[0]:g}-{cm1.y_km[-1]:g} km)")
        return int(np.argmin(np.abs(cm1.y_km - y)))
    if opt["auto_row"] == "max_wind":
        from scipy.ndimage import uniform_filter

        speed = diagnostics.low_level_wind(cm1, t).speed
        size = max(1, round(opt["smooth_km"] * 1000.0 / (cm1.dx or 1000.0)))
        smooth = uniform_filter(np.nan_to_num(speed), size=size)
        return int(np.unravel_index(np.argmax(smooth), smooth.shape)[0])
    return _apex_row(cm1, t, opt["cp_threshold_k"])


def _apex_row(cm1: CM1File, t: int, threshold: float) -> int:
    """Row of the forward-most point of the largest surface cold pool (the bow apex)."""
    from scipy.ndimage import label

    cold = cm1.read("thpert", t, level=0) <= threshold
    labels, n = label(cold)
    if n == 0:  # no cold pool yet: the row with the strongest reflectivity
        cref = cm1.read("cref", t)
        return int(np.unravel_index(np.nanargmax(cref), cref.shape)[0])
    sizes = np.bincount(labels.ravel())[1:]
    jj, ii = np.nonzero(labels == 1 + int(np.argmax(sizes)))
    front = ii.max()
    return int(np.median(jj[ii >= front - 1]))  # middle of a flat leading edge


def _window(x_km: np.ndarray, anchor: int, width_km: float, trailing: float) -> slice:
    if not 0.0 <= trailing <= 1.0:
        raise ValueError(f"trailing_fraction must be between 0 and 1, got {trailing}")
    if width_km >= x_km[-1] - x_km[0]:
        return slice(0, x_km.size)
    lo = x_km[anchor] - trailing * width_km
    lo = min(max(lo, x_km[0]), x_km[-1] - width_km)  # shift, rather than shrink, at the domain edges
    i0 = int(np.searchsorted(x_km, lo))
    i1 = int(np.searchsorted(x_km, lo + width_km, side="right"))
    return slice(i0, i1)


# ------------------------------------------------------------------ section
_CACHE: OrderedDict[tuple, np.ndarray] = OrderedDict()
_CACHE_SIZE = [0]


def _cached(key: tuple, load) -> np.ndarray:
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    value = load()
    _CACHE[key] = value
    _CACHE_SIZE[0] += value.nbytes
    while _CACHE_SIZE[0] > CACHE_BYTES and len(_CACHE) > 1:
        _, old = _CACHE.popitem(last=False)
        _CACHE_SIZE[0] -= old.nbytes
    return value


def clear_cache() -> None:
    _CACHE.clear()
    _CACHE_SIZE[0] = 0


class Section:
    """An x-z section of one output time, averaged over a band of rows across the line.

    ``band(name)`` is (nz, nrows, nx) on scalar points; ``field(name)`` is its
    mean over the rows, (nz, nx). Derived quantities that multiply fields are
    formed on the band and then averaged (``mean``).
    """

    def __init__(self, cm1: CM1File, place: Placement, top_km: float, average_km: float = 0.0):
        self.cm1 = cm1
        self.place = place
        self.j, self.xs = place.j, place.xs
        z = cm1.z_km
        self.ktop = max(2, int(np.searchsorted(z, top_km, side="right")))
        self.z = z[: self.ktop]
        self.x = cm1.x_km[self.xs]
        self.y_km = float(cm1.y_km[self.j])
        dy = float(np.median(np.diff(cm1.y_km))) if cm1.y_km.size > 1 else 1.0
        half = max(0, round(0.5 * float(average_km or 0.0) / dy))
        self.rows = slice(max(0, self.j - half), min(cm1.y_km.size, self.j + half + 1))
        self.band_km = (self.rows.stop - self.rows.start) * dy  # width actually averaged

    @property
    def seconds(self) -> float:
        return self.place.seconds

    @property
    def averaged(self) -> bool:
        return self.rows.stop - self.rows.start > 1

    # ------------------------------------------------------------- fields
    def band(self, name: str) -> np.ndarray:
        """(nz, nrows, nx) on scalar points; staggered u and w are averaged onto them."""
        key = (str(self.cm1.path), self.place.time, name, self.rows.start, self.rows.stop,
               self.xs.start, self.xs.stop, self.ktop)  # fmt: skip
        return _cached(key, lambda: self._load(name))

    def _load(self, name: str) -> np.ndarray:
        cm1, t, k = self.cm1, self.place.time, self.ktop
        where = cm1.location(name)
        if where == "u":
            faces = cm1.read(
                name, t, level=slice(0, k), y=self.rows, x=slice(self.xs.start, self.xs.stop + 1)
            )
            return 0.5 * (faces[..., 1:] + faces[..., :-1])
        if where == "w":
            faces = cm1.read(name, t, level=slice(0, k + 1), y=self.rows, x=self.xs)
            return 0.5 * (faces[1:] + faces[:-1])
        if where != "scalar":
            raise ValueError(f"{name!r} is not a 3-D scalar, u or w field")
        return cm1.read(name, t, level=slice(0, k), y=self.rows, x=self.xs)

    @staticmethod
    def mean(band: np.ndarray) -> np.ndarray:
        """Average a (nz, nrows, nx) band over its rows."""
        return np.nanmean(band, axis=1)

    def field(self, name: str) -> np.ndarray:
        return self.mean(self.band(name))

    def u_band(self) -> np.ndarray:
        return self.band("uinterp" if self.cm1.has("uinterp") else "u")

    def w_band(self) -> np.ndarray:
        return self.band("winterp" if self.cm1.has("winterp") else "w")

    def u(self) -> np.ndarray:
        return self.mean(self.u_band())

    def w(self) -> np.ndarray:
        return self.mean(self.w_band())

    def u_relative(self) -> tuple[np.ndarray, str]:
        """u minus the storm motion c_x, and a label saying which frame that is."""
        u_s = self.place.storm_u
        if u_s is None:
            return self.u(), "ground-relative (storm motion unknown)"
        return self.u() - u_s, f"storm-relative, $c_x$ = {u_s:.1f} m s$^{{-1}}$ ({self.place.storm_source})"

    def surface(self, name: str) -> np.ndarray:
        """Lowest level of a field along the section (band mean)."""
        return self.field(name)[0]

    # ------------------------------------------------------------ derived
    def yvort(self) -> np.ndarray:
        """Horizontal vorticity about y, eta = du/dz - dw/dx (s-1); CM1's yvort if written."""
        if self.cm1.has("yvort"):
            return self.field("yvort")
        z_m, x_m = self.z * 1000.0, self.x * 1000.0
        return np.gradient(self.u(), z_m, axis=0) - np.gradient(self.w(), x_m, axis=1)  # linear: mean first

    def buoyancy(self) -> np.ndarray:
        return self.mean(diagnostics.buoyancy_from(self.cm1, lambda n: self.band(n).astype(np.float64)))

    def density_band(self) -> np.ndarray:
        return diagnostics.density_from(self.cm1, lambda n: self.band(n).astype(np.float64))

    def base_u(self) -> tuple[np.ndarray, str]:
        """Base-state (initial) wind u0(z) on this section's levels: u - upert, else the east edge."""
        if self.cm1.has("u", "upert"):
            prof = _base_profile(str(self.cm1.path), "u", "upert")
            return prof[: self.ktop], "initial wind (u - upert)"
        row = self.cm1.read(
            "uinterp" if self.cm1.has("uinterp") else "u",
            self.place.time,
            level=slice(0, self.ktop),
            y=self.j,
        )
        return row[:, -10:].mean(axis=1), "mean of the 10 easternmost columns"


@lru_cache(maxsize=32)
def _base_profile(path: str, total: str, pert: str) -> np.ndarray:
    """CM1 base state is horizontally uniform and fixed in time, so one column per file is enough."""
    with CM1File(path) as cm1:
        return diagnostics.base_state_profile(cm1, total, pert)


def plan_window(cm1: CM1File, place: Placement, zoom: float) -> tuple[slice, slice]:
    """Square plan-view window around the section, half-width ``zoom`` x section length."""
    xs = place.xs
    half = max(2, int(zoom * (xs.stop - xs.start)))
    mid = (xs.start + xs.stop) // 2
    nx, ny = cm1.x_km.size, cm1.y_km.size
    return slice(max(0, place.j - half), min(ny, place.j + half)), slice(
        max(0, mid - half), min(nx, mid + half)
    )
