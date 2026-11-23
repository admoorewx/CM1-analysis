"""One level of CM1's native C-grid winds, checked for the partition.

Built on the shared :class:`cm1analysis.io.CM1File`. The partition needs more
than the plotting code does, so this module checks it explicitly and fails
loudly otherwise:

- ``u`` and ``v`` are 4-D ``(time, z, y, x)`` fields on the staggered C-grid
  (``u`` on ``(yh, xf)``, ``v`` on ``(yf, xh)``);
- the face coordinates ``xf`` and ``yf`` are uniform (a stretched grid would
  need different operators);
- the chosen level is a constant-height surface (no terrain), checked against
  a 3-D height field such as ``zh`` when the file has one.
"""

from __future__ import annotations

import logging

import numpy as np

from ..io import LENGTH_TO_METERS, UNIFORM_TOLERANCE, CM1File
from .diagnostics import ROI

log = logging.getLogger(__name__)

FLAT_TOLERANCE_M = 1e-3


def _spacing_meters(coord) -> float:
    """Uniform spacing of a 1-D coordinate in m, or ValueError if it is stretched."""
    units = getattr(coord, "units", None)
    if units not in LENGTH_TO_METERS:
        raise ValueError(f"coordinate {coord.name!r} has unsupported units {units!r}")
    values = np.asarray(coord[:], dtype=np.float64) * LENGTH_TO_METERS[units]
    if values.size < 2:
        raise ValueError(f"coordinate {coord.name!r} needs at least two points")
    steps = np.diff(values)
    mean = (values[-1] - values[0]) / (values.size - 1)
    if np.max(np.abs(steps / mean - 1.0)) > UNIFORM_TOLERANCE:
        raise ValueError(
            f"coordinate {coord.name!r} is not uniform: spacing runs from {steps.min()} to {steps.max()} m"
        )
    return round(float(mean), 3)  # removes float32 rounding from km coordinates


class WindLevel:
    """The native winds of one CM1 file on one level, with the grid the partition writer needs.

    ``level`` is a model level index; if ``height_km`` is given instead, the
    nearest nominal scalar level is used (terrain-free runs only, which is
    checked). The netCDF coordinate variables are kept as ``x_center``,
    ``x_face``, ``y_center``, ``y_face`` and ``time_var`` so the output file
    can copy their names, dimensions and units.
    """

    def __init__(self, cm1: CM1File, level: int = 0, height_km: float | None = None):
        self.cm1 = cm1
        self.path = cm1.path
        self.ds = cm1.ds
        for name in ("u", "v"):
            if not cm1.has(name):
                raise ValueError(f"{self.path} has no native C-grid variable {name!r}")
        u, v = self.ds["u"], self.ds["v"]
        if u.ndim != 4 or v.ndim != 4:
            raise ValueError(f"expected 4-D u and v, got {u.dimensions} and {v.dimensions}")
        if cm1.location("u") != "u" or cm1.location("v") != "v":
            raise ValueError(
                f"u {u.dimensions} and v {v.dimensions} are not on the C-grid "
                "(expected u on (z, yh, xf) and v on (z, yf, xh))"
            )
        _, self.z_dim, self.y_center_dim, self.x_face_dim = u.dimensions
        _, _, self.y_face_dim, self.x_center_dim = v.dimensions

        self.x_center, self.x_face = self.ds["xh"], self.ds["xf"]
        self.y_center, self.y_face = self.ds["yh"], self.ds["yf"]
        self.nx, self.ny = self.x_center.size, self.y_center.size
        if (u.shape[2:], v.shape[2:]) != ((self.ny, self.nx + 1), (self.ny + 1, self.nx)):
            raise ValueError(
                f"u {u.shape} and v {v.shape} do not match the C-grid for nx={self.nx}, ny={self.ny}"
            )
        self.dx = _spacing_meters(self.x_face)
        self.dy = _spacing_meters(self.y_face)

        self.time_var = self.ds["time"]
        self.times = cm1.times

        nk = u.shape[1]
        if height_km is not None:
            level = cm1.level_index(height_km)  # levels are flat (checked below): height above ground
        if not 0 <= level < nk:
            raise ValueError(f"level {level} is outside 0..{nk - 1}")
        self.level = int(level)
        self.level_height_m = float(cm1.z_km[self.level] * 1000.0)
        self._check_flat_level()

    def _check_flat_level(self) -> None:
        """Fail if the level is not a constant-height surface, as over terrain."""
        dims = (self.z_dim, self.y_center_dim, self.x_center_dim)
        for var in self.ds.variables.values():
            if var.dimensions == dims and getattr(var, "units", "") in LENGTH_TO_METERS:
                height = np.asarray(var[self.level], dtype=np.float64)
                spread = float(height.max() - height.min()) * LENGTH_TO_METERS[var.units]
                if spread > FLAT_TOLERANCE_M:
                    raise ValueError(
                        f"level {self.level} is not a constant-height surface according to {var.name!r}"
                    )
                return
        log.warning("%s has no 3-D height field; flat level not checked", self.path.name)

    def read_wind(self, time_index: int) -> tuple[np.ndarray, np.ndarray]:
        """(u, v) on the C-grid faces at this level, in float64 with NaN for missing values."""
        u = self.cm1.read("u", time_index, level=self.level, dtype=np.float64)
        v = self.cm1.read("v", time_index, level=self.level, dtype=np.float64)
        return u, v

    def read_cref(self, time_index: int) -> np.ndarray | None:
        """Composite reflectivity, or None if the file has none."""
        if not self.cm1.has("cref"):
            return None
        return self.cm1.read("cref", time_index, dtype=np.float64)

    def roi_from_limits(self, xmin_km: float, xmax_km: float, ymin_km: float, ymax_km: float) -> ROI:
        """Build an ROI from limits in km on the scalar grid, for a fixed region of interest."""
        x_km, y_km = self.cm1.x_km, self.cm1.y_km
        roi = ROI(
            int(np.searchsorted(x_km, xmin_km)),
            int(np.searchsorted(x_km, xmax_km, side="right")),
            int(np.searchsorted(y_km, ymin_km)),
            int(np.searchsorted(y_km, ymax_km, side="right")),
        )
        if roi.i1 <= roi.i0 or roi.j1 <= roi.j0:
            raise ValueError(f"ROI {[xmin_km, xmax_km, ymin_km, ymax_km]} km selects no cells: {roi}")
        return roi
