"""Solver inputs read from a CM1 file, with the checks the decomposition relies on.

The shared :class:`cm1analysis.io.CM1File` opens the file and finds its time
axis and hydrometeors. :class:`SolverSource` adds what the Poisson problems
need beyond that, and refuses files that break their assumptions:

- the fields u, v, w, th, thpert, prs, prspert, qv and qvpert are present, 4-D
  and ordered (time, z, y, x), with u, v and w on a C-grid with staggered w;
- the horizontal spacing is uniform (the vertical grid may be stretched);
- the terrain is flat: a 3-D height field (CM1's ``zh``), when present, is level
  at its nominal height on the bottom and top levels.

Dimension and coordinate names are taken from the file. Fields are returned in
float64 on a crop of the domain.
"""

from __future__ import annotations

import logging

import numpy as np

from ..io import LENGTH_TO_METERS, CM1File
from . import forcing, vforce
from .grid import Box, Grid

log = logging.getLogger(__name__)

UNIFORM_TOLERANCE = 1e-3  # float32 km coordinates vary by about 1e-4.
FLAT_TOLERANCE_M = 0.05
REQUIRED = ("u", "v", "w", "th", "thpert", "prs", "prspert", "qv", "qvpert")


def _single_coordinate(ds, dim):
    matches = [name for name, var in ds.variables.items() if var.dimensions == (dim,)]
    if len(matches) != 1:
        raise ValueError(f"expected one coordinate for dimension {dim!r}, found {matches}")
    return ds.variables[matches[0]]


def _meters(coord):
    units = getattr(coord, "units", None)
    if units not in LENGTH_TO_METERS:
        raise ValueError(f"coordinate {coord.name!r} has unsupported units {units!r}")
    return np.asarray(coord[:], dtype=np.float64) * LENGTH_TO_METERS[units]


def _uniform_spacing(coord):
    values = _meters(coord)
    steps = np.diff(values)
    mean = (values[-1] - values[0]) / (values.size - 1)
    if np.max(np.abs(steps / mean - 1.0)) > UNIFORM_TOLERANCE:
        raise ValueError(
            f"coordinate {coord.name!r} is not uniform: spacing runs from {steps.min()} to {steps.max()} m"
        )
    return round(float(mean), 3)  # Removes float32 rounding from km coordinates.


class SolverSource:
    """A validated view of one open :class:`CM1File` for the pressure decomposition.

    The file stays owned by the caller; this class never closes it.
    """

    def __init__(self, cm1: CM1File):
        self.cm1 = cm1
        self.path = cm1.path
        self.ds = cm1.ds
        self._load_grid()

    def _load_grid(self):
        missing = self.cm1.missing(REQUIRED)
        if missing:
            raise ValueError(f"{self.path} is missing {missing}")
        u, v, w, th = (self.ds[n] for n in ("u", "v", "w", "th"))
        for var in (u, v, w, th):
            if var.ndim != 4:
                raise ValueError(f"expected 4-D {var.name}, got {var.dimensions}")
        time_dim, self.z_dim, self.y_dim, self.x_dim = th.dimensions
        self.x_face_dim, self.y_face_dim, self.zf_dim = u.dimensions[3], v.dimensions[2], w.dimensions[1]
        expected = {
            "u": (time_dim, self.z_dim, self.y_dim, self.x_face_dim),
            "v": (time_dim, self.z_dim, self.y_face_dim, self.x_dim),
            "w": (time_dim, self.zf_dim, self.y_dim, self.x_dim),
        }
        for name, dims in expected.items():
            if self.ds[name].dimensions != dims:
                raise ValueError(f"{name} has dimensions {self.ds[name].dimensions}, expected {dims}")
        for name in REQUIRED[3:]:
            if self.ds[name].dimensions != th.dimensions:
                raise ValueError(f"{name} is not on the scalar grid: {self.ds[name].dimensions}")

        self.x_center = _single_coordinate(self.ds, self.x_dim)
        self.y_center = _single_coordinate(self.ds, self.y_dim)
        self.nx, self.ny = self.x_center.size, self.y_center.size
        if (u.shape[3], v.shape[2], w.shape[1]) != (self.nx + 1, self.ny + 1, th.shape[1] + 1):
            raise ValueError("u, v and w do not match a C-grid with staggered w")
        self.dx = _uniform_spacing(_single_coordinate(self.ds, self.x_face_dim))
        self.dy = _uniform_spacing(_single_coordinate(self.ds, self.y_face_dim))
        self.x_face = _single_coordinate(self.ds, self.x_face_dim)
        self.y_face = _single_coordinate(self.ds, self.y_face_dim)
        self.z_coord = _single_coordinate(self.ds, self.z_dim)
        self.zf_coord = _single_coordinate(self.ds, self.zf_dim)
        zf, zc = _meters(self.zf_coord), _meters(self.z_coord)
        self.grid = Grid(self.dx, self.dy, zf - zf[0], zc - zf[0])
        self._check_flat_terrain(zc)

        self.time_var = _single_coordinate(self.ds, time_dim)
        self.times = np.asarray(self.time_var[:], dtype=np.float64)
        self.hydrometeors = self.cm1.hydrometeors

    def _check_flat_terrain(self, zc):
        """Fail over terrain: the 3-D height of the bottom and top levels must be flat."""
        dims = (self.z_dim, self.y_dim, self.x_dim)
        for var in self.ds.variables.values():
            if var.dimensions == dims and getattr(var, "units", "") in LENGTH_TO_METERS:
                to_m = LENGTH_TO_METERS[var.units]
                for k in (0, len(zc) - 1):
                    height = np.asarray(var[k], dtype=np.float64) * to_m
                    if (
                        height.max() - height.min() > FLAT_TOLERANCE_M
                        or abs(height.mean() - zc[k]) > FLAT_TOLERANCE_M
                    ):
                        raise ValueError(
                            f"level {k} is not flat at its nominal height according to {var.name!r}"
                        )
                return
        log.warning("%s has no 3-D height field; flat terrain not checked", self.path.name)

    def read_cref(self, time_index, box=None):
        if not self.cm1.has("cref"):
            return None
        cells = (slice(None), slice(None)) if box is None else box.cells
        return np.ma.filled(self.ds["cref"][(time_index,) + cells].astype(np.float64), np.nan)

    def roi_from_limits(self, xmin_km, xmax_km, ymin_km, ymax_km):
        """An ROI from limits in km (the roi_km option)."""
        x_km, y_km = _meters(self.x_center) / 1000.0, _meters(self.y_center) / 1000.0
        roi = Box(
            int(np.searchsorted(x_km, xmin_km)),
            int(np.searchsorted(x_km, xmax_km, side="right")),
            int(np.searchsorted(y_km, ymin_km)),
            int(np.searchsorted(y_km, ymax_km, side="right")),
        )
        if roi.nx <= 0 or roi.ny <= 0:
            raise ValueError(f"ROI limits select no cells: {roi}")
        return roi

    def nearest_row(self, y_km):
        """Index of the cell-center row closest to y_km (the section_y_km option)."""
        y = _meters(self.y_center) / 1000.0
        if not y[0] <= y_km <= y[-1]:
            raise ValueError(f"section y {y_km:g} km is outside the domain ({y[0]:g}-{y[-1]:g} km)")
        return int(np.argmin(np.abs(y - y_km)))

    def _read(self, name, time_index, jj, ii):
        data = self.ds[name][time_index, :, jj, ii]
        return np.ma.filled(data.astype(np.float64), np.nan)

    def read(self, time_index, crop):
        """Read one time inside crop. Returns (Inputs, w on faces, base-state spreads)."""
        jj, ii = crop.cells
        u = self._read("u", time_index, jj, slice(crop.i0, crop.i1 + 1))
        uc = 0.5 * (u[..., 1:] + u[..., :-1])
        del u
        v = self._read("v", time_index, slice(crop.j0, crop.j1 + 1), ii)
        vc = 0.5 * (v[:, 1:] + v[:, :-1])
        del v
        w = self._read("w", time_index, jj, ii)
        wc = self.grid.faces_to_centers(w)

        spreads = {}
        perturbations = {}
        totals = {}
        profiles = {}
        for total, pert, key in (("th", "thpert", "th0"), ("qv", "qvpert", "qv0"), ("prs", "prspert", "p0")):
            totals[total] = self._read(total, time_index, jj, ii)
            perturbations[pert] = self._read(pert, time_index, jj, ii)
            recovered = totals[total] - perturbations[pert]
            profiles[key] = recovered.mean(axis=(1, 2))
            spreads[f"base_{key}_spread"] = float(np.max(np.abs(recovered - profiles[key][:, None, None])))
            del recovered
        rho_model = forcing.dry_density(totals["prs"], totals["th"], totals["qv"])
        del totals

        qw = np.zeros_like(uc)
        for name in self.hydrometeors:
            qw += self._read(name, time_index, jj, ii)

        fields = {"uc": uc, "vc": vc, "wc": wc, "qw": qw, "rho_model": rho_model, **perturbations}
        bad = [name for name, a in fields.items() if not np.all(np.isfinite(a))]
        if bad:
            raise ValueError(f"non-finite values in {bad} at time index {time_index}")
        base = forcing.BaseState(self.grid, profiles["th0"], profiles["qv0"], profiles["p0"])
        inputs = vforce.Inputs(
            self.grid,
            uc,
            vc,
            wc,
            perturbations["thpert"],
            perturbations["qvpert"],
            qw,
            base,
            prspert=perturbations["prspert"],
            rho_model=rho_model,
        )
        return inputs, w, spreads
