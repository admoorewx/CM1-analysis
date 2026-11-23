"""Read-only access to CM1 netCDF output.

CM1 writes its grid as fixed coordinate names: ``xh``/``yh`` (scalar points),
``xf``/``yf`` (staggered u and v points), ``z``/``zf`` (nominal scalar and w
heights) and ``time``. Which *fields* a file holds depends on the output
namelist, so nothing beyond ``time``, ``xh`` and ``yh`` is required here.
Callers ask :meth:`CM1File.has` or :meth:`CM1File.missing` before reading.

Fields are returned as NumPy arrays with masked values filled with NaN.
Arrays keep the file's dtype (float32 for CM1) unless ``dtype`` is given, which
matters at full resolution: one 3-D field of a 1600 x 1600 x 88 grid is about
0.9 GB in float32.
"""

from __future__ import annotations

import glob
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

LENGTH_TO_METERS = {"km": 1000.0, "m": 1.0}
UNIFORM_TOLERANCE = 1e-3  # relative; float32 km coordinates vary by about 1e-4
DEFAULT_PATTERN = "cm1out_*.nc"

# Where a field sits on the C-grid, keyed by the coordinate variables of its
# horizontal and vertical dimensions.
_LOCATIONS = {
    ("z", "yh", "xh"): "scalar",
    ("z", "yh", "xf"): "u",
    ("z", "yf", "xh"): "v",
    ("zf", "yh", "xh"): "w",
    ("yh", "xh"): "surface",
}

Index = int | slice | None


class CM1File:
    """One CM1 output file.

    Use as a context manager so the netCDF handle is always closed::

        with CM1File("cm1out_000120.nc") as cm1:
            if cm1.has("cref"):
                cref = cm1.read("cref", time=0)
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.ds = Dataset(self.path, "r")
        try:
            self._load_grid()
        except Exception:
            self.ds.close()
            raise

    # ------------------------------------------------------------------ grid
    def _load_grid(self) -> None:
        for name in ("time", "xh", "yh"):
            if name not in self.ds.variables:
                raise ValueError(f"{self.path} is not CM1 output: no {name!r} coordinate")

        # Map each dimension to the CM1 coordinate that labels it (ni -> xh, ...).
        self._coord_of_dim: dict[str, str] = {}
        for name in ("time", "xh", "xf", "yh", "yf", "z", "zf"):
            var = self.ds.variables.get(name)
            if var is not None and var.ndim == 1:
                self._coord_of_dim[var.dimensions[0]] = name

        self.times = np.asarray(self.ds["time"][:], dtype=np.float64)
        self.x_km = _km(self.ds["xh"])
        self.y_km = _km(self.ds["yh"])
        self.xf_km = _km(self.ds["xf"]) if "xf" in self.ds.variables else None
        self.yf_km = _km(self.ds["yf"]) if "yf" in self.ds.variables else None
        self.z_km = self._nominal_heights("z")
        self.zf_km = self._nominal_heights("zf")
        self.dx = _uniform_spacing_m(self.x_km)
        self.dy = _uniform_spacing_m(self.y_km)

    def _nominal_heights(self, name: str) -> np.ndarray | None:
        var = self.ds.variables.get(name)
        if var is None or var.ndim != 1:
            return None
        return _km(var)

    @property
    def ntimes(self) -> int:
        return self.times.size

    @property
    def shape(self) -> tuple[int, int, int | None]:
        """(nz, ny, nx) of the scalar grid; nz is None for files without 3-D output."""
        nz = None if self.z_km is None else self.z_km.size
        return nz, self.y_km.size, self.x_km.size

    # ------------------------------------------------------------- variables
    @property
    def variables(self) -> frozenset[str]:
        return frozenset(self.ds.variables)

    def has(self, *names: str) -> bool:
        return all(name in self.ds.variables for name in names)

    def missing(self, names: Sequence[str]) -> list[str]:
        return [name for name in names if name not in self.ds.variables]

    def location(self, name: str) -> str:
        """Grid location of a field: 'scalar', 'u', 'v', 'w', 'surface' or 'other'."""
        dims = self._var(name).dimensions
        if dims and dims[0] == self._time_dim:
            dims = dims[1:]
        coords = tuple(self._coord_of_dim.get(dim, "?") for dim in dims)
        return _LOCATIONS.get(coords, "other")

    @property
    def hydrometeors(self) -> list[str]:
        """Hydrometeor mixing ratios: scalar-grid 'q*' fields in kg/kg other than vapor."""
        return sorted(
            name
            for name, var in self.ds.variables.items()
            if name.startswith("q")
            and not name.startswith("qv")
            and getattr(var, "units", "") == "kg/kg"
            and self.location(name) == "scalar"
        )

    def units(self, name: str) -> str:
        return getattr(self._var(name), "units", "")

    def long_name(self, name: str) -> str:
        return getattr(self._var(name), "long_name", name)

    # ------------------------------------------------------------------ read
    def read(
        self,
        name: str,
        time: int = 0,
        level: Index = None,
        y: Index = None,
        x: Index = None,
        dtype: np.typing.DTypeLike | None = None,
    ) -> np.ndarray:
        """Read one time of a field, optionally subset by level, row and column indices.

        ``level`` is rejected for 2-D (surface) fields. Integer indices drop
        that axis, slices keep it, as in NumPy.
        """
        var = self._var(name)
        dims = var.dimensions
        key: list[Index] = []
        if dims and dims[0] == self._time_dim:
            if not -self.ntimes <= time < self.ntimes:
                raise IndexError(f"time index {time} out of range for {self.ntimes} time(s)")
            key.append(time)
            dims = dims[1:]
        if len(dims) == 3:
            key.append(level)
        elif level is not None:
            raise ValueError(f"{name!r} has no vertical dimension; do not pass level")
        if len(dims) not in (2, 3):
            raise ValueError(f"{name!r} has dimensions {var.dimensions}; expected a 2-D or 3-D field")
        key += [y, x]
        data = var[tuple(slice(None) if k is None else k for k in key)]
        data = np.ma.filled(np.ma.asarray(data).astype(dtype or var.dtype, copy=False), np.nan)
        return np.asarray(data)

    # --------------------------------------------------------------- lookup
    def level_index(self, height_km: float, staggered: bool = False) -> int:
        """Index of the nominal level nearest height_km (w levels if staggered)."""
        heights = self.zf_km if staggered else self.z_km
        if heights is None:
            raise ValueError(f"{self.path.name} has no vertical coordinate")
        # Accept anything inside the model domain (bottom to top w level), not just
        # between scalar levels, so 0.1 km maps to the lowest level.
        bottom, top = (self.zf_km[0], self.zf_km[-1]) if self.zf_km is not None else heights[[0, -1]]
        if not bottom - 1e-6 <= height_km <= top + 1e-6:
            raise ValueError(f"{height_km:g} km is outside the domain ({bottom:g}-{top:g} km)")
        return int(np.argmin(np.abs(heights - height_km)))

    def time_index(self, seconds: float) -> int:
        """Index of the output time nearest the given simulation time in seconds."""
        return int(np.argmin(np.abs(self.times - seconds)))

    def summary(self) -> dict:
        """Grid and contents, for ``cm1plot inventory`` and logging."""
        nz, ny, nx = self.shape
        return {
            "path": str(self.path),
            "times_s": self.times.tolist(),
            "shape_zyx": (nz, ny, nx),
            "dx_m": self.dx,
            "dy_m": self.dy,
            "ztop_km": None if self.zf_km is None else float(self.zf_km[-1]),
            "hydrometeors": self.hydrometeors,
            "fields": {
                name: self.location(name)
                for name, var in self.ds.variables.items()
                if var.ndim >= 2 and name not in ("zh",)
            },
        }

    # -------------------------------------------------------------- helpers
    @property
    def _time_dim(self) -> str:
        return self.ds["time"].dimensions[0]

    def _var(self, name: str):
        try:
            return self.ds.variables[name]
        except KeyError:
            raise KeyError(f"{self.path.name} has no variable {name!r}") from None

    def close(self) -> None:
        self.ds.close()

    def __enter__(self) -> CM1File:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        nz, ny, nx = self.shape
        return f"CM1File({self.path.name!r}, nz={nz}, ny={ny}, nx={nx}, ntimes={self.ntimes})"


# ---------------------------------------------------------------- run level
def find_output_files(run_dir: str | Path, pattern: str = DEFAULT_PATTERN) -> list[Path]:
    """Sorted CM1 output files in a run directory (CM1 zero-pads file numbers)."""
    files = sorted(Path(p) for p in glob.glob(str(Path(run_dir) / pattern)))
    if not files:
        raise FileNotFoundError(f"no files matching {pattern!r} in {run_dir}")
    return files


def iter_times(paths: Sequence[str | Path]) -> Iterator[tuple[CM1File, int, float]]:
    """Yield (file, time index, time in s) across files, whether each holds one time or many.

    The file is open only while its times are being yielded.
    """
    for path in paths:
        with CM1File(path) as cm1:
            for index, seconds in enumerate(cm1.times):
                yield cm1, index, float(seconds)


# ------------------------------------------------------------------ private
def _km(var) -> np.ndarray:
    units = getattr(var, "units", "km")
    if units not in LENGTH_TO_METERS:
        raise ValueError(f"coordinate {var.name!r} has unsupported units {units!r}")
    return np.asarray(var[:], dtype=np.float64) * LENGTH_TO_METERS[units] / 1000.0


def _uniform_spacing_m(coord_km: np.ndarray) -> float | None:
    """Grid spacing in m, or None if the axis is stretched or has a single point."""
    if coord_km.size < 2:
        return None
    steps = np.diff(coord_km)
    mean = (coord_km[-1] - coord_km[0]) / (coord_km.size - 1)
    if np.max(np.abs(steps / mean - 1.0)) > UNIFORM_TOLERANCE:
        return None
    return round(float(mean) * 1000.0, 3)  # rounding removes float32 noise from km coordinates
