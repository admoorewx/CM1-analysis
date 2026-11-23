"""Synthetic CM1 output files that mirror the real netCDF layout at a tiny size."""

from __future__ import annotations

import numpy as np
import pytest
from netCDF4 import Dataset

from cm1analysis import registry

NX, NY, NZ = 8, 6, 5
DX_KM = 0.5

SCALAR = ("time", "nk", "nj", "ni")
SURFACE = ("time", "nj", "ni")
DIMS = {
    "th": SCALAR,
    "thpert": SCALAR,
    "qv": SCALAR,
    "qvpert": SCALAR,
    "prs": SCALAR,
    "prspert": SCALAR,
    "winterp": SCALAR,
    "qc": SCALAR,
    "qr": SCALAR,
    "qi": SCALAR,
    "qs": SCALAR,
    "qg": SCALAR,
    "qhl": SCALAR,
    "cbudevp": SCALAR,
    "cbudmltg": SCALAR,
    "uinterp": SCALAR,
    "vinterp": SCALAR,
    "u": ("time", "nk", "nj", "nip1"),
    "v": ("time", "nk", "njp1", "ni"),
    "w": ("time", "nkp1", "nj", "ni"),
    "cref": SURFACE,
    "sws": SURFACE,
    "sus": SURFACE,
    "cpc": SURFACE,
    "u10": SURFACE,
    "v10": SURFACE,
    "t2": SURFACE,
    "zh": ("nk", "nj", "ni"),
}
UNITS = {
    "th": "K",
    "thpert": "K",
    "prs": "Pa",
    "prspert": "Pa",
    "t2": "K",
    "cref": "dBZ",
    "zh": "m",
    "cbudevp": "K/s",
    "cbudmltg": "K/s",
}
for _q in ("qv", "qvpert", "qc", "qr", "qi", "qs", "qg", "qhl"):
    UNITS[_q] = "kg/kg"

# (mean, spread) of the synthetic values, so derived fields are physically plausible
VALUES = {
    "th": (300.0, 1.0),
    "thpert": (-1.0, 2.0),
    "qv": (0.012, 0.002),
    "qvpert": (0.0, 0.001),
    "prs": (95000.0, 500.0),
    "prspert": (0.0, 200.0),
    "winterp": (0.0, 5.0),
    "cref": (30.0, 15.0),
    "sws": (20.0, 8.0),
    "sus": (10.0, 5.0),
    "cpc": (15.0, 5.0),
    "t2": (300.0, 3.0),
    "zh": (1000.0, 1.0),
    "cbudevp": (-0.005, 0.002),
    "cbudmltg": (-0.003, 0.001),
}
FULL_FIELDS = tuple(DIMS)
BASIC_FIELDS = ("th", "qv", "qr", "u", "v", "w", "cref", "sws", "zh")


def write_cm1_file(path, times=(7200.0,), fields=BASIC_FIELDS, x_km=None, seed=0):
    """Write a CM1-style file: dims ni/nj/nk (+1 staggered), coords xh/xf/yh/yf/z/zf in km."""
    x = np.arange(NX) * DX_KM + DX_KM / 2 if x_km is None else np.asarray(x_km, dtype=float)
    nx = x.size
    y = np.arange(NY) * DX_KM + DX_KM / 2
    zf = np.linspace(0.0, 2.0, NZ + 1)
    z = 0.5 * (zf[1:] + zf[:-1])
    rng = np.random.default_rng(seed)

    with Dataset(path, "w") as ds:
        sizes = {
            "ni": nx,
            "nj": NY,
            "nk": NZ,
            "nip1": nx + 1,
            "njp1": NY + 1,
            "nkp1": NZ + 1,
            "time": len(times),
            "one": 1,
        }
        for name, size in sizes.items():
            ds.createDimension(name, None if name == "time" else size)

        def coord(name, dim, values, units="km"):
            var = ds.createVariable(name, "f4", (dim,))
            var.units = units
            var[:] = values

        coord("xh", "ni", x)
        coord("xf", "nip1", np.append(x - DX_KM / 2, x[-1] + DX_KM / 2))
        coord("yh", "nj", y)
        coord("yf", "njp1", np.append(y - DX_KM / 2, y[-1] + DX_KM / 2))
        coord("z", "nk", z)
        coord("zf", "nkp1", zf)
        coord("time", "time", np.asarray(times), units="seconds")

        for name in fields:
            dims = DIMS[name]
            var = ds.createVariable(name, "f4", dims, fill_value=-999.0)
            var.units = UNITS.get(name, "m/s")
            var.long_name = f"synthetic {name}"
            mean, spread = VALUES.get(name, (0.0, 1.0))
            if name.startswith("q") and name not in ("qv", "qvpert"):
                mean, spread = 0.001, 0.0005
            data = mean + spread * rng.standard_normal(tuple(sizes[d] for d in dims))
            if name.startswith("q") and name not in ("qvpert",):
                data = np.abs(data)
            var[:] = data.astype("f4")
    return path


def write_run(directory, ntimes=3, interval_s=3600.0, fields=FULL_FIELDS, seed=0):
    """A run directory of one-time files cm1out_000001.nc ... like CM1 writes them."""
    directory.mkdir(parents=True, exist_ok=True)
    return [
        write_cm1_file(
            directory / f"cm1out_{i + 1:06d}.nc", times=(i * interval_s,), fields=fields, seed=seed + i
        )
        for i in range(ntimes)
    ]


@pytest.fixture
def cm1_path(tmp_path):
    return write_cm1_file(tmp_path / "cm1out_000001.nc")


@pytest.fixture
def clean_registry(monkeypatch):
    """An empty registry for the duration of one test."""
    monkeypatch.setattr(registry, "_REGISTRY", {})
    monkeypatch.setattr(registry, "_COMMON_OPTIONS", {})
    return registry
