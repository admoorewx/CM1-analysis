"""Synthetic CM1 output files that mirror the real netCDF layout at a tiny size."""

from __future__ import annotations

import numpy as np
import pytest
from netCDF4 import Dataset

from cm1analysis import registry

NX, NY, NZ = 8, 6, 5
DX_KM = 0.5


def write_cm1_file(
    path,
    times=(7200.0,),
    fields=("th", "qv", "qr", "u", "v", "w", "cref", "sws", "zh"),
    x_km=None,
):
    """Write a CM1-style file: dims ni/nj/nk (+1 staggered), coords xh/xf/yh/yf/z/zf in km."""
    x = np.arange(NX) * DX_KM + DX_KM / 2 if x_km is None else np.asarray(x_km, dtype=float)
    nx = x.size
    y = np.arange(NY) * DX_KM + DX_KM / 2
    zf = np.linspace(0.0, 2.0, NZ + 1)
    z = 0.5 * (zf[1:] + zf[:-1])
    rng = np.random.default_rng(0)

    with Dataset(path, "w") as ds:
        for name, size in (("ni", nx), ("nj", NY), ("nk", NZ), ("nip1", nx + 1), ("njp1", NY + 1),
                           ("nkp1", NZ + 1), ("time", None), ("one", 1)):  # fmt: skip
            ds.createDimension(name, size)

        def coord(name, dim, values, units="km"):
            var = ds.createVariable(name, "f4", (dim,))
            var.units = units
            var[:] = values

        xf = np.append(x - DX_KM / 2, x[-1] + DX_KM / 2)
        coord("xh", "ni", x)
        coord("xf", "nip1", xf)
        coord("yh", "nj", y)
        coord("yf", "njp1", np.append(y - DX_KM / 2, y[-1] + DX_KM / 2))
        coord("z", "nk", z)
        coord("zf", "nkp1", zf)
        coord("time", "time", np.asarray(times), units="seconds")
        nt = len(times)

        dims = {
            "th": ("time", "nk", "nj", "ni"), "qv": ("time", "nk", "nj", "ni"),
            "qr": ("time", "nk", "nj", "ni"), "u": ("time", "nk", "nj", "nip1"),
            "v": ("time", "nk", "njp1", "ni"), "w": ("time", "nkp1", "nj", "ni"),
            "cref": ("time", "nj", "ni"), "sws": ("time", "nj", "ni"), "zh": ("nk", "nj", "ni"),
        }  # fmt: skip
        units = {"th": "K", "qv": "kg/kg", "qr": "kg/kg", "cref": "dBZ", "zh": "m"}
        sizes = {"ni": nx, "nj": NY, "nk": NZ, "nip1": nx + 1, "njp1": NY + 1, "nkp1": NZ + 1, "time": nt}
        for name in fields:
            var = ds.createVariable(name, "f4", dims[name], fill_value=-999.0)
            var.units = units.get(name, "m/s")
            var.long_name = f"synthetic {name}"
            var[:] = rng.normal(size=tuple(sizes[d] for d in dims[name])).astype("f4")
    return path


@pytest.fixture
def cm1_path(tmp_path):
    return write_cm1_file(tmp_path / "cm1out_000001.nc")


@pytest.fixture
def clean_registry(monkeypatch):
    """An empty registry for the duration of one test."""
    monkeypatch.setattr(registry, "_REGISTRY", {})
    return registry
