"""Partition reader, writer and command line on a small file laid out like real CM1 output."""

import numpy as np
import pytest
from netCDF4 import Dataset

from cm1analysis.io import CM1File
from cm1analysis.wind import WindLevel
from cm1analysis.wind.__main__ import main as partition_main

NX, NY, NK = 120, 100, 3


def make_cm1_file(path, x_units="km", terrain=False, stretched=False):
    scale = 1.0 if x_units == "km" else 1000.0
    with Dataset(path, "w", format="NETCDF4_CLASSIC") as ds:
        for name, size in {
            "ni": NX,
            "nj": NY,
            "nk": NK,
            "nip1": NX + 1,
            "njp1": NY + 1,
            "time": 2,
        }.items():
            ds.createDimension(name, size)
        ds.nx = NX
        ds.wbc = 2

        def coord(name, dim, values, units):
            var = ds.createVariable(name, "f4", (dim,))
            var[:] = values
            var.units = units

        xf = np.arange(NX + 1, dtype=np.float64)
        if stretched:
            xf = xf**1.2
        coord("xf", "nip1", xf * scale, x_units)
        coord("xh", "ni", 0.5 * (xf[1:] + xf[:-1]) * scale, x_units)
        coord("yf", "njp1", np.arange(NY + 1.0), "km")
        coord("yh", "nj", np.arange(NY) + 0.5, "km")
        coord("z", "nk", np.array([0.025, 0.075, 0.125]), "km")
        coord("time", "time", np.array([300.0, 600.0]), "seconds")

        zh = ds.createVariable("zh", "f4", ("nk", "nj", "ni"))
        zh.units = "m"
        heights = np.array([25.0, 75.0, 125.0])[:, None, None] * np.ones((NK, NY, NX))
        if terrain:
            heights[:, 10, 10] += 100.0
        zh[:] = heights

        rng = np.random.default_rng(3)
        u = ds.createVariable("u", "f4", ("time", "nk", "nj", "nip1"))
        v = ds.createVariable("v", "f4", ("time", "nk", "njp1", "ni"))
        cref = ds.createVariable("cref", "f4", ("time", "nj", "ni"))
        u[:] = 5.0 + rng.normal(0, 3, (2, NK, NY, NX + 1))
        v[:] = rng.normal(0, 3, (2, NK, NY + 1, NX))
        reflectivity = np.zeros((2, NY, NX))
        reflectivity[:, 40:60, 50:70] = 30.0
        cref[:] = reflectivity


@pytest.fixture
def cm1_path(tmp_path):
    path = tmp_path / "cm1out_000007.nc"
    make_cm1_file(path)
    return path


def run(cm1_path, tmp_path, *extra):
    return partition_main([str(cm1_path), "--output", str(tmp_path / "out"), *extra])


def test_reader_gets_grid_from_file(cm1_path):
    with CM1File(cm1_path) as cm1:
        source = WindLevel(cm1, 0)
        assert (source.nx, source.ny) == (NX, NY)
        assert source.dx == 1000.0 and source.dy == 1000.0
        assert source.level_height_m == pytest.approx(25.0)
        u, v = source.read_wind(1)
    assert u.shape == (NY, NX + 1) and v.shape == (NY + 1, NX)
    assert u.dtype == np.float64 and v.dtype == np.float64


def test_coordinates_in_meters_are_converted(tmp_path):
    path = tmp_path / "cm1out_000001.nc"
    make_cm1_file(path, x_units="m")
    with CM1File(path) as cm1:
        assert WindLevel(cm1, 0).dx == 1000.0


def test_stretched_grid_fails_loudly(tmp_path):
    path = tmp_path / "cm1out_000001.nc"
    make_cm1_file(path, stretched=True)
    with CM1File(path) as cm1, pytest.raises(ValueError, match="not uniform"):
        WindLevel(cm1, 0)


def test_terrain_fails_loudly(tmp_path):
    path = tmp_path / "cm1out_000001.nc"
    make_cm1_file(path, terrain=True)
    with CM1File(path) as cm1, pytest.raises(ValueError, match="constant-height"):
        WindLevel(cm1, 0)


def test_level_out_of_range_fails_loudly(cm1_path):
    with CM1File(cm1_path) as cm1, pytest.raises(ValueError, match="outside"):
        WindLevel(cm1, NK)


def test_height_selects_nearest_level(cm1_path):
    with CM1File(cm1_path) as cm1:
        assert WindLevel(cm1, height_km=0.08).level == 1
        with pytest.raises(ValueError, match="outside"):
            WindLevel(cm1, height_km=5.0)


def test_missing_winds_fail_loudly(tmp_path):
    path = tmp_path / "cm1out_000001.nc"
    make_cm1_file(path)
    with Dataset(path, "a") as ds:
        ds.renameVariable("v", "vv")
    with CM1File(path) as cm1, pytest.raises(ValueError, match="no native C-grid variable 'v'"):
        WindLevel(cm1)


def test_output_contents(cm1_path, tmp_path):
    assert run(cm1_path, tmp_path) == 0
    target = tmp_path / "out" / "wind_decomp_000007.nc"
    assert not (tmp_path / "out" / "wind_decomp_000007.nc.tmp").exists()
    with Dataset(target) as ds, CM1File(cm1_path) as cm1:
        source = WindLevel(cm1, 0)
        assert ds["psi"].shape == (2, NY + 1, NX + 1)
        assert ds["chi"].shape == (2, NY, NX)
        assert ds["u_h"].shape == (2, NY, NX + 1)
        assert ds["v_h"].shape == (2, NY + 1, NX)
        assert ds["psi"].units == "m2 s-1"
        assert ds["u_psi"].units == "m s-1"
        assert ds["delta"].units == "s-1"
        assert ds["psi"].datatype == "f8"
        assert ds.wbc == 2 and ds.nx == NX  # Copied from the source.
        assert ds.variant == "A"
        assert "u_rot" not in ds.variables

        u, v = source.read_wind(1)
        total_u = ds["u_psi"][1] + ds["u_chi"][1] + ds["u_h"][1]
        total_v = ds["v_psi"][1] + ds["v_chi"][1] + ds["v_h"][1]
        assert np.allclose(total_u, u, atol=1e-11)
        assert np.allclose(total_v, v, atol=1e-11)

        zeta = ds["zeta"][1]
        assert zeta.mask[0].all() and zeta.mask[-1].all()
        assert zeta.mask[:, 0].all() and zeta.mask[:, -1].all()
        assert not zeta.mask[1:-1, 1:-1].any()

        diag = ds["diagnostics"]
        assert diag["validation_failures"][:].tolist() == [0, 0]
        assert diag["crop_change_vpsi"].shape == (2, 3)
        assert diag["crop_fraction"][:].tolist() == [0.1, 0.2, 0.3]
        # Threshold reached at i 50-69 and j 40-59, padded by 25 cells then clamped.
        assert diag["roi_i0"][0] == 25
        assert diag["roi_i1"][0] == 95
        assert diag["roi_j0"][0] == 20
        assert diag["roi_j1"][0] == 80


def test_existing_output_is_not_overwritten(cm1_path, tmp_path):
    assert run(cm1_path, tmp_path) == 0
    with pytest.raises(SystemExit, match="exists"):
        run(cm1_path, tmp_path)
    assert run(cm1_path, tmp_path, "--overwrite") == 0


def test_time_range_and_level(cm1_path, tmp_path):
    assert run(cm1_path, tmp_path, "--time-range", "500", "700", "--level", "2") == 0
    with Dataset(tmp_path / "out" / "wind_decomp_000007.nc") as ds:
        assert ds["time"][:].tolist() == [600.0]
        assert ds.level_index == 2
        assert ds.level_height_m == pytest.approx(125.0)


@pytest.mark.parametrize("variant", ["B", "C"])
def test_variant_writes_labeled_winds_and_keeps_harmonic(cm1_path, tmp_path, variant):
    assert run(cm1_path, tmp_path, "--variant", variant) == 0
    with Dataset(tmp_path / "out" / "wind_decomp_000007.nc") as ds:
        assert ds.variant == variant
        assert "u_h" in ds.variables
        total = ds["u_rot"][0] + ds["u_div"][0]
        expected = ds["u_psi"][0] + ds["u_chi"][0] + ds["u_h"][0]
        assert np.allclose(total, expected, atol=1e-11)


def test_manual_roi(cm1_path, tmp_path):
    assert run(cm1_path, tmp_path, "--roi", "40", "80", "30", "70") == 0
    with Dataset(tmp_path / "out" / "wind_decomp_000007.nc") as ds:
        assert ds.roi_source == "manual"
        assert ds["diagnostics"]["roi_i0"][0] == 40
        assert ds["diagnostics"]["roi_i1"][0] == 80
