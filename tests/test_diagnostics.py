import numpy as np
import pytest
from conftest import NX, NY, NZ, write_cm1_file
from netCDF4 import Dataset

from cm1analysis import diagnostics
from cm1analysis.io import CM1File


def set_fields(path, **values):
    with Dataset(path, "a") as ds:
        for name, value in values.items():
            ds[name][:] = value


def test_buoyancy_uses_cm1_base_state_and_all_hydrometeors(tmp_path):
    fields = ("th", "thpert", "qv", "qvpert", "qc", "qr", "qi", "qs", "qg", "qhl")
    path = write_cm1_file(tmp_path / "b.nc", fields=fields)
    set_fields(
        path,
        th=299.0,
        thpert=-3.0,
        qv=0.014,
        qvpert=0.002,
        qc=0.001,
        qr=0.002,
        qi=0.0,
        qs=0.0005,
        qg=0.0005,
        qhl=0.001,
    )
    with CM1File(path) as cm1:
        b = diagnostics.buoyancy(cm1, 0, level=1)
    theta0 = 299.0 - (-3.0)
    expected = 9.81 * (-3.0 / theta0 + 0.61 * 0.002 - (0.001 + 0.002 + 0.0005 + 0.0005 + 0.001))
    assert b.shape == (NY, NX)
    np.testing.assert_allclose(b, expected, rtol=1e-5)  # includes hail, which the legacy code dropped


def test_buoyancy_dry_file(tmp_path):
    path = write_cm1_file(tmp_path / "dry.nc", fields=("th", "thpert"))
    set_fields(path, th=300.0, thpert=3.0)
    with CM1File(path) as cm1:
        np.testing.assert_allclose(diagnostics.buoyancy(cm1, 0, 0), 9.81 * 3.0 / 297.0, rtol=1e-6)


def test_low_level_wind_prefers_10m(tmp_path):
    path = write_cm1_file(tmp_path / "w.nc", fields=("u10", "v10", "uinterp", "vinterp"))
    set_fields(path, u10=3.0, v10=4.0, uinterp=1.0, vinterp=1.0)
    with CM1File(path) as cm1:
        wind = diagnostics.low_level_wind(cm1, 0)
    assert wind.label == "10-m"
    np.testing.assert_allclose(wind.speed, 5.0)


def test_low_level_wind_falls_back_to_lowest_level(tmp_path):
    path = write_cm1_file(tmp_path / "w.nc", fields=("uinterp", "vinterp"))
    with CM1File(path) as cm1:
        wind = diagnostics.low_level_wind(cm1, 0)
        assert wind.label.startswith("lowest-level (200 m")
        np.testing.assert_array_equal(wind.u, cm1.read("uinterp", level=0))


def test_low_level_wind_from_staggered_components(tmp_path):
    path = write_cm1_file(tmp_path / "w.nc", fields=("u", "v"))
    with Dataset(path, "a") as ds:
        ds["u"][:] = np.broadcast_to(np.arange(NX + 1, dtype="f4"), ds["u"].shape)
        ds["v"][:] = 2.0
    with CM1File(path) as cm1:
        wind = diagnostics.low_level_wind(cm1, 0)
        sub = diagnostics.low_level_wind(cm1, 0, window=(slice(1, 4), slice(2, 5)))
    np.testing.assert_allclose(wind.u[0], np.arange(NX) + 0.5)  # face average onto scalar points
    assert wind.v.shape == (NY, NX)
    np.testing.assert_allclose(sub.u[0], np.arange(2, 5) + 0.5)
    assert sub.u.shape == sub.v.shape == (3, 3)


def test_missing_recipe_names_alternatives(tmp_path):
    path = write_cm1_file(tmp_path / "x.nc", fields=("cref",))
    with CM1File(path) as cm1:
        assert diagnostics.recipe(cm1, "low_level_wind") is None
        with pytest.raises(KeyError, match="u10\\+v10 or uinterp\\+vinterp or u\\+v"):
            diagnostics.low_level_wind(cm1, 0)


def test_surface_pressure_falls_back_to_lowest_level(tmp_path):
    path = write_cm1_file(tmp_path / "p.nc", fields=("prs",))
    with CM1File(path) as cm1:
        p, which = diagnostics.surface_pressure(cm1, 0)
        np.testing.assert_allclose(p, cm1.read("prs", level=0) / 100.0)
    assert which == "Lowest-level"


def test_column_reduce_matches_full_read(tmp_path):
    path = write_cm1_file(tmp_path / "q.nc", fields=("qr", "w"))
    with CM1File(path) as cm1:
        full = cm1.read("qr")
        np.testing.assert_array_equal(diagnostics.column_reduce(cm1, "qr", 0), full.max(axis=0))
        np.testing.assert_array_equal(diagnostics.column_reduce(cm1, "qr", 0, np.fmin), full.min(axis=0))
        # z = 0.2, 0.6, 1.0, ... km: top_km=0.7 keeps the lowest two levels
        np.testing.assert_array_equal(
            diagnostics.column_reduce(cm1, "qr", 0, top_km=0.7), full[:2].max(axis=0)
        )
        # w levels are zf = 0, 0.4, 0.8, ...: top_km=0.5 keeps two
        w = cm1.read("w")
        np.testing.assert_array_equal(
            diagnostics.column_reduce(cm1, "w", 0, np.fmin, top_km=0.5), w[:2].min(axis=0)
        )
        assert diagnostics.column_reduce(cm1, "qr", 0, window=(slice(0, 2), slice(0, 3))).shape == (2, 3)
    assert NZ == 5


def test_air_density_ideal_gas(tmp_path):
    path = write_cm1_file(tmp_path / "rho.nc", fields=("prs", "th", "qv"))
    set_fields(path, prs=90000.0, th=305.0, qv=0.01)
    with CM1File(path) as cm1:
        rho = diagnostics.air_density(cm1, 0, level=2)
    temp = 305.0 * (0.9 ** (287.04 / 1005.7)) * (1 + 0.61 * 0.01)
    np.testing.assert_allclose(rho, 90000.0 / (287.04 * temp), rtol=1e-5)


def test_w_scalar_averages_faces_without_winterp(tmp_path):
    path = write_cm1_file(tmp_path / "w.nc", fields=("w",))
    with Dataset(path, "a") as ds:
        ds["w"][:] = np.arange(NZ + 1, dtype="f4")[None, :, None, None] * np.ones(ds["w"].shape, "f4")
    with CM1File(path) as cm1:
        np.testing.assert_allclose(diagnostics.w_scalar(cm1, 0, 2), 2.5)


def test_column_mass_flux_extremes_match_brute_force(tmp_path):
    path = write_cm1_file(tmp_path / "f.nc", fields=("prs", "th", "qv", "winterp", "qr", "qg"))
    with CM1File(path) as cm1:
        fmax, fmin = diagnostics.column_mass_flux_extremes(cm1, 0, ["qr", "qg"])
        for name in ("qr", "qg"):
            flux = np.stack(
                [
                    1000
                    * diagnostics.air_density(cm1, 0, k)
                    * cm1.read("winterp", level=k)
                    * cm1.read(name, level=k)
                    for k in range(NZ)
                ]
            )
            np.testing.assert_allclose(fmax[name], flux.max(axis=0), rtol=1e-6)
            np.testing.assert_allclose(fmin[name], flux.min(axis=0), rtol=1e-6)
        top, _ = diagnostics.column_mass_flux_extremes(cm1, 0, ["qr"], top_km=0.7)
        assert np.all(top["qr"] <= fmax["qr"] + 1e-12)
