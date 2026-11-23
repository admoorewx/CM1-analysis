"""End-to-end compute tests on a small file laid out like real CM1 output (ported from the legacy test_io)."""

import numpy as np
import pytest
from netCDF4 import Dataset

from cm1analysis.io import CM1File
from cm1analysis.pressure import ComputeOptions, SolverSource, compute_file, forcing, stored_options
from cm1analysis.pressure.grid import Box

NX, NY = 60, 50
ZF_KM = np.concatenate([np.arange(0.0, 1.0, 0.1), np.linspace(1.0, 4.0, 9), np.arange(4.5, 12.01, 0.75)])
NZ = ZF_KM.size - 1


def make_cm1_file(path, terrain=False, base_spread=0.0):
    zf = ZF_KM
    zc = 0.5 * (zf[1:] + zf[:-1])
    with Dataset(path, "w", format="NETCDF4_CLASSIC") as ds:
        dims = {"ni": NX, "nj": NY, "nk": NZ, "nip1": NX + 1, "njp1": NY + 1, "nkp1": NZ + 1, "time": 1}
        for name, size in dims.items():
            ds.createDimension(name, size)
        ds.wbc = 2

        def coord(name, dim, values, units):
            var = ds.createVariable(name, "f4", (dim,))
            var[:] = values
            var.units = units

        coord("xh", "ni", np.arange(NX) + 0.5, "km")
        coord("xf", "nip1", np.arange(NX + 1.0), "km")
        coord("yh", "nj", np.arange(NY) + 0.5, "km")
        coord("yf", "njp1", np.arange(NY + 1.0), "km")
        coord("z", "nk", zc, "km")
        coord("zf", "nkp1", zf, "km")
        coord("time", "time", np.array([3600.0]), "seconds")

        zh = ds.createVariable("zh", "f4", ("nk", "nj", "ni"))
        zh.units = "m"
        heights = zc[:, None, None] * 1000.0 * np.ones((NZ, NY, NX))
        if terrain:
            heights[:, 10, 10] += 100.0
        zh[:] = heights

        z = zc[:, None, None] * 1000.0
        th0 = 300.0 + 4e-3 * z
        qv0 = 0.014 * np.exp(-z / 2500.0)
        # Hydrostatic in Exner form, as CM1 builds its base state.
        pi = np.empty(NZ)
        pi[0] = (97000.0 / forcing.P00) ** (forcing.RD / forcing.CP)
        thv = (th0 * (1 + forcing.REPS * qv0) / (1 + qv0)).ravel()
        for k in range(1, NZ):
            dz = (zc[k] - zc[k - 1]) * 1000.0
            pi[k] = pi[k - 1] - forcing.G * dz / (forcing.CP * 0.5 * (thv[k] + thv[k - 1]))
        p0 = (forcing.P00 * pi ** (forcing.CP / forcing.RD))[:, None, None]

        y, x = np.meshgrid(np.arange(NY) + 0.5, np.arange(NX) + 0.5, indexing="ij")
        blob = np.exp(-(((x - 30) / 6) ** 2 + ((y - 25) / 6) ** 2))[None] * np.exp(
            -(((z - 3000) / 1500) ** 2)
        )
        rng = np.random.default_rng(0)
        shape4 = (1, NZ, NY, NX)

        def var(name, units, values, dims=("time", "nk", "nj", "ni")):
            v = ds.createVariable(name, "f4", dims)
            v.units = units
            v[:] = values

        thpert = -4.0 * blob
        qvpert = -1e-3 * blob
        prspert = 150.0 * blob
        var("thpert", "K", thpert[None])
        var("th", "K", (th0 + thpert + base_spread * rng.normal(size=(NZ, NY, NX)))[None])
        var("qvpert", "kg/kg", qvpert[None])
        var("qv", "kg/kg", (qv0 + qvpert)[None])
        var("prspert", "Pa", prspert[None])
        var("prs", "Pa", (p0 + prspert)[None])
        var("qc", "kg/kg", 5e-4 * blob[None])
        var("qr", "kg/kg", 1e-3 * blob[None])
        var("u", "m/s", 10.0 + rng.normal(0, 2, (1, NZ, NY, NX + 1)), ("time", "nk", "nj", "nip1"))
        var("v", "m/s", rng.normal(0, 2, (1, NZ, NY + 1, NX)), ("time", "nk", "njp1", "ni"))
        wf = np.zeros((1, NZ + 1, NY, NX))
        wf[0, 1:-1] = rng.normal(0, 1, (NZ - 1, NY, NX))
        var("w", "m/s", wf, ("time", "nkp1", "nj", "ni"))
        cref = ds.createVariable("cref", "f4", ("time", "nj", "ni"))
        cref.units = "dBZ"
        cref[:] = 40.0 * blob.max(axis=0)[None]
        del shape4


@pytest.fixture
def cm1_path(tmp_path):
    path = tmp_path / "cm1out_000012.nc"
    make_cm1_file(path)
    return path


def run(cm1_path, tmp_path, **options):
    options = {"buffer_km": 8.0, "ztop_km": 10.0} | options
    return compute_file(cm1_path, tmp_path / "out", **options)


QUICK = {"sensitivity": (), "crop_test": False, "frame_test": False}


def test_reader_recovers_grid_and_base_state(cm1_path):
    with CM1File(cm1_path) as cm1:
        source = SolverSource(cm1)
        assert (source.nx, source.ny, source.grid.nz) == (NX, NY, NZ)
        assert source.dx == 1000.0
        assert source.grid.zf[-1] == pytest.approx(12000.0, abs=0.01)
        assert source.hydrometeors == ["qc", "qr"]
        inputs, w, spreads = source.read(0, Box(5, 45, 4, 40))
    assert inputs.uc.shape == (NZ, 36, 40) and w.shape == (NZ + 1, 36, 40)
    assert inputs.uc.dtype == np.float64
    assert spreads["base_th0_spread"] < 1e-3
    p_form, exner_form = inputs.base.hydrostatic_residuals()
    assert exner_form < 1e-4  # float32 storage limits this.


def test_reader_rejects_missing_fields(tmp_path):
    path = tmp_path / "cm1out_000001.nc"
    make_cm1_file(path)
    with Dataset(path, "a") as ds:
        ds.renameVariable("prspert", "p_other")
    with CM1File(path) as cm1, pytest.raises(ValueError, match="missing"):
        SolverSource(cm1)


def test_end_to_end_run_writes_output(cm1_path, tmp_path):
    out = run(cm1_path, tmp_path, splits="all")
    assert out == tmp_path / "out" / "wforce_000012.nc"
    with Dataset(out) as ds:
        assert ds.variant == "A"
        for name in ("p_d", "p_b", "dvpgf", "bpgf", "b", "B", "w", "model_vpgf", "B_loading", "dvpgf_shear"):
            assert name in ds.variables
        roi_nx = ds.roi_i1 - ds.roi_i0
        assert ds["dvpgf"].shape[1:] == (NZ + 1, ds.roi_j1 - ds.roi_j0, roi_nx)
        diag = ds["diagnostics"]
        assert int(diag["validation_failures"][0]) == 0
        assert diag["poisson_residual_dynamic"][0] < 1e-10
        assert np.isfinite(diag["sens_dvpgf_H"][0])
        assert np.all(np.isfinite(diag["crop_B_rel"][0]))
    stored = stored_options(out)
    assert stored["splits"] == ["loading", "shear", "strain"]
    assert ComputeOptions(splits="all", buffer_km=8.0, ztop_km=10.0).satisfied_by(stored)
    assert not ComputeOptions(splits="all", buffer_km=9.0, ztop_km=10.0).satisfied_by(stored)


def test_horizontal_outputs_are_written(cm1_path, tmp_path):
    out = run(cm1_path, tmp_path, along_wind=True, sensitivity=("B", "C", "D"))
    with Dataset(out) as ds:
        assert ds.horizontal == 1 and ds.along_wind == 1
        ny, nx = ds.roi_j1 - ds.roi_j0, ds.roi_i1 - ds.roi_i0
        assert ds["hdpgf_x"].dimensions == ("time", "nk", "nj", "nip1")
        assert ds["hdpgf_x"].shape == (1, NZ, ny, nx + 1)
        assert ds["hbpgf_y"].dimensions == ("time", "nk", "njp1", "ni")
        assert ds["hbpgf_y"].shape == (1, NZ, ny + 1, nx)
        assert ds["xf"].size == nx + 1 and ds["yf"].size == ny + 1
        assert ds["hdpgf_s"].shape == (1, NZ, ny, nx)
        assert "model_hpgf_x" in ds.variables
        diag = ds["diagnostics"]
        for v in "ABCD":
            assert np.isfinite(diag[f"hcons_{v}_rel_rms"][0]), v
        assert diag["hcons_D_rel_rms"][0] < 1e-6  # Exact by construction, up to float32 input.
        assert np.isfinite(diag["sens_hdpgf_C"][0]) and np.isnan(diag["sens_hdpgf_H"][0])
        assert np.all(np.isfinite(diag["crop_hcons_rel_rms"][0]))
    # A file with the horizontal forces also serves a request without them, not the reverse.
    stored = stored_options(out)
    plain = ComputeOptions(buffer_km=8.0, ztop_km=10.0, sensitivity=("B", "C", "D"))
    assert plain.satisfied_by(stored)
    assert not ComputeOptions(buffer_km=8.0, ztop_km=10.0, along_wind=True).satisfied_by(stored)


def test_horizontal_without_crop_test(cm1_path, tmp_path):
    out = run(cm1_path, tmp_path, horizontal=True, sensitivity=(), crop_test=False)
    with Dataset(out) as ds:
        assert ds["diagnostics"]["crop_hdpgf_rel"].shape == (1, 3)
        assert "hdpgf_s" not in ds.variables


def test_existing_output_is_kept(cm1_path, tmp_path):
    run(cm1_path, tmp_path, **QUICK)
    with pytest.raises(FileExistsError):
        run(cm1_path, tmp_path)
    assert run(cm1_path, tmp_path, overwrite=True, **QUICK).exists()


def test_time_range_outside_the_file_writes_nothing(cm1_path, tmp_path):
    assert run(cm1_path, tmp_path, time_range=(0.0, 100.0), **QUICK) is None
    assert not (tmp_path / "out" / "wforce_000012.nc").exists()


def test_terrain_is_rejected(tmp_path):
    path = tmp_path / "cm1out_000001.nc"
    make_cm1_file(path, terrain=True)
    with CM1File(path) as cm1, pytest.raises(ValueError, match="not flat"):
        SolverSource(cm1)


@pytest.mark.parametrize("variant", ["D", "H"])
def test_other_variants_run(cm1_path, tmp_path, variant):
    with Dataset(run(cm1_path, tmp_path, variant=variant, sensitivity=(), crop_test=False)) as ds:
        assert int(ds["diagnostics"]["validation_failures"][0]) == 0
        if variant == "H":
            assert ds["p_d"].units == "1"  # Exner function, not pressure


def test_section_mode_matches_full_roi(cm1_path, tmp_path):
    common = {"sensitivity": (), "crop_test": False, "frame_test": False, "buffer_km": 8.0, "ztop_km": 10.0}
    full = compute_file(cm1_path, tmp_path / "full", **common)
    section = compute_file(cm1_path, tmp_path / "section", section_y_km=25.2, section_x_km=(10, 50), **common)
    with Dataset(full) as full, Dataset(section) as sec:
        assert sec.section == 1 and sec["dvpgf"].shape[2] == 1
        assert float(sec["yh"][0]) == 25.5
        assert (sec.crop_j0, sec.crop_j1) == (17, 34)  # One row plus 8 km each side.
        j = int(np.argmin(np.abs(full["yh"][:] - 25.5)))
        x_full, x_sec = np.asarray(full["xh"][:]), np.asarray(sec["xh"][:])
        shared = np.intersect1d(x_full, x_sec)
        assert shared.size >= 10
        in_full, in_sec = np.isin(x_full, shared), np.isin(x_sec, shared)
        for name in ("dvpgf", "B"):
            a = np.asarray(sec[name][0, 1:-1, 0], dtype=np.float64)[:, in_sec]
            b = np.asarray(full[name][0, 1:-1, j], dtype=np.float64)[:, in_full]
            # Different crops, so agreement is to the lateral-boundary error, not roundoff.
            assert np.sqrt(np.mean((a - b) ** 2)) < 0.2 * np.sqrt(np.mean(b**2))


@pytest.mark.parametrize(
    "options",
    [
        {"section_y_km": 25.0, "roi_km": (0, 10, 0, 10)},
        {"section_x_km": (0, 10)},
        {"variant": "Z"},
        {"splits": ("vorticity",)},
        {"sensitivity": ("A",)},
    ],
)
def test_bad_options_are_rejected(options):
    with pytest.raises(ValueError):
        ComputeOptions(**options)
