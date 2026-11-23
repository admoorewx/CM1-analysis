import numpy as np
import pytest
from conftest import DX_KM, NX, NY, NZ, write_cm1_file
from netCDF4 import Dataset

from cm1analysis.io import CM1File, find_output_files, iter_times


def test_grid(cm1_path):
    with CM1File(cm1_path) as cm1:
        assert cm1.shape == (NZ, NY, NX)
        assert cm1.dx == DX_KM * 1000.0
        assert cm1.dy == DX_KM * 1000.0
        assert cm1.ntimes == 1
        assert cm1.times[0] == 7200.0
        assert cm1.zf_km[-1] == pytest.approx(2.0)


def test_field_locations(cm1_path):
    with CM1File(cm1_path) as cm1:
        assert cm1.location("th") == "scalar"
        assert cm1.location("u") == "u"
        assert cm1.location("v") == "v"
        assert cm1.location("w") == "w"
        assert cm1.location("cref") == "surface"


def test_hydrometeors_exclude_vapor(cm1_path):
    with CM1File(cm1_path) as cm1:
        assert cm1.hydrometeors == ["qr"]


def test_has_and_missing(cm1_path):
    with CM1File(cm1_path) as cm1:
        assert cm1.has("th", "cref")
        assert not cm1.has("th", "dbz")
        assert cm1.missing(["th", "dbz", "prspert"]) == ["dbz", "prspert"]


def test_read_shapes(cm1_path):
    with CM1File(cm1_path) as cm1:
        assert cm1.read("th").shape == (NZ, NY, NX)
        assert cm1.read("th", level=2).shape == (NY, NX)
        assert cm1.read("th", y=3).shape == (NZ, NX)
        assert cm1.read("th", level=slice(0, 2), x=slice(1, 4)).shape == (2, NY, 3)
        assert cm1.read("u").shape == (NZ, NY, NX + 1)
        assert cm1.read("w").shape == (NZ + 1, NY, NX)
        assert cm1.read("cref").shape == (NY, NX)
        assert cm1.read("zh").shape == (NZ, NY, NX)  # static field, no time axis


def test_read_keeps_dtype_unless_asked(cm1_path):
    with CM1File(cm1_path) as cm1:
        assert cm1.read("th").dtype == np.float32
        assert cm1.read("th", dtype=np.float64).dtype == np.float64


def test_read_fills_masked_with_nan(cm1_path):
    with Dataset(cm1_path, "a") as ds:
        ds["cref"][0, 1, 2] = np.ma.masked
    with CM1File(cm1_path) as cm1:
        cref = cm1.read("cref")
        assert np.isnan(cref[1, 2])
        assert np.isfinite(cref).sum() == cref.size - 1


def test_read_errors(cm1_path):
    with CM1File(cm1_path) as cm1:
        with pytest.raises(ValueError, match="no vertical dimension"):
            cm1.read("cref", level=0)
        with pytest.raises(IndexError):
            cm1.read("th", time=1)
        with pytest.raises(KeyError, match="dbz"):
            cm1.read("dbz")


def test_level_and_time_lookup(tmp_path):
    path = write_cm1_file(tmp_path / "multi.nc", times=(0.0, 300.0, 600.0))
    with CM1File(path) as cm1:
        assert cm1.level_index(0.2) == 0
        assert cm1.level_index(0.05) == 0  # below the first scalar level, inside the domain
        assert cm1.level_index(1.9) == NZ - 1
        assert cm1.level_index(2.0, staggered=True) == NZ
        assert cm1.time_index(290.0) == 1
        with pytest.raises(ValueError, match="outside"):
            cm1.level_index(5.0)


def test_stretched_grid_has_no_uniform_spacing(tmp_path):
    path = write_cm1_file(tmp_path / "stretched.nc", x_km=[0.25, 0.75, 1.5, 3.0])
    with CM1File(path) as cm1:
        assert cm1.dx is None
        assert cm1.dy == DX_KM * 1000.0


def test_rejects_non_cm1_file(tmp_path):
    path = tmp_path / "other.nc"
    with Dataset(path, "w") as ds:
        ds.createDimension("x", 3)
        ds.createVariable("lon", "f4", ("x",))
    with pytest.raises(ValueError, match="not CM1 output"):
        CM1File(path)


def test_surface_only_file(tmp_path):
    path = write_cm1_file(tmp_path / "sfc.nc", fields=("cref", "sws"))
    with CM1File(path) as cm1:
        assert cm1.read("sws").shape == (NY, NX)
        assert cm1.hydrometeors == []


def test_find_and_iterate_run(tmp_path):
    for i, t in enumerate((0.0, 300.0, 600.0), start=1):
        write_cm1_file(tmp_path / f"cm1out_{i:06d}.nc", times=(t,))
    files = find_output_files(tmp_path)
    assert [f.name for f in files] == ["cm1out_000001.nc", "cm1out_000002.nc", "cm1out_000003.nc"]
    assert [t for _, _, t in iter_times(files)] == [0.0, 300.0, 600.0]
    with pytest.raises(FileNotFoundError):
        find_output_files(tmp_path, pattern="wforce_*.nc")
