"""End-to-end: the wind-partition plot through the CLI, including its partition cache."""

import pytest
from conftest import write_cm1_file, write_run
from netCDF4 import Dataset
from test_wind_io import NX, NY, make_cm1_file

from cm1analysis import cli

# No zh: the synthetic zh has random noise, which the partition would rightly reject as terrain.
WIND_FIELDS = ("u", "v", "cref", "prspert")
TIMES = (0, 3600, 7200)


def names(directory, suffix=".png"):
    return sorted(p.relative_to(directory).as_posix() for p in directory.rglob(f"*{suffix}"))


@pytest.fixture
def run_dir(tmp_path):
    write_run(tmp_path / "data" / "control", ntimes=3, interval_s=3600.0, fields=WIND_FIELDS)
    return tmp_path / "data" / "control"


def plot(run_dir, out, *extra):
    return cli.main(["run", "wind", str(run_dir), "-o", str(out), "--dpi", "40", *extra])


def test_partition_plot_computes_and_caches(run_dir, tmp_path, capsys):
    out = tmp_path / "out"
    assert plot(run_dir, out) == 0
    assert names(out) == [
        f"control/wind/wind_partition/wind_partition_control_lowest_{t:06d}s.png" for t in TIMES
    ]
    cached = [f"control/wind/derived/wind_decomp_control_k000_A_{t:06d}s.nc" for t in TIMES]
    assert names(out, ".nc") == cached
    with Dataset(out / cached[1]) as ds:
        assert ds["time"][:].tolist() == [3600.0]
        assert ds.level_index == 0 and ds.variant == "A"
        assert ds["diagnostics"]["validation_failures"].shape == (1,)
    assert "0 failure(s)" in capsys.readouterr().out

    # A second run reuses the cache instead of recomputing it.
    stamps = {name: (out / name).stat().st_mtime_ns for name in cached}
    assert plot(run_dir, out, "--times", "3600") == 0
    assert {name: (out / name).stat().st_mtime_ns for name in cached} == stamps

    # Asking for recompute rewrites just the selected time.
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("output_dir: unused\nplots:\n  wind:\n    wind_partition: {recompute: true}\n")
    assert plot(run_dir, out, "--times", "3600", "-c", str(cfg)) == 0
    assert (out / cached[1]).stat().st_mtime_ns != stamps[cached[1]]
    assert (out / cached[0]).stat().st_mtime_ns == stamps[cached[0]]


def test_options_select_level_and_variant(run_dir, tmp_path):
    out = tmp_path / "out"
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "output_dir: unused\nplots:\n  wind:\n"
        "    extent_km: [0.5, 3.5, 0.5, 2.5]\n"
        "    wind_partition: {height_km: 1.0, variant: C, roi_km: [1, 3, 1, 2]}\n"
    )
    assert plot(run_dir, out, "-c", str(cfg), "--times", "7200") == 0
    assert names(out) == ["control/wind/wind_partition/wind_partition_control_z1km_007200s.png"]
    (cached,) = names(out, ".nc")
    with Dataset(out / cached) as ds:
        assert ds.variant == "C" and "u_div" in ds.variables
        assert ds.roi_source == "manual"
        assert ds.level_index == 2  # synthetic levels are at 0.2, 0.6, 1.0, 1.4, 1.8 km


def test_partition_with_reflectivity_roi(tmp_path):
    """A larger, legacy-style file: the automatic ROI drives the view and the sensitivity tests."""
    run_dir = tmp_path / "data" / "big"
    run_dir.mkdir(parents=True)
    make_cm1_file(run_dir / "cm1out_000007.nc")
    out = tmp_path / "out"
    assert plot(run_dir, out) == 0
    assert names(out) == [
        f"big/wind/wind_partition/wind_partition_big_lowest_{t:06d}s.png" for t in (300, 600)
    ]
    with Dataset(out / "big/wind/derived/wind_decomp_big_k000_A_000600s.nc") as ds:
        diag = ds["diagnostics"]
        assert [int(diag[n][0]) for n in ("roi_i0", "roi_i1", "roi_j0", "roi_j1")] == [25, 95, 20, 80]
        assert ds["chi"].shape == (1, NY, NX)


def test_files_without_native_winds_are_skipped(tmp_path, capsys):
    run_dir = tmp_path / "data" / "sfc"
    run_dir.mkdir(parents=True)
    write_cm1_file(run_dir / "cm1out_000001.nc", fields=("cref", "sws"))
    out = tmp_path / "out"
    assert plot(run_dir, out) == 0
    assert names(out) == []
    assert "wind_partition" in capsys.readouterr().out
