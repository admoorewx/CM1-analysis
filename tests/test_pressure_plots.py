"""End-to-end: the pressure plots through the CLI, computing and caching the derived files."""

import pytest
from conftest import write_run
from netCDF4 import Dataset
from test_pressure_compute import make_cm1_file

from cm1analysis import cli
from cm1analysis.plots import pressure as pressure_plots

QUICK = {"sensitivity": "[]", "crop_test": "false", "frame_test": "false", "buffer_km": 8, "ztop_km": 10}


def config(tmp_path, **options):
    path = tmp_path / "cfg.yaml"
    values = QUICK | options
    path.write_text(
        "output_dir: unused\nplots:\n  pressure:\n" + "".join(f"    {k}: {v}\n" for k, v in values.items())
    )
    return path


@pytest.fixture
def run_dir(tmp_path):
    directory = tmp_path / "data" / "control"
    directory.mkdir(parents=True)
    make_cm1_file(directory / "cm1out_000001.nc")
    return directory


def plot(run_dir, out, cfg, *extra):
    return cli.main(["run", "pressure", str(run_dir), "-o", str(out), "-c", str(cfg), "--dpi", "40", *extra])


def pngs(directory):
    return sorted(p.relative_to(directory).as_posix() for p in directory.rglob("*.png"))


def test_plots_compute_then_reuse_the_derived_file(run_dir, tmp_path, monkeypatch, capsys):
    out = tmp_path / "out"
    cfg = config(tmp_path, height_km=3)
    assert plot(run_dir, out, cfg) == 0
    assert pngs(out) == [
        "control/pressure/horizontal_forcing/horizontal_forcing_control_003600s.png",
        "control/pressure/vertical_forcing/vertical_forcing_control_003600s.png",
    ]
    derived = out / "control" / "pressure" / "derived" / "wforce_control_003600s.nc"
    with Dataset(derived) as ds:
        assert ds.horizontal == 1 and "hdpgf_x" in ds.variables
        assert int(ds["diagnostics"]["validation_failures"][0]) == 0
    assert "0 failure(s)" in capsys.readouterr().out

    # Same options: the cached file is used and nothing is solved.
    def no_compute(*args, **kwargs):
        raise AssertionError("the derived file should have been reused")

    monkeypatch.setattr(pressure_plots, "compute", no_compute)
    assert plot(run_dir, out, cfg, "--times", "3600") == 0
    assert "0 failure(s)" in capsys.readouterr().out
    monkeypatch.undo()

    # A changed compute option recomputes the file.
    assert plot(run_dir, out, config(tmp_path, height_km=3, variant="H"), "--plots", "vertical_forcing") == 0
    with Dataset(derived) as ds:
        assert ds.variant == "H"


def test_section_mode_draws_sections(run_dir, tmp_path, capsys):
    out = tmp_path / "out"
    cfg = config(tmp_path, section_y_km=25, section_x_km="[10, 50]", x_range_km="[15, 45]")
    assert plot(run_dir, out, cfg) == 0
    assert len(pngs(out)) == 2
    with Dataset(out / "control" / "pressure" / "derived" / "wforce_control_003600s.nc") as ds:
        assert ds.section == 1 and ds["dvpgf"].shape[2] == 1
    assert "0 failure(s)" in capsys.readouterr().out


def test_files_without_the_solver_fields_are_skipped(tmp_path, capsys):
    run_dir = tmp_path / "sfc"
    write_run(run_dir, ntimes=1, fields=("cref", "u", "v", "w", "th", "qv"))
    assert plot(run_dir, tmp_path / "out", config(tmp_path)) == 0
    text = capsys.readouterr().out
    assert "skipped" in text and "prspert" in text
    assert not (tmp_path / "out").exists() or pngs(tmp_path / "out") == []
