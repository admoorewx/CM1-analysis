"""Timeseries metrics, CSV output, and both plots end to end."""

import csv

import numpy as np
import pytest
from conftest import write_cm1_file, write_run

from cm1analysis import cli, registry, sections
from cm1analysis.io import CM1File
from cm1analysis.pipeline import RunInput
from cm1analysis.plots import timeseries as ts

SECTION_OPTS = {**registry.category_options("cross_section", {}), "window_km": 10.0}


def opts(**kw):
    base = {**registry.category_options("timeseries", {}), "section": SECTION_OPTS}
    return {**base, **kw}


@pytest.fixture(autouse=True)
def fresh_cache():
    ts._series.cache_clear()
    sections._placements.cache_clear()
    sections.clear_cache()
    yield


def test_metrics_match_direct_reductions(tmp_path):
    files = write_run(tmp_path / "ctl", ntimes=3)
    data = ts.series(RunInput("ctl", tuple(files)), list(ts.METRICS)[:4], opts())
    assert data.times.tolist() == [0.0, 3600.0, 7200.0]
    for n, path in enumerate(files):
        with CM1File(path) as cm1:
            assert data.values["max_sws"][n] == pytest.approx(cm1.read("sws").max())
            assert data.values["min_thpert"][n] == pytest.approx(cm1.read("thpert", level=0).min())
            assert data.values["max_sus"][n] == pytest.approx(cm1.read("sus").max())
            speed = np.hypot(cm1.read("u10"), cm1.read("v10"))
            assert data.values["max_wind"][n] == pytest.approx(speed.max())


def test_times_come_from_the_files_not_their_names(tmp_path):
    # file names out of step with model time (as with irregular output)
    write_cm1_file(tmp_path / "cm1out_000001.nc", times=(600.0,), fields=("sws",))
    write_cm1_file(tmp_path / "cm1out_000002.nc", times=(0.0, 300.0), fields=("sws",))
    run = RunInput.from_paths([tmp_path], "cm1out_*.nc")
    assert ts.series(run, ["max_sws"], opts()).times.tolist() == [0.0, 300.0, 600.0]


def test_extent_restricts_the_reduction(tmp_path):
    files = write_run(tmp_path / "ctl", ntimes=1)
    run = RunInput("ctl", tuple(files))
    full = ts.series(run, ["max_sws"], opts()).values["max_sws"][0]
    part = ts.series(run, ["max_sws"], opts(extent_km=[0.0, 1.5, 0.0, 1.5])).values["max_sws"][0]
    with CM1File(files[0]) as cm1:
        assert part == pytest.approx(cm1.read("sws", y=slice(0, 3), x=slice(0, 3)).max())
    assert part <= full


def test_missing_fields_skip_only_that_metric(tmp_path, caplog):
    files = write_run(tmp_path / "ctl", ntimes=2, fields=("sws", "u10", "v10"))
    data = ts.series(RunInput("ctl", tuple(files)), ["max_sws", "min_thpert"], opts())
    assert list(data.values) == ["max_sws"]
    assert "min_thpert skipped" in caplog.text
    with pytest.raises(ValueError, match="unknown metric"):
        ts.series(RunInput("ctl", tuple(files)), ["max_speed"], opts())


def test_run_timeseries_figure_and_csv(tmp_path):
    write_run(tmp_path / "data" / "ctl", ntimes=3)
    out = tmp_path / "out"
    assert cli.main(["run", "timeseries", str(tmp_path / "data" / "ctl"), "-o", str(out), "--dpi", "40"]) == 0
    png = out / "ctl" / "timeseries" / "timeseries" / "timeseries_ctl.png"
    assert png.exists()
    rows = list(csv.reader(png.with_suffix(".csv").open()))
    assert rows[0] == ["time_s", *ts.DEFAULT_METRICS]
    assert [r[0] for r in rows[1:]] == ["0", "3600", "7200"]


def test_experiment_comparison_aligns_runs_of_different_length(tmp_path):
    data = tmp_path / "data"
    write_run(data / "ctl", ntimes=3, seed=0)
    write_run(data / "a", ntimes=2, seed=5)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        f"output_dir: {tmp_path / 'out'}\ndpi: 40\nreference: control/ctl\nexperiments:\n"
        f"  control:\n    ctl: {{path: {data / 'ctl'}, color: k}}\n"
        f"  lr:\n    a: {{path: {data / 'a'}, color: red}}\n"
        "plots:\n  timeseries:\n    metrics: [max_wind, min_thpert]\n"
    )
    assert cli.main(["run", "timeseries", "-c", str(cfg), "-e", "lr", "--scope", "experiment"]) == 0
    folder = tmp_path / "out" / "lr" / "timeseries" / "compare_timeseries"
    assert sorted(p.name for p in folder.glob("*.png")) == [
        "compare_timeseries_lr_max_wind.png",
        "compare_timeseries_lr_min_thpert.png",
    ]
    rows = list(csv.reader((folder / "compare_timeseries_lr_max_wind.csv").open()))
    assert rows[0] == ["time_s", "ctl", "a"]
    assert rows[3][0] == "7200" and rows[3][2] == ""  # the shorter run is blank, not shifted


def test_severe_coverage_is_share_of_storm_area(tmp_path):
    from netCDF4 import Dataset

    path = write_cm1_file(tmp_path / "s.nc", fields=("u10", "v10", "cref"))
    with Dataset(path, "a") as ds:
        cref = np.zeros(ds["cref"].shape, "f4")
        cref[..., :4, :] = 45.0  # 4 rows x 8 cols = 32 storm cells
        ds["cref"][:] = cref
        u = np.full(ds["u10"].shape, 5.0, "f4")
        u[..., 0, :4] = 30.0  # 4 severe cells
        ds["u10"][:] = u
        ds["v10"][:] = 0.0
    data = ts.series(RunInput("s", (path,)), ["severe_coverage"], opts())
    assert data.values["severe_coverage"][0] == pytest.approx(100.0 * 4 / 32)


def test_system_speed_and_cold_pool_wind_follow_the_storm(tmp_path):
    from test_sections import SPEED, write_moving_storm

    files = write_moving_storm(tmp_path / "run", ntimes=3)
    data = ts.series(
        RunInput("run", tuple(files)),
        ["system_speed", "cold_pool_wind"],
        opts(lead_extent_km=3.0, lead_depth_km=1.5),
    )
    assert data.values["system_speed"] == pytest.approx([SPEED] * 3, rel=0.05)
    # u rises linearly from 5 to 25 m/s over the 5 levels; the highest level below 1.5 km is z = 1.4 km
    with CM1File(files[0]) as cm1:
        k = int(np.searchsorted(cm1.z_km, 1.5, side="right")) - 1
        expected = cm1.read("uinterp", level=k).max()
    assert data.values["cold_pool_wind"] == pytest.approx([expected] * 3, rel=1e-5)


def test_speed_vs_cold_pool_wind_plot(tmp_path):
    from test_sections import write_moving_storm

    write_moving_storm(tmp_path / "data" / "run", ntimes=3)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "output_dir: unused\nplots:\n  cross_section: {window_km: 10.0}\n"
        "  timeseries: {lead_extent_km: 3.0, lead_depth_km: 1.5}\n"
    )
    out = tmp_path / "out"
    args = [
        "run",
        "timeseries",
        str(tmp_path / "data" / "run"),
        "-o",
        str(out),
        "-c",
        str(cfg),
        "--dpi",
        "40",
        "--plots",
        "speed_vs_cold_pool_wind",
    ]
    assert cli.main(args) == 0
    png = out / "run" / "timeseries" / "speed_vs_cold_pool_wind" / "speed_vs_cold_pool_wind_run.png"
    rows = list(csv.reader(png.with_suffix(".csv").open()))
    assert rows[0] == ["time_s", "system_speed", "cold_pool_wind"] and len(rows) == 4
