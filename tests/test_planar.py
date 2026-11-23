"""End-to-end: every planar plot on small synthetic runs, through the CLI."""

import pytest
from conftest import write_run

from cm1analysis import cli, registry
from cm1analysis.registry import Scope


def names(directory):
    return sorted(p.relative_to(directory).as_posix() for p in directory.rglob("*.png"))


@pytest.fixture
def run_dir(tmp_path):
    write_run(tmp_path / "data" / "control", ntimes=3, interval_s=3600.0)
    return tmp_path / "data" / "control"


HEIGHT_PLOTS = {"thpert": ["lowest", "z1km"], "prspert": ["lowest", "z1km"], "buoyancy": ["lowest", "z1km"]}
SPECIES = ["qc", "qg", "qhl", "qi", "qr", "qs"]


def test_all_single_run_plots(run_dir, tmp_path, capsys):
    out = tmp_path / "out"
    cfg = tmp_path / "cfg.yaml"  # the synthetic grid is 2 km deep, so ask for heights inside it
    cfg.write_text(
        "output_dir: unused\nplots:\n  planar:\n"
        + "".join(f"    {name}: {{heights_km: [0, 1.0]}}\n" for name in HEIGHT_PLOTS)
        + "    downdraft_swath: {top_km: 1.0}\n"
    )
    assert cli.main(["run", "planar", str(run_dir), "-o", str(out), "-c", str(cfg), "--dpi", "40"]) == 0
    written = names(out)
    times = (0, 3600, 7200)
    for spec in registry.plots("planar", Scope.TIME):
        if spec.name in HEIGHT_PLOTS:
            suffixes = HEIGHT_PLOTS[spec.name]
        elif spec.name == "hydrometeor_flux":
            suffixes = SPECIES
        else:
            suffixes = [None]
        expected = sorted(
            f"control/planar/{spec.name}/{spec.name}_control_" + (f"{s}_" if s else "") + f"{t:06d}s.png"
            for s in suffixes
            for t in times
        )
        assert [w for w in written if w.startswith(f"control/planar/{spec.name}/")] == expected
    for name in ("overview", "cooling_budget"):
        assert len([w for w in written if f"/{name}/" in w]) == 3
    assert len([w for w in written if "/downdraft_swath/" in w]) == 4  # three frames and the final swath
    assert "control/planar/wind_swath/wind_swath_control.png" in written
    assert "control/planar/refl_progression/refl_progression_control.png" in written
    assert "control/planar/downdraft_swath/downdraft_swath_control.png" in written
    assert "0 failure(s)" in capsys.readouterr().out


def test_plot_and_time_selection(run_dir, tmp_path):
    out = tmp_path / "out"
    args = [
        "run",
        "planar",
        str(run_dir),
        "-o",
        str(out),
        "--dpi",
        "40",
        "--plots",
        "cref,overview",
        "--times",
        "3600",
    ]
    assert cli.main(args) == 0
    assert names(out) == [
        "control/planar/cref/cref_control_003600s.png",
        "control/planar/overview/overview_control_003600s.png",
    ]


def test_missing_fields_are_skipped_not_failed(tmp_path, capsys):
    run_dir = tmp_path / "sfc"
    write_run(run_dir, ntimes=1, fields=("cref", "u10", "v10"))
    assert (
        cli.main(
            ["run", "planar", str(run_dir), "-o", str(tmp_path / "out"), "--dpi", "40", "--scope", "time"]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "skipped thpert: needs thpert" in out
    assert "skipped buoyancy: needs buoyancy (th+thpert)" in out
    assert {p.parent.name for p in (tmp_path / "out").rglob("*.png")} == {"cref", "wind10"}


def test_extent_and_options_from_config(run_dir, tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        f"output_dir: {tmp_path / 'out'}\ndpi: 40\n"
        "plots:\n  planar:\n    extent_km: [0.5, 3.0, 0.0, 2.5]\n    thpert: {heights_km: [1.0, 1.1]}\n"
    )
    assert cli.main(["run", "planar", str(run_dir), "-c", str(cfg), "--plots", "thpert", "--times", "0"]) == 0
    # 1.0 and 1.1 km fall on the same level, which is drawn once
    assert names(tmp_path / "out") == ["control/planar/thpert/thpert_control_z1km_000000s.png"]

    cfg.write_text(f"output_dir: {tmp_path / 'out'}\nplots:\n  planar:\n    thpert: {{level: 1}}\n")
    assert cli.main(["run", "planar", str(run_dir), "-c", str(cfg), "--plots", "thpert"]) == 1


@pytest.fixture
def experiment(tmp_path):
    data = tmp_path / "data"
    write_run(data / "control", ntimes=2, seed=0)
    write_run(data / "lr" / "a", ntimes=2, seed=10)
    write_run(data / "lr" / "b", ntimes=2, seed=20)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        f"output_dir: {tmp_path / 'out'}\ndpi: 40\nreference: control/control\n"
        "experiments:\n"
        f"  control:\n    control: {{path: {data / 'control'}}}\n"
        f"  lapse_rate:\n    A: {{path: {data / 'lr' / 'a'}, label: 'LR A'}}\n"
        f"    B: {{path: {data / 'lr' / 'b'}}}\n"
    )
    return cfg


def test_experiment_comparisons(experiment, tmp_path):
    args = ["run", "planar", "-c", str(experiment), "-e", "lapse_rate", "--scope", "experiment"]
    assert cli.main(args) == 0
    assert names(tmp_path / "out") == [
        "lapse_rate/planar/compare/compare_lapse_rate_000000s.png",
        "lapse_rate/planar/compare/compare_lapse_rate_003600s.png",
        "lapse_rate/planar/compare_diff/compare_diff_lapse_rate_A_000000s.png",
        "lapse_rate/planar/compare_diff/compare_diff_lapse_rate_A_003600s.png",
        "lapse_rate/planar/compare_diff/compare_diff_lapse_rate_B_000000s.png",
        "lapse_rate/planar/compare_diff/compare_diff_lapse_rate_B_003600s.png",
        "lapse_rate/planar/swath_diff/swath_diff_lapse_rate_A.png",
        "lapse_rate/planar/swath_diff/swath_diff_lapse_rate_B.png",
    ]


def test_difference_on_mismatched_grids_fails_cleanly(tmp_path, capsys):
    from conftest import FULL_FIELDS, write_cm1_file

    data = tmp_path / "data"
    write_run(data / "control", ntimes=1)
    (data / "wide").mkdir()
    write_cm1_file(data / "wide" / "cm1out_000001.nc", times=(0.0,), fields=FULL_FIELDS,
                   x_km=[0.25 + 0.5 * i for i in range(10)])  # fmt: skip
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        f"output_dir: {tmp_path / 'out'}\nreference: control/control\nexperiments:\n"
        f"  control:\n    control: {{path: {data / 'control'}}}\n"
        f"  g:\n    wide: {{path: {data / 'wide'}}}\n"
    )
    args = [
        "run",
        "planar",
        "-c",
        str(cfg),
        "-e",
        "g",
        "--scope",
        "experiment",
        "--plots",
        "compare,swath_diff",
    ]
    assert cli.main(args) == 1
    out = capsys.readouterr().out
    assert "FAILED  swath_diff" in out and "grids differ" in out
    assert names(tmp_path / "out") == ["g/planar/compare/compare_g_000000s.png"]  # side by side still works


def test_comparison_without_reference_fails_clearly(experiment, tmp_path, capsys):
    text = experiment.read_text().replace("reference: control/control\n", "")
    experiment.write_text(text)
    args = ["run", "planar", "-c", str(experiment), "-e", "lapse_rate", "--plots", "swath_diff"]
    assert cli.main(args) == 1
    assert "needs a reference run" in capsys.readouterr().out
