import pytest
from conftest import DX_KM, write_cm1_file, write_run

from cm1analysis import registry
from cm1analysis.io import CM1File
from cm1analysis.pipeline import RunInput, Settings, run_experiment, run_plots, window_for
from cm1analysis.registry import Scope


def test_window_for_extent(cm1_path):
    with CM1File(cm1_path) as cm1:
        ys, xs = window_for(cm1, [0.5, 2.0, 0.0, 1.0])  # x points 0.75..1.75, y 0.25..0.75
        assert cm1.x_km[xs].tolist() == pytest.approx([0.75, 1.25, 1.75])
        assert cm1.y_km[ys].tolist() == pytest.approx([0.25, 0.75])
        with pytest.raises(ValueError, match="fewer than 2"):
            window_for(cm1, [0.0, 0.3, 0.0, 3.0])
        with pytest.raises(ValueError, match="xmin, xmax"):
            window_for(cm1, [0.0, 1.0])


def test_run_input_from_directory_and_files(tmp_path):
    files = write_run(tmp_path / "control", ntimes=2)
    run = RunInput.from_paths([tmp_path / "control"], "cm1out_*.nc")
    assert run.name == "control" and run.files == tuple(files)
    assert RunInput.from_paths(files[::-1], "x", name="c").files == tuple(files)
    with pytest.raises(FileNotFoundError):
        RunInput.from_paths([tmp_path / "nope.nc"], "x")


def test_resolve_options(clean_registry):
    clean_registry.common_options("planar", {"extent_km": None})
    clean_registry.register("a", "planar", options={"level_km": None})(lambda ctx: None)
    clean_registry.register("b", "planar")(lambda ctx: None)
    spec = clean_registry.get("a")
    assert clean_registry.resolve_options(spec, {}) == {"extent_km": None, "level_km": None}
    merged = clean_registry.resolve_options(spec, {"extent_km": [0, 1, 0, 1], "a": {"level_km": 1.0}})
    assert merged == {"extent_km": [0, 1, 0, 1], "level_km": 1.0}
    # a per-plot block may also override the common options for that plot only
    assert clean_registry.resolve_options(spec, {"a": {"extent_km": [0, 2, 0, 2]}})["extent_km"] == [
        0,
        2,
        0,
        2,
    ]
    with pytest.raises(ValueError, match="unknown key"):
        clean_registry.resolve_options(spec, {"extnt_km": 1})
    with pytest.raises(ValueError, match="unknown option"):
        clean_registry.resolve_options(spec, {"a": {"level": 1}})
    with pytest.raises(ValueError, match="redefines common"):
        clean_registry.register("c", "planar", options={"extent_km": 1})(lambda ctx: None)


def test_derived_requirements(clean_registry, tmp_path):
    clean_registry.register("w", "planar", requires=["low_level_wind"])(lambda ctx: None)
    with CM1File(write_cm1_file(tmp_path / "a.nc", fields=("u", "v"))) as cm1:
        assert [s.name for s in clean_registry.check(cm1)[0]] == ["w"]
    with CM1File(write_cm1_file(tmp_path / "b.nc", fields=("cref",))) as cm1:
        _, skipped = clean_registry.check(cm1)
    assert skipped == {"w": ["low_level_wind (u10+v10 or uinterp+vinterp or u+v)"]}


def _settings(tmp_path, **kw):
    return Settings(output_dir=tmp_path / "out", dpi=40, **kw)


def test_run_plots_calls_each_scope_and_records_failures(clean_registry, tmp_path):
    calls = []

    @clean_registry.register("per_time", "planar", requires=["cref"])
    def per_time(ctx):
        calls.append(("time", ctx.seconds, ctx.read("cref").shape))
        if ctx.seconds == 3600.0:
            raise RuntimeError("boom")

    @clean_registry.register("per_run", "planar", scope=Scope.RUN, requires=["cref"])
    def per_run(ctx):
        calls.append(("run", [f.seconds for f in ctx.frames()]))

    clean_registry.register("needs_dbz", "planar", requires=["dbz"])(lambda ctx: None)

    run = RunInput("ctl", tuple(write_run(tmp_path / "ctl", ntimes=3)))
    report = run_plots(run, registry.plots("planar"), _settings(tmp_path))
    assert ("time", 0.0, (6, 8)) in calls and ("time", 7200.0, (6, 8)) in calls
    assert ("run", [0.0, 3600.0, 7200.0]) in calls
    assert report.skipped == {"needs_dbz": ["dbz"]}
    assert [(name, err) for name, _, err in report.failures] == [("per_time", "RuntimeError: boom")]


def test_time_selection_and_output_names(clean_registry, tmp_path):
    import matplotlib.pyplot as plt

    @clean_registry.register("snap", "planar")
    def snap(ctx):
        fig, _ = plt.subplots()
        ctx.save(fig, ctx.seconds)

    run = RunInput("ctl", tuple(write_run(tmp_path / "ctl", ntimes=3)))
    report = run_plots(run, registry.plots("planar"), _settings(tmp_path, times=(3600.0,)))
    assert [p.relative_to(tmp_path / "out").as_posix() for p in report.written] == [
        "ctl/planar/snap/snap_ctl_003600s.png"
    ]


def test_experiment_common_times(clean_registry, tmp_path):
    seen = {}

    @clean_registry.register("cmp", "planar", scope=Scope.EXPERIMENT)
    def cmp(ctx):
        seen["runs"] = [r.name for r in ctx.all_runs]
        seen["times"] = [t for t, _ in ctx.common_times()]

    ref = RunInput("ctl", tuple(write_run(tmp_path / "ctl", ntimes=3)))
    short = RunInput("exp", tuple(write_run(tmp_path / "exp", ntimes=2)))
    report = run_experiment("group", [short], registry.plots("planar"), _settings(tmp_path), reference=ref)
    assert not report.failures
    assert seen == {"runs": ["ctl", "exp"], "times": [0.0, 3600.0]}
    assert DX_KM == 0.5
