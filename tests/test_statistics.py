"""Effect-size functions against hand and scipy values, and the statistics plots end to end."""

import csv

import numpy as np
import pytest
import scipy.stats
from conftest import write_cm1_file, write_run
from netCDF4 import Dataset

from cm1analysis import cli, registry, stats
from cm1analysis.pipeline import RunInput
from cm1analysis.plots import statistics as st


@pytest.fixture(autouse=True)
def fresh_cache():
    st._accumulate.cache_clear()
    yield


# ------------------------------------------------------------------ stats
def test_cohens_d_by_hand():
    a, b = [1.0, 2.0, 3.0, 4.0], [3.0, 4.0, 5.0, 6.0]
    # equal variances 5/3, so pooled SD = sqrt(5/3); means differ by -2
    assert stats.cohens_d(a, b) == pytest.approx(-2.0 / np.sqrt(5.0 / 3.0))
    assert stats.cohens_d(b, a) == pytest.approx(2.0 / np.sqrt(5.0 / 3.0))
    assert np.isnan(stats.cohens_d([1.0, 1.0], [1.0, 1.0]))
    assert np.isnan(stats.cohens_d([1.0], [1.0, 2.0]))


def test_cliffs_delta_matches_pairwise_count():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 10, 200).astype(float)  # ties included
    b = rng.integers(2, 12, 150).astype(float)
    diff = a[:, None] - b[None, :]
    expected = (np.sum(diff > 0) - np.sum(diff < 0)) / diff.size
    assert stats.cliffs_delta(a, b) == pytest.approx(expected)
    assert stats.cliffs_delta(a, b) == pytest.approx(-stats.cliffs_delta(b, a))
    assert stats.cliffs_delta([5.0, 6.0], [1.0, 2.0]) == 1.0
    # equals 2 * (U / (na nb)) - 1 with the Mann-Whitney U of a
    u = scipy.stats.mannwhitneyu(a, b).statistic
    assert stats.cliffs_delta(a, b) == pytest.approx(2 * u / (a.size * b.size) - 1)


def test_ks_statistic_matches_scipy():
    rng = np.random.default_rng(2)
    a, b = rng.normal(0, 1, 300), rng.normal(0.4, 1.3, 250)
    assert stats.ks_statistic(a, b) == pytest.approx(scipy.stats.ks_2samp(a, b).statistic)
    ties_a, ties_b = np.array([1.0, 1.0, 2.0, 3.0]), np.array([1.0, 2.0, 2.0])
    assert stats.ks_statistic(ties_a, ties_b) == pytest.approx(scipy.stats.ks_2samp(ties_a, ties_b).statistic)


def test_robust_scale_and_quantile_effects():
    ref = np.array([1.0, 2.0, 3.0, 4.0, 100.0])  # median 3, |x - 3| = 2,1,0,1,97 -> MAD 1
    assert stats.robust_scale(ref) == pytest.approx(1.4826)
    run = ref + 2.0
    q = stats.quantile_effects(run, ref, [50, 100])
    assert q[50.0].diff == pytest.approx(2.0)
    assert q[50.0].sqd == pytest.approx(2.0 / 1.4826)
    assert q[50.0].ratio == pytest.approx(5.0 / 3.0)
    assert q[100.0].q_run == 102.0 and q[100.0].q_ref == 100.0


def test_labels_and_describe():
    assert stats.cliffs_label(0.1) == "negligible"
    assert stats.cliffs_label(-0.4) == "medium"
    assert stats.cliffs_label(0.5) == "large"
    assert stats.cohens_label(0.3) == "small"
    assert stats.cohens_label(float("nan")) == ""
    d = stats.describe([1.0, 2.0, 3.0, np.nan, 4.0])
    assert d["n"] == 4 and d["mean"] == 2.5 and d["min"] == 1.0 and d["max"] == 4.0
    assert d["std"] == pytest.approx(np.std([1, 2, 3, 4], ddof=1))


def test_pearson_r():
    x = np.arange(10.0)
    assert stats.pearson_r(x, 2 * x + 1) == pytest.approx(1.0)
    y = np.where(x > 4, np.nan, -x)
    assert stats.pearson_r(x, y) == pytest.approx(-1.0)  # only the finite pairs count
    assert np.isnan(stats.pearson_r(x, np.ones(10)))


# ------------------------------------------------------------ accumulation
def opts(**kw):
    return {**registry.category_options("statistics", {}), **kw}


def test_swath_and_cold_pool_accumulate_over_all_times(tmp_path):
    files = write_run(tmp_path / "ctl", ntimes=3, fields=("u10", "v10", "thpert", "sws"))
    acc = st.accumulate(RunInput("ctl", tuple(files)), opts())
    speeds, colds = [], []
    for path in files:
        with Dataset(path) as ds:
            speeds.append(np.hypot(ds["u10"][0], ds["v10"][0]))
            colds.append(ds["thpert"][0, 0])
    np.testing.assert_allclose(acc.swath, np.max(speeds, axis=0), rtol=1e-6)
    np.testing.assert_allclose(acc.cold_pool, np.min(colds, axis=0), rtol=1e-6)
    assert acc.times.tolist() == [0.0, 3600.0, 7200.0]
    assert acc.cell_area_km2 == pytest.approx(np.full((6, 8), 0.25))
    sws = st.accumulate(RunInput("ctl", tuple(files)), opts(swath_field="sws"))
    assert sws.swath_label.endswith("(sws)")


def test_sample_mask_and_area_on_a_stretched_grid(tmp_path):
    x = np.array([0.5, 1.5, 3.0, 5.0, 7.5, 10.5, 14.0, 18.0])  # widening cells
    path = write_cm1_file(tmp_path / "s.nc", fields=("u10", "v10", "thpert"), x_km=x)
    with Dataset(path, "a") as ds:
        u = np.full(ds["u10"].shape, 5.0, "f4")
        u[..., :, -1] = 30.0  # the widest column is damaging
        u[..., 0, 0] = 12.0  # in the sample but below the damaging threshold
        ds["u10"][:] = u
        ds["v10"][:] = 0.0
    s = st.sample(RunInput("s", (path,)), st.KINDS["swath"], opts())
    assert sorted(s.values.tolist()) == [12.0] + [30.0] * 6
    width = np.gradient(x)[-1]
    assert s.area_km2 == pytest.approx(6 * 0.5 * width)
    assert s.extreme == 30.0


# ------------------------------------------------------------ end to end
def write_experiment(tmp_path, reference=True):
    """A reference and two runs whose winds are stronger and cold pools colder, in turn."""
    data = tmp_path / "data"
    for n, name in enumerate(("ctl", "a", "b")):
        files = write_run(data / name, ntimes=2, seed=10 * n)
        for path in files:
            with Dataset(path, "a") as ds:
                ds["u10"][:] = ds["u10"][:] * 8.0 + 15.0 + 4.0 * n
                ds["thpert"][:] = ds["thpert"][:] - 1.5 * n
    ref = "reference: control/ctl\n" if reference else ""
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        f"output_dir: {tmp_path / 'out'}\ndpi: 40\n{ref}experiments:\n"
        f"  control:\n    ctl: {{path: {data / 'ctl'}, color: k}}\n"
        f"  grp:\n    a: {{path: {data / 'a'}, color: red, marker: s}}\n"
        f"    b: {{path: {data / 'b'}, color: blue, marker: ^}}\n"
        "plots:\n  statistics:\n    wind_mask_mps: 5.0\n    cold_pool_mask_k: 0.0\n"
        "    distributions: {kde_sample_cap: 30}\n"
    )
    return cfg


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def test_experiment_statistics_plots(tmp_path):
    cfg = write_experiment(tmp_path)
    assert cli.main(["run", "statistics", "-c", str(cfg), "-e", "grp", "--scope", "experiment"]) == 0
    out = tmp_path / "out" / "grp" / "statistics"
    for name in ("distributions", "histograms", "quantile_differences", "effect_sizes"):
        assert (out / name / f"{name}_grp.png").exists(), name

    rows = read_csv(out / "effect_sizes" / "effect_sizes_grp.csv")
    by = {(r["sample"], r["run"]): r for r in rows}
    assert set(by) == {(k, r) for k in ("swath", "cold_pool") for r in ("ctl", "a", "b")}
    assert float(by["swath", "ctl"]["cliffs_delta"]) == 0.0
    # stronger wind gives positive effects, colder pools negative ones, growing with the shift
    assert 0 < float(by["swath", "a"]["cliffs_delta"]) < float(by["swath", "b"]["cliffs_delta"])
    assert float(by["cold_pool", "b"]["cohens_d"]) < float(by["cold_pool", "a"]["cohens_d"]) < 0
    assert by["swath", "a"]["qd90"] != "" and by["swath", "a"]["qd10"] == ""

    # the CSV agrees with the stats functions on the same samples
    opt = opts(wind_mask_mps=5.0, cold_pool_mask_k=0.0)
    runs = {r: RunInput.from_paths([tmp_path / "data" / r], "cm1out_*.nc") for r in ("ctl", "a")}
    ref, run = (st.sample(runs[r], st.KINDS["swath"], opt) for r in ("ctl", "a"))
    assert float(by["swath", "a"]["ks_d"]) == pytest.approx(stats.ks_statistic(run.values, ref.values), 1e-5)
    assert int(by["swath", "a"]["n"]) == run.values.size

    report = (out / "effect_sizes" / "effect_sizes_grp.txt").read_text()
    assert "Group sensitivity" in report and "swath: n = 2" in report

    q = read_csv(out / "quantile_differences" / "quantile_differences_grp.csv")
    assert {r["run"] for r in q} == {"a", "b"}  # the reference is not compared with itself
    h = read_csv(out / "histograms" / "histograms_grp.csv")
    assert {r["run"] for r in h} == {"ctl", "a", "b"}


def test_reference_is_required_only_where_compared(tmp_path, caplog):
    cfg = write_experiment(tmp_path, reference=False)
    args = ["run", "statistics", "-c", str(cfg), "-e", "grp", "--scope", "experiment"]
    assert cli.main(args) != 0  # quantile_differences and effect_sizes fail
    assert "reference run" in caplog.text
    out = tmp_path / "out" / "grp" / "statistics"
    assert (out / "distributions" / "distributions_grp.png").exists()
    assert not (out / "effect_sizes").exists() or not list((out / "effect_sizes").glob("*.png"))


def test_unknown_options_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown sample"):
        st._kinds(opts(samples=["swath", "updraft"]))
    with pytest.raises(ValueError, match="swath_field"):
        st.accumulate(RunInput("x", (tmp_path / "none.nc",)), opts(swath_field="u10"))


def test_wind_correlations_run_plot(tmp_path):
    files = write_run(tmp_path / "data" / "ctl", ntimes=3)
    with Dataset(files[1], "a") as ds:  # make cold-pool intensity track the wind exactly at t = 3600 s
        ds["v10"][:] = 0.0
        ds["u10"][:] = np.abs(ds["u10"][:]) + 1.0
        ds["cpc"][:] = 2.0 * ds["u10"][:]
    out = tmp_path / "out"
    args = ["run", "statistics", str(tmp_path / "data" / "ctl"), "-o", str(out), "--dpi", "40"]
    assert cli.main([*args, "--plots", "wind_correlations"]) == 0
    png = out / "ctl" / "statistics" / "wind_correlations" / "wind_correlations_ctl.png"
    rows = read_csv(png.with_suffix(".csv"))
    assert [r["time_s"] for r in rows] == ["0", "3600", "7200"]
    assert float(rows[1]["cpc"]) == pytest.approx(1.0, abs=1e-5)
    assert {"cpc", "thpert", "cbudevp", "cbudmltg"} <= set(rows[0])
    assert all(-1.0 <= float(r["thpert"]) <= 1.0 for r in rows)
