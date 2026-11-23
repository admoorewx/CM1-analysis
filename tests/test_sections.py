"""Cross-section placement, storm motion, section physics, and the plots end to end."""

import numpy as np
import pytest
from conftest import FULL_FIELDS, NY, write_cm1_file
from netCDF4 import Dataset

from cm1analysis import cli, sections
from cm1analysis.io import CM1File

NXW = 40  # a wider grid than the default, so windows and fronts have room
DX = 0.5
X = np.arange(NXW) * DX + DX / 2
SPEED = 20.0  # m/s, gust-front motion in the synthetic run
INTERVAL = 120.0  # s between outputs


def front_x(seconds: float) -> float:
    return 9.0 + SPEED * seconds / 1000.0  # km


def write_moving_storm(directory, ntimes=3, apex_row=3):
    """A cold pool (theta' = -5 K) west of a gust front that advances SPEED m/s; the apex is one row ahead."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for n in range(ntimes):
        t = n * INTERVAL
        path = write_cm1_file(directory / f"cm1out_{n + 1:06d}.nc", times=(t,), fields=FULL_FIELDS, x_km=X)
        with Dataset(path, "a") as ds:
            gx = front_x(t)
            behind = np.where(X <= gx, -5.0, 0.0)
            th = np.zeros(ds["thpert"].shape, "f4")
            th[:, :, :, :] = behind  # whole depth, every row
            th[:, :, apex_row, (X > gx) & (X <= gx + DX)] = -5.0  # apex row reaches one point further
            ds["thpert"][:] = th
            ds["th"][:] = 300.0 + th
            cref = np.zeros(ds["cref"].shape, "f4")
            cref[..., np.argmin(np.abs(X - gx))] = 60.0
            ds["cref"][:] = cref
            ds["u"][:] = np.broadcast_to(
                np.linspace(5.0, 25.0, ds["u"].shape[1])[None, :, None, None], ds["u"].shape
            )
            ds["uinterp"][:] = np.broadcast_to(
                np.linspace(5.0, 25.0, ds["uinterp"].shape[1])[None, :, None, None], ds["uinterp"].shape
            )
            ds["winterp"][:] = 0.0
            ds["w"][:] = 0.0
        paths.append(path)
    return paths


OPTIONS = {
    "y_km": None,
    "auto_row": "apex",
    "window_km": 10.0,
    "trailing_fraction": 0.75,
    "cp_threshold_k": -2.0,
    "storm_u": None,
    "smooth_km": 1.0,
    "motion_window_s": 3600.0,
    "leading_edge": "cold_pool",
    "edge_dbz": 40.0,
}


@pytest.fixture(autouse=True)
def fresh_cache():
    sections.clear_cache()
    sections._placements.cache_clear()
    yield


def test_placement_follows_apex_and_gust_front(tmp_path):
    files = write_moving_storm(tmp_path / "run")
    places = sections.placements(files, OPTIONS)
    for n, path in enumerate(files):
        p = places[(path, 0)]
        assert p.j == 3  # the apex row
        assert p.gust_x_km == pytest.approx(front_x(n * INTERVAL) + DX, abs=DX / 2 + 1e-6)
        width = X[p.xs][-1] - X[p.xs][0]
        assert width == pytest.approx(10.0, abs=DX)
        behind = p.gust_x_km - X[p.xs][0]
        assert behind == pytest.approx(7.5, abs=DX)  # trailing_fraction of the window behind the front


def test_storm_motion_from_gust_front(tmp_path):
    files = write_moving_storm(tmp_path / "run")
    places = sections.placements(files, OPTIONS)
    speeds = [places[(f, 0)].storm_u for f in files]
    assert speeds == pytest.approx([SPEED] * 3, rel=0.05)
    assert {places[(f, 0)].storm_source for f in files} == {"gust front, 60-min fit"}

    fixed = sections.placements(files, {**OPTIONS, "storm_u": 12.5})
    assert {(p.storm_u, p.storm_source) for p in fixed.values()} == {(12.5, "configured")}


def test_single_time_has_unknown_motion_and_pinned_row(tmp_path):
    files = write_moving_storm(tmp_path / "run", ntimes=1)
    place = sections.placements(files, {**OPTIONS, "y_km": 0.8})[(files[0], 0)]
    assert place.storm_u is None and place.storm_source == "unknown"
    assert place.j == 1  # y = 0.75 km is nearest 0.8
    with pytest.raises(ValueError, match="outside the domain"):
        sections.placements(files, {**OPTIONS, "y_km": 99.0})


def test_section_fields_and_physics(tmp_path):
    files = write_moving_storm(tmp_path / "run", ntimes=2)
    places = sections.placements(files, OPTIONS)
    with CM1File(files[0]) as cm1:
        sec = sections.Section(cm1, places[(files[0], 0)], top_km=1.5)
        assert sec.z[-1] <= 1.5 and sec.field("thpert").shape == (sec.ktop, sec.x.size)
        # staggered u averaged onto scalar points equals uinterp here
        np.testing.assert_allclose(sec.field("u"), sec.field("uinterp"), atol=1e-5)
        # u rises linearly with height and w = 0, so eta = du/dz everywhere
        dudz = np.gradient(sec.u(), sec.z * 1000.0, axis=0)
        np.testing.assert_allclose(sec.yvort(), dudz, rtol=1e-6)
        u_rel, label = sec.u_relative()
        np.testing.assert_allclose(u_rel, sec.u() - sec.place.storm_u)
        assert "storm-relative" in label
        # the section buoyancy matches the plan-view buoyancy on the same row
        from cm1analysis.diagnostics import buoyancy

        plan = buoyancy(cm1, 0, level=1)[sec.j, sec.xs]
        np.testing.assert_allclose(sec.buoyancy()[1], plan, rtol=1e-6)


def test_base_state_u_from_upert(tmp_path):
    path = write_cm1_file(tmp_path / "u.nc", fields=("u", "upert", "uinterp", "thpert", "cref"))
    with Dataset(path, "a") as ds:
        profile = np.linspace(2.0, 10.0, ds["u"].shape[1])
        ds["upert"][:] = 1.5
        ds["u"][:] = profile[None, :, None, None] + 1.5
    place = sections.Placement(path, 0, 0.0, 2, slice(0, 4), None, None, "unknown")
    with CM1File(path) as cm1:
        u0, source = sections.Section(cm1, place, top_km=10.0).base_u()
    assert source == "initial wind (u - upert)"
    np.testing.assert_allclose(u0, profile, rtol=1e-6)


def test_cache_returns_same_array_and_respects_budget(tmp_path, monkeypatch):
    files = write_moving_storm(tmp_path / "run", ntimes=1)
    place = sections.placements(files, OPTIONS)[(files[0], 0)]
    with CM1File(files[0]) as cm1:
        sec = sections.Section(cm1, place, top_km=2.0)
        assert sec.band("thpert") is sec.band("thpert")
        monkeypatch.setattr(sections, "CACHE_BYTES", 1)
        sec.field("th")
        assert len(sections._CACHE) == 1  # older entries evicted


def test_all_cross_sections_end_to_end(tmp_path, capsys):
    write_moving_storm(tmp_path / "data" / "ctl", ntimes=2)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "output_dir: unused\nplots:\n  cross_section:\n    window_km: 10.0\n    top_km: 1.8\n"
        "    vector_dx_km: 1.0\n    vector_dz_km: 0.4\n"
        "    cmt_section: {lead_extent_km: 2.0, lead_depth_km: 1.0,\n"
        "                  trail_extent_km: 5.0, trail_depth_km: 1.5}\n"
    )
    out = tmp_path / "out"
    args = [
        "run",
        "cross_section",
        str(tmp_path / "data" / "ctl"),
        "-o",
        str(out),
        "-c",
        str(cfg),
        "--dpi",
        "40",
    ]
    assert cli.main(args) == 0, capsys.readouterr().out
    written = sorted(p.relative_to(out).as_posix() for p in out.rglob("*.png"))
    names = {
        "w_section",
        "thpert_section",
        "prspert_section",
        "sr_wind_section",
        "yvort_section",
        "buoyancy_section",
        "cmt_section",
        "hydrometeor_section",
        "hydrometeor_flux_section",
        "cooling_section",
    }
    assert written == sorted(f"ctl/cross_section/{n}/{n}_ctl_{t:06d}s.png" for n in names for t in (0, 120))


def test_experiment_section_comparison(tmp_path):
    data = tmp_path / "data"
    write_moving_storm(data / "ctl", ntimes=2)
    write_moving_storm(data / "exp", ntimes=2, apex_row=4)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        f"output_dir: {tmp_path / 'out'}\ndpi: 40\nreference: control/ctl\nexperiments:\n"
        f"  control:\n    ctl: {{path: {data / 'ctl'}}}\n  g:\n    exp: {{path: {data / 'exp'}}}\n"
        "plots:\n  cross_section:\n    window_km: 10.0\n    top_km: 1.8\n"
    )
    args = ["run", "cross_section", "-c", str(cfg), "-e", "g", "--scope", "experiment"]
    assert cli.main(args) == 0
    written = sorted(p.name for p in (tmp_path / "out").rglob("*.png"))
    assert written == ["compare_thpert_section_g_000000s.png", "compare_thpert_section_g_000120s.png"]
    assert NY == 6


def test_storm_motion_is_a_line_fit_over_the_window():
    # 5-min output, edge moving 20 m/s with ~1 km of random jitter: the 1-h fit is far steadier
    times = np.arange(0.0, 14401.0, 300.0)
    jitter = np.random.default_rng(1).normal(0.0, 1.0, times.size)
    track = [(t, 100.0 + 0.02 * t + j) for t, j in zip(times, jitter, strict=True)]
    mid = slice(6, -6)  # times with a full window
    smooth = np.array(sections.storm_motion(track, 3600.0)[mid])
    noisy = np.array(sections.storm_motion(track, 0.0)[mid])  # neighbors only
    assert np.abs(smooth - 20.0).max() < 1.5
    assert np.std(noisy - 20.0) > 3 * np.std(smooth - 20.0)
    assert sections.storm_motion([(0.0, None), (300.0, 5.0)], 3600.0) == [None, None]


def test_reflectivity_leading_edge(tmp_path):
    files = write_moving_storm(tmp_path / "run")
    places = sections.placements(files, {**OPTIONS, "leading_edge": "reflectivity", "edge_dbz": 40.0})
    # the synthetic cref peak (60 dBZ) sits on the front, so it moves at the same speed
    assert [places[(f, 0)].storm_u for f in files] == pytest.approx([SPEED] * 3, rel=0.05)
    assert places[(files[0], 0)].storm_source == "40 dBZ edge, 60-min fit"
    with pytest.raises(ValueError, match="leading_edge"):
        sections.placements(files, {**OPTIONS, "leading_edge": "front"})


def test_band_average_forms_products_row_by_row(tmp_path):
    files = write_moving_storm(tmp_path / "run", ntimes=1)
    place = sections.placements(files, OPTIONS)[(files[0], 0)]
    with Dataset(files[0], "a") as ds:  # u and w anti-correlated across rows: mean(u*w) != mean(u)*mean(w)
        sign = np.where(np.arange(NY) % 2 == 0, 1.0, -1.0)[None, None, :, None]
        ds["uinterp"][:] = 10.0 * sign * np.ones(ds["uinterp"].shape)
        ds["winterp"][:] = -2.0 * sign * np.ones(ds["winterp"].shape)
    with CM1File(files[0]) as cm1:
        sec = sections.Section(cm1, place, top_km=1.5, average_km=10.0)
        assert sec.averaged and sec.band("thpert").shape[1] == NY
        np.testing.assert_allclose(sec.field("uinterp"), 0.0, atol=1e-6)  # rows cancel
        np.testing.assert_allclose(sec.mean(sec.u_band() * sec.w_band()), -20.0)  # product first
        one = sections.Section(cm1, place, top_km=1.5)
        assert not one.averaged and one.band("thpert").shape[1] == 1


def test_cmt_terms_analytic(tmp_path):
    """u = a + b z + s x and p' = k x on a uniform w: every term has a closed form."""
    from cm1analysis.plots.cross_section import cmt_terms

    path = write_cm1_file(tmp_path / "c.nc", fields=FULL_FIELDS, x_km=X)
    a, b, slope, k, w0, c_x = 5.0, 4e-3, 2e-4, 0.05, 2.0, 12.0  # u in m/s, z and x in m, p' in Pa
    with Dataset(path, "a") as ds:
        z = ds["z"][:] * 1000.0
        x_s = ds["xh"][:] * 1000.0
        x_f = ds["xf"][:] * 1000.0
        ds["uinterp"][:] = np.broadcast_to(
            a + b * z[:, None, None] + slope * x_s[None, None, :], ds["uinterp"].shape
        )
        ds["u"][:] = np.broadcast_to(a + b * z[:, None, None] + slope * x_f[None, None, :], ds["u"].shape)
        ds["upert"][:] = np.broadcast_to(slope * x_f[None, None, :], ds["upert"].shape)  # so u0 = a + b z
        ds["winterp"][:] = w0
        ds["prspert"][:] = np.broadcast_to(k * x_s[None, None, :], ds["prspert"].shape)
    place = sections.Placement(path, 0, 0.0, 2, slice(5, 30), 10.0, c_x, "configured")
    with CM1File(path) as cm1:
        sec = sections.Section(cm1, place, top_km=1.8, average_km=1.0)
        terms, source = cmt_terms(sec)
        rho = sec.mean(sec.density_band())
    u = a + b * sec.z[:, None] * 1000.0 + slope * sec.x[None, :] * 1000.0
    assert source == "initial wind (u - upert)"
    np.testing.assert_allclose(terms["flux"], (u - c_x) * w0, rtol=1e-4)
    np.testing.assert_allclose(terms["convergence"], -b * w0 * 3600.0, rtol=1e-3)
    np.testing.assert_allclose(terms["va_pert"], 0.0, atol=1e-3)  # u' = s x has no vertical shear
    np.testing.assert_allclose(terms["va_base"], -w0 * b * 3600.0, rtol=1e-3)
    np.testing.assert_allclose(terms["ha"], -(u - c_x) * slope * 3600.0, rtol=1e-3)
    np.testing.assert_allclose(terms["pga"], -k / rho * 3600.0, rtol=1e-3)
