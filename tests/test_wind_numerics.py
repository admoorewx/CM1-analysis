"""Validation table and synthetic cases from cm1_wind_decomposition.md."""

import numpy as np
import pytest

from cm1analysis.wind import diagnostics, ops, solve
from cm1analysis.wind.diagnostics import ROI, roi_from_reflectivity, sensitivity, validate
from cm1analysis.wind.partition import apply_variant, partition_wind

TOL = 1e-10
# A rectangular, anisotropic grid catches any swapped axis or spacing.
NX, NY, DX, DY = 96, 64, 1000.0, 750.0


def random_wind(nx=NX, ny=NY, seed=1):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 5.0, (ny, nx + 1)), rng.normal(0.0, 5.0, (ny + 1, nx))


def corner_gaussian(cx, cy, sigma, nx=NX, ny=NY):
    x = np.arange(nx + 1) * DX
    y = np.arange(ny + 1) * DY
    xx, yy = np.meshgrid(x, y)
    return 1e6 * np.exp(-((xx - cx * nx * DX) ** 2 + (yy - cy * ny * DY) ** 2) / (2 * sigma**2))


def center_gaussian(cx, cy, sigma, nx=NX, ny=NY):
    x = (np.arange(nx) + 0.5) * DX
    y = (np.arange(ny) + 0.5) * DY
    xx, yy = np.meshgrid(x, y)
    return 1e6 * np.exp(-((xx - cx * nx * DX) ** 2 + (yy - cy * ny * DY) ** 2) / (2 * sigma**2))


# ── Operators and solvers ──────────────────────────────────────────────────────


def test_shape_mismatch_fails_loudly():
    u, v = random_wind()
    with pytest.raises(ValueError):
        partition_wind(u[:, :-1], v, DX, DY)


def test_non_finite_input_fails_loudly():
    u, v = random_wind()
    u[3, 3] = np.nan
    with pytest.raises(ValueError):
        partition_wind(u, v, DX, DY)


def test_output_shapes_and_precision():
    u, v = random_wind()
    p = partition_wind(u.astype(np.float32), v.astype(np.float32), DX, DY)
    assert p.psi.shape == (NY + 1, NX + 1)
    assert p.chi.shape == (NY, NX)
    assert p.delta.shape == (NY, NX)
    assert p.zeta.shape == (NY - 1, NX - 1)
    assert p.u_h.shape == (NY, NX + 1)
    assert p.v_h.shape == (NY + 1, NX)
    for field in (p.psi, p.chi, p.u_h, p.v_h, p.delta, p.zeta):
        assert field.dtype == np.float64


def test_chi_dst_matches_sparse():
    u, v = random_wind()
    delta = ops.divergence(u, v, DX, DY)
    dst = solve.solve_chi(delta, DX, DY)
    sparse = solve.solve_chi_sparse(delta, DX, DY)
    assert np.max(np.abs(dst - sparse)) < TOL * np.max(np.abs(sparse))


def test_psi_dst_matches_sparse():
    u, v = random_wind()
    zeta = ops.vorticity(u, v, DX, DY)
    dst = solve.solve_psi(zeta, DX, DY)
    sparse = solve.solve_psi_sparse(zeta, DX, DY)
    assert np.max(np.abs(dst - sparse)) < TOL * np.max(np.abs(sparse))


def test_psi_is_zero_on_outer_ring():
    u, v = random_wind()
    psi = partition_wind(u, v, DX, DY).psi
    assert not (psi[0].any() or psi[-1].any() or psi[:, 0].any() or psi[:, -1].any())


def test_west_edge_laplacian_row():
    """The spec's edge row: (chi[j,1] - 3 chi[j,0]) / dx^2 plus the y-terms."""
    u, v = random_wind()
    delta = ops.divergence(u, v, DX, DY)
    chi = solve.solve_chi(delta, DX, DY)
    j = NY // 2
    row = (chi[j, 1] - 3 * chi[j, 0]) / DX**2 + (chi[j + 1, 0] - 2 * chi[j, 0] + chi[j - 1, 0]) / DY**2
    assert row == pytest.approx(delta[j, 0], rel=1e-9)


# ── Validation table ───────────────────────────────────────────────────────────


@pytest.fixture
def checks():
    u, v = random_wind()
    return validate(partition_wind(u, v, DX, DY), u, v, DX, DY)


def test_validation_table_passes(checks):
    assert diagnostics.failed_checks(checks) == []


@pytest.mark.parametrize("name", ["div_vpsi_rel", "vor_vchi_rel", "div_vh_rel", "vor_vh_rel"])
def test_operator_residuals(checks, name):
    assert checks[name] < TOL


def test_boundary_conditions(checks):
    assert checks["vpsi_normal_edge_max"] == 0.0
    assert checks["chi_ghost_rule_max"] == 0.0


def test_net_harmonic_flux_is_roundoff(checks):
    assert checks["vh_net_flux_rel"] < TOL


def test_ke_cross_terms(checks):
    for name in ("psi_chi", "psi_h", "chi_h"):
        assert abs(checks[f"ke_cross_{name}"]) < 0.01
        assert abs(checks[f"ke_cross_{name}_edgewt"]) < TOL


def test_failed_checks_reports_a_bad_value(checks):
    checks["div_vh_rel"] = 1e-3
    assert diagnostics.failed_checks(checks) == ["div_vh_rel"]


def test_broken_recovery_operator_is_detected():
    """A wrong operator must fail these checks, or they prove nothing."""
    u, v = random_wind()
    p = partition_wind(u, v, DX, DY)
    p.u_psi = p.u_psi * 1.01  # V_psi is no longer nondivergent.
    p.u_h = u - p.u_psi - p.u_chi
    failed = diagnostics.failed_checks(validate(p, u, v, DX, DY))
    assert "div_vpsi_rel" in failed and "div_vh_rel" in failed


# ── Synthetic cases ────────────────────────────────────────────────────────────


def test_compact_vortex_is_all_rotational():
    # sigma = 3 km puts the 24 km y half-width at 8 sigma, so psi ~ 0 on the ring.
    psi_in = corner_gaussian(0.5, 0.5, 3000.0)
    u, v = ops.rotational_wind(psi_in, DX, DY)
    p = partition_wind(u, v, DX, DY)
    scale = max(np.abs(u).max(), np.abs(v).max())
    assert max(np.abs(p.u_chi).max(), np.abs(p.v_chi).max()) < TOL * scale
    assert max(np.abs(p.u_h).max(), np.abs(p.v_h).max()) < 1e-4 * scale
    assert np.abs(p.u_psi - u).max() < 1e-4 * scale
    assert np.abs(p.v_psi - v).max() < 1e-4 * scale
    assert diagnostics.failed_checks(validate(p, u, v, DX, DY)) == []


def test_compact_potential_is_all_divergent():
    chi_in = center_gaussian(0.5, 0.5, 3000.0)
    u, v = ops.divergent_wind(chi_in, DX, DY)
    p = partition_wind(u, v, DX, DY)
    scale = max(np.abs(u).max(), np.abs(v).max())
    assert max(np.abs(p.u_psi).max(), np.abs(p.v_psi).max()) < TOL * scale
    assert max(np.abs(p.u_h).max(), np.abs(p.v_h).max()) < 1e-4 * scale
    assert np.abs(p.chi - chi_in).max() < 1e-4 * np.abs(chi_in).max()
    assert diagnostics.failed_checks(validate(p, u, v, DX, DY)) == []


def test_uniform_wind_is_all_harmonic():
    u = np.full((NY, NX + 1), 12.5)
    v = np.full((NY + 1, NX), -3.0)
    p = partition_wind(u, v, DX, DY)
    assert not p.psi.any() and not p.chi.any()
    assert np.array_equal(p.u_h, u) and np.array_equal(p.v_h, v)
    assert diagnostics.failed_checks(validate(p, u, v, DX, DY)) == []


def test_vortex_near_edge_has_harmonic_flow():
    psi_in = corner_gaussian(0.97, 0.5, 6000.0)
    psi_in -= psi_in[:, 0:1]  # Leaves nonzero psi on the east edge.
    u, v = ops.rotational_wind(psi_in, DX, DY)
    p = partition_wind(u, v, DX, DY)
    scale = max(np.abs(u).max(), np.abs(v).max())
    assert max(np.abs(p.u_h).max(), np.abs(p.v_h).max()) > 0.01 * scale
    assert diagnostics.failed_checks(validate(p, u, v, DX, DY)) == []


# ── Variants ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("variant", ["A", "B", "C"])
def test_variant_parts_sum_to_wind(variant):
    u, v = random_wind()
    p = partition_wind(u, v, DX, DY)
    u_rot, v_rot, u_div, v_div = apply_variant(p, variant)
    u_rest = p.u_h if variant == "A" else 0.0
    v_rest = p.v_h if variant == "A" else 0.0
    assert np.allclose(u_rot + u_div + u_rest, u, rtol=0, atol=1e-11)
    assert np.allclose(v_rot + v_div + v_rest, v, rtol=0, atol=1e-11)


def test_unknown_variant_rejected():
    u, v = random_wind()
    with pytest.raises(ValueError):
        apply_variant(partition_wind(u, v, DX, DY), "D")


# ── ROI and sensitivity ────────────────────────────────────────────────────────


def test_roi_pads_and_keeps_out_of_buffer():
    cref = np.zeros((200, 300))
    cref[80:120, 100:200] = 30.0
    roi = roi_from_reflectivity(cref, 25.0, 1000.0, 1000.0)
    assert (roi.i0, roi.i1, roi.j0, roi.j1) == (75, 225, 55, 145)

    cref[:] = 30.0
    roi = roi_from_reflectivity(cref, 25.0, 1000.0, 1000.0)
    assert (roi.i0, roi.i1, roi.j0, roi.j1) == (20, 280, 20, 180)


def test_roi_threshold_is_respected():
    cref = np.zeros((200, 300))
    cref[80:120, 100:200] = 24.9
    assert roi_from_reflectivity(cref, 25.0, 1000.0, 1000.0) is None


def test_sensitivity_without_roi_is_nan_not_an_error():
    u, v = random_wind()
    out = sensitivity(partition_wind(u, v, DX, DY), u, v, DX, DY, None)
    assert np.isnan(out["bracket_a_b_raw"])
    assert not out["crop_evaluated"].any()


def test_crops_that_cut_the_roi_are_not_evaluated():
    u, v = random_wind(200, 200)
    p = partition_wind(u, v, 1000.0, 1000.0)
    # Fits the 10% crop (20..180) only.
    out = sensitivity(p, u, v, 1000.0, 1000.0, ROI(25, 175, 25, 175))
    assert out["crop_evaluated"].tolist() == [1, 0, 0]
    assert np.isfinite(out["crop_change_vpsi"][0])
    assert np.isnan(out["crop_change_vpsi"][1:]).all()


def test_uniform_background_only_affects_raw_bracket():
    psi = corner_gaussian(0.5, 0.5, 6000.0, 200, 200)
    u, v = ops.rotational_wind(psi, DX, DY)
    chi = center_gaussian(0.5, 0.5, 6000.0, 200, 200)
    uc, vc = ops.divergent_wind(chi, DX, DY)
    u, v = u + uc + 10.0, v + vc
    p = partition_wind(u, v, DX, DY)
    out = sensitivity(p, u, v, DX, DY, ROI(60, 140, 60, 140))
    assert out["bracket_a_b_mean_removed"] < 0.01
    assert out["bracket_a_b_raw"] > 20.0 * out["bracket_a_b_mean_removed"]
    assert out["bracket_flag"] == 0
    assert out["vh_roi_mean_u"] == pytest.approx(10.0, abs=1e-3)


def test_crop_of_compact_flow_changes_little():
    psi = corner_gaussian(0.5, 0.5, 6000.0, 200, 200)
    u, v = ops.rotational_wind(psi, DX, DY)
    p = partition_wind(u, v, DX, DY)
    out = sensitivity(p, u, v, DX, DY, ROI(70, 130, 70, 130))
    assert out["crop_evaluated"].all()
    assert (out["crop_change_vpsi"] < 1e-3).all()
    assert not out["crop_flag"].any()
