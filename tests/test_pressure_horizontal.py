"""Horizontal-force checks for HDPGF and HBPGF."""

import numpy as np
import pytest
from test_pressure_physics import (
    DX,
    INVARIANT_VARIANTS,
    NX,
    NY,
    NZ,
    RADIUS,
    V0,
    flow_inputs,
    make_inputs,
    smooth_random,
    stretched,
    vortex_case,
)

from cm1analysis.pressure import hforce, vforce
from cm1analysis.pressure.grid import Box, Grid


def whole(shape):
    return Box(0, shape[2], 0, shape[1])


def horizontal(dec, inputs):
    return hforce.forces(dec.p_d, dec.p_b, whole(inputs.shape), inputs.grid, dec.setup.rho_c, dec.setup.kc)


def finite_scale(a):
    return float(np.nanmax(np.abs(a)))


@pytest.fixture
def grid():
    return Grid(DX, DX, stretched(NZ))


@pytest.mark.parametrize("variant", INVARIANT_VARIANTS)
def test_layered_buoyancy_gives_no_horizontal_force(grid, variant):
    """b(z) only and V = 0: HBPGF = 0 and HDPGF = 0."""
    layered = 2.0 * np.sin(grid.zc / 3000.0)[:, None, None] * np.ones((NZ, NY, NX))
    inputs = make_inputs(grid, thpert=layered)
    h = horizontal(vforce.decompose(inputs, variant), inputs)
    b_scale = 9.81 * 2.0 / 300.0
    for c in "xy":
        assert finite_scale(h[f"hbpgf_{c}"]) < 1e-10 * b_scale
        assert finite_scale(h[f"hdpgf_{c}"]) == 0.0


@pytest.mark.parametrize("variant", INVARIANT_VARIANTS)
def test_hbpgf_ignores_layered_part_of_b(grid, variant):
    """Adding f(z) to b changes p_b only by a function of z, so HBPGF must not change."""
    blobs = smooth_random(grid, 4)
    layered = (1.5 * np.cos(grid.zc / 2500.0))[:, None, None] * np.ones((NZ, NY, NX))
    base_in, shifted_in = make_inputs(grid, thpert=blobs), make_inputs(grid, thpert=blobs + layered)
    base = horizontal(vforce.decompose(base_in, variant), base_in)
    shifted = horizontal(vforce.decompose(shifted_in, variant), shifted_in)
    for c in "xy":
        ref = base[f"hbpgf_{c}"]
        assert finite_scale(shifted[f"hbpgf_{c}"] - ref) < 1e-10 * finite_scale(ref)


def test_face_force_matches_a_known_gradient(grid):
    """p = a x + c y on a constant density gives -(a, c) / rho at every interior face."""
    z, y, x = np.meshgrid(grid.zc, (np.arange(NY) + 0.5) * DX, (np.arange(NX) + 0.5) * DX, indexing="ij")
    rho = np.full(NZ, 1.2)
    phi = 3e-3 * x - 2e-3 * y
    roi = Box(5, NX - 5, 4, NY - 4)
    fx = hforce.face_force(phi, roi, 2, DX, rho)
    fy = hforce.face_force(phi, roi, 1, DX, rho)
    assert fx.shape == (NZ, roi.ny, roi.nx + 1) and fy.shape == (NZ, roi.ny + 1, roi.nx)
    assert np.allclose(fx, -3e-3 / 1.2, rtol=1e-12) and np.allclose(fy, 2e-3 / 1.2, rtol=1e-12)
    # On the crop edge the outer neighbour is missing, so the edge face is NaN.
    edge = hforce.face_force(phi, whole(phi.shape), 2, DX, rho)
    assert np.isnan(edge[..., 0]).all() and np.isnan(edge[..., -1]).all()
    assert np.isfinite(edge[..., 1:-1]).all()


# ── Cold pool ─────────────────────────────────────────────────────────────────

POOL_N, POOL_R, POOL_DEPTH = 96, 8.0e3, 1500.0


def test_cold_pool_pushes_outward():
    """A Gaussian cold pool with V = 0 gives a pure HBPGF directed away from its center."""
    grid = Grid(DX, DX, stretched(NZ))
    n = POOL_N
    z, y, x = np.meshgrid(grid.zc, (np.arange(n) + 0.5) * DX, (np.arange(n) + 0.5) * DX, indexing="ij")
    center = 0.5 * n * DX  # Lies on face n/2 in x and y; 40 km from every crop edge.
    r2 = (x - center) ** 2 + (y - center) ** 2
    thpert = -6.0 * np.exp(-r2 / POOL_R**2 - (z / POOL_DEPTH) ** 2)
    inputs = make_inputs(grid, thpert=thpert, nx=n, ny=n)
    dec = vforce.decompose(inputs, "A")
    h = horizontal(dec, inputs)
    assert finite_scale(h["hdpgf_x"]) == 0.0 and finite_scale(h["hdpgf_y"]) == 0.0

    mid = n // 2
    assert dec.p_b[0, mid, mid] > dec.p_b[0, mid, 2]  # High pressure under the pool.
    row = (mid - 1, mid)  # The two cell rows either side of the center line.
    for cells in (3, 6, 10, 15):
        east = h["hbpgf_x"][0, row, mid + cells].mean()
        west = h["hbpgf_x"][0, row, mid - cells].mean()
        north = h["hbpgf_y"][0, mid + cells, row].mean()
        south = h["hbpgf_y"][0, mid - cells, row].mean()
        assert east > 0.0 and west < 0.0 and north > 0.0 and south < 0.0, cells
        assert east == pytest.approx(-west, rel=1e-6)  # Symmetric about the center.


# ── Cyclostrophic vortex ──────────────────────────────────────────────────────


def test_vortex_hdpgf_balances_centrifugal_force():
    """At the radius of maximum wind the inward HDPGF balances v^2 / r."""
    inputs, _ = vortex_case(500.0)
    dec = vforce.decompose(inputs, "A")
    h = horizontal(dec, inputs)
    n = inputs.shape[2]
    mid = n // 2  # The center is on face mid; cell rows mid - 1 and mid straddle it.
    step = int(round(RADIUS / 500.0))
    expected = V0**2 / RADIUS
    rows = (mid - 1, mid)
    east = h["hdpgf_x"][:, rows, mid + step].mean(axis=1)
    west = h["hdpgf_x"][:, rows, mid - step].mean(axis=1)
    north = h["hdpgf_y"][:, mid + step, rows].mean(axis=1)
    south = h["hdpgf_y"][:, mid - step, rows].mean(axis=1)
    for force, sign in ((east, -1.0), (west, 1.0), (north, -1.0), (south, 1.0)):
        assert np.allclose(force, sign * expected, rtol=0.02), (force, expected)
    assert finite_scale(h["hbpgf_x"]) == 0.0


# ── Full analysis ─────────────────────────────────────────────────────────────


def test_analyze_writes_horizontal_fields_and_checks(grid):
    inputs = flow_inputs(grid)
    inputs.prspert = 50.0 * smooth_random(grid, 8)
    roi = Box(12, NX - 12, 10, NY - 10)
    fields, diag, _ = vforce.analyze(
        inputs, roi, kmax=NZ - 4, horizontal=True, along_wind=True, log=lambda *_: None
    )
    assert vforce.failed_checks(diag) == []
    for name in ("hdpgf_x", "hbpgf_x", "model_hpgf_x"):
        assert fields[name].shape == (NZ, roi.ny, roi.nx + 1)
        assert np.isfinite(fields[name]).all()  # The buffer supplies every outer neighbour.
    for name in ("hdpgf_y", "hbpgf_y", "model_hpgf_y"):
        assert fields[name].shape == (NZ, roi.ny + 1, roi.nx)
    assert fields["hdpgf_s"].shape == (NZ, roi.ny, roi.nx)

    for name in vforce.VARIANTS:
        assert np.isfinite(diag[f"hcons_{name}_rel_rms"]), name
    for name in vforce.VARIANTS[1:]:
        assert np.isfinite(diag[f"sens_hdpgf_{name}"]) and np.isfinite(diag[f"sens_hbpgf_{name}"])
    # Variant D sets p_d = p' - p_b, so it matches the model by construction (a bound).
    assert diag["hcons_D_rel_rms"] < 1e-12
    assert diag["sens_hbpgf_D"] == 0.0
    assert np.isfinite(diag["frame_hdpgf_rel"])
    for key in ("crop_hdpgf_rel", "crop_hbpgf_rel", "crop_hcons_rel_rms"):
        assert np.all(np.isfinite(diag[key])), key


def test_along_wind_projection():
    fx = np.full((1, 1, 2), 2.0)
    fy = np.full((1, 2, 1), -1.0)
    uc, vc = np.array([[[3.0]]]), np.array([[[4.0]]])
    assert hforce.along_wind(fx, fy, uc, vc)[0, 0, 0] == pytest.approx((3.0 * 2.0 - 4.0) / 5.0)
    calm = hforce.along_wind(fx, fy, 0.05 * uc, 0.05 * vc)  # 0.25 m/s, below MIN_SPEED.
    assert np.isnan(calm).all()
