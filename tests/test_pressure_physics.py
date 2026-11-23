"""Known-answer tests of the decomposition.

Inputs are written out from closed-form expressions, not built with the
package's own operators, so a shared mistake cannot cancel.
"""

import numpy as np
import pytest

from cm1analysis.pressure import forcing, vforce
from cm1analysis.pressure.grid import Box, Grid

NX, NY, NZ = 48, 40, 24
DX = DY = 1000.0
INVARIANT_VARIANTS = ("A", "B", "H")


def stretched(nz, top=16000.0):
    xi = np.linspace(0.0, 1.0, nz + 1)
    return top * (xi - 0.35 * np.sin(np.pi * xi) / np.pi)


def make_base(grid):
    z = grid.zc
    return forcing.BaseState(grid, 300.0 + 4e-3 * z, 0.014 * np.exp(-z / 2500.0), 1.0e5 * np.exp(-z / 8500.0))


def make_inputs(grid, uc=None, vc=None, wc=None, thpert=None, qvpert=None, qw=None, nx=NX, ny=NY):
    shape = (grid.nz, ny, nx)
    zero = np.zeros(shape)
    base = make_base(grid)
    rho_model = np.broadcast_to(base.rho[:, None, None], shape).copy()
    return vforce.Inputs(
        grid,
        zero.copy() if uc is None else uc,
        zero.copy() if vc is None else vc,
        zero.copy() if wc is None else wc,
        zero.copy() if thpert is None else thpert,
        zero.copy() if qvpert is None else qvpert,
        zero.copy() if qw is None else qw,
        base,
        prspert=zero.copy(),
        rho_model=rho_model,
    )


def coords(grid, nx=NX, ny=NY):
    x = (np.arange(nx) + 0.5) * grid.dx
    y = (np.arange(ny) + 0.5) * grid.dy
    return np.meshgrid(grid.zc, y, x, indexing="ij")


def smooth_random(grid, seed, nx=NX, ny=NY):
    """Sum of a few Gaussian blobs, so the test does not depend on grid noise."""
    rng = np.random.default_rng(seed)
    z, y, x = coords(grid, nx, ny)
    out = np.zeros_like(z)
    for _ in range(4):
        x0, y0 = rng.uniform(0.3, 0.7) * nx * grid.dx, rng.uniform(0.3, 0.7) * ny * grid.dy
        z0, amp = rng.uniform(1000.0, 8000.0), rng.normal()
        out += amp * np.exp(-(((x - x0) / 6e3) ** 2 + ((y - y0) / 6e3) ** 2 + ((z - z0) / 2e3) ** 2))
    return out


def scale(a):
    return float(np.max(np.abs(a)))


@pytest.fixture
def grid():
    return Grid(DX, DY, stretched(NZ))


@pytest.mark.parametrize("variant", INVARIANT_VARIANTS)
def test_layered_buoyancy_is_cancelled(grid, variant):
    """b(z) only and V = 0: buoyancy force = -b, B = 0 and p_d = 0."""
    layered = 2.0 * np.sin(grid.zc / 3000.0)[:, None, None] * np.ones((NZ, NY, NX))
    dec = vforce.decompose(make_inputs(grid, thpert=layered), variant)
    assert scale(dec.bpgf + dec.b_face) < 1e-10 * scale(dec.b_face)
    assert scale(dec.B) < 1e-10 * scale(dec.b_face)
    assert scale(dec.p_d) == 0.0
    assert scale(dec.dvpgf) == 0.0


@pytest.mark.parametrize("variant", INVARIANT_VARIANTS)
def test_effective_buoyancy_ignores_layered_part(grid, variant):
    """Adding f(z) to b leaves B unchanged with the default boundary conditions."""
    blobs = smooth_random(grid, 4)
    layered = (1.5 * np.cos(grid.zc / 2500.0))[:, None, None] * np.ones((NZ, NY, NX))
    base_case = vforce.decompose(make_inputs(grid, thpert=blobs), variant)
    shifted = vforce.decompose(make_inputs(grid, thpert=blobs + layered), variant)
    assert scale(shifted.B - base_case.B) < 1e-10 * scale(base_case.B)


def test_dirichlet_lateral_condition_is_not_base_state_invariant(grid):
    """Variant C pins p_b = 0 on the sides, so a layered b leaks into B (documented)."""
    blobs = smooth_random(grid, 4)
    layered = (1.5 * np.cos(grid.zc / 2500.0))[:, None, None] * np.ones((NZ, NY, NX))
    base_case = vforce.decompose(make_inputs(grid, thpert=blobs), "C")
    shifted = vforce.decompose(make_inputs(grid, thpert=blobs + layered), "C")
    assert scale(shifted.B - base_case.B) > 1e-3 * scale(base_case.B)


# ── Cyclostrophic vortex ───────────────────────────────────────────────────────

RADIUS, V0, RHO = 10.0e3, 20.0, 1.1


def vortex_case(spacing):
    length = 16.0 * RADIUS
    n = int(round(length / spacing))
    grid = Grid(spacing, spacing, stretched(6, top=6000.0))
    xf, xc = np.arange(n + 1) * spacing, (np.arange(n) + 0.5) * spacing
    center = 0.5 * length

    def wind(x, y):
        r2 = (x - center) ** 2 + (y - center) ** 2
        speed_over_r = V0 / RADIUS * np.exp(0.5 * (1.0 - r2 / RADIUS**2))
        return -speed_over_r * (y - center), speed_over_r * (x - center)

    xu, yu = np.meshgrid(xf, xc)
    xv, yv = np.meshgrid(xc, xf)
    u = np.broadcast_to(wind(xu, yu)[0], (grid.nz, n, n + 1)).copy()
    v = np.broadcast_to(wind(xv, yv)[1], (grid.nz, n + 1, n)).copy()
    w = np.zeros((grid.nz + 1, n, n))
    zero = np.zeros((grid.nz, n, n))
    base = make_base(grid)
    base.rho, base.rho_f = np.full(grid.nz, RHO), np.full(grid.nz + 1, RHO)
    inputs = vforce.from_cgrid(grid, u, v, w, thpert=zero, qvpert=zero, qw=zero, base=base)

    xs, ys = np.meshgrid(xc, xc)
    r2 = (xs - center) ** 2 + (ys - center) ** 2
    # dp/dr = rho v^2 / r integrates to p = -rho V0^2 e / 2 exp(-r^2 / R^2).
    exact = -RHO * V0**2 * np.e / 2.0 * np.exp(-r2 / RADIUS**2)
    return inputs, exact


def test_cyclostrophic_vortex_error_falls_fourfold():
    errors = []
    for spacing in (2000.0, 1000.0, 500.0):
        inputs, exact = vortex_case(spacing)
        p = vforce.decompose(inputs, "A").p_d
        for level in p:
            assert scale(level - p[0]) < 1e-9 * scale(p[0])  # z-independent, as it should be.
        error = (p[0] - p[0].mean()) - (exact - exact.mean())
        errors.append(scale(error) / scale(exact))
    ratios = np.array(errors[:-1]) / np.array(errors[1:])
    assert np.all(ratios > 3.5), (errors, ratios)
    assert errors[-1] < 0.01


# ── Parts sum and the other variants ───────────────────────────────────────────


def flow_inputs(grid):
    z, y, x = coords(grid)
    uc = 10.0 * np.tanh((z - 3000.0) / 2000.0) + 3.0 * smooth_random(grid, 1)
    vc = 3.0 * smooth_random(grid, 2)
    wc = 8.0 * smooth_random(grid, 3) * np.sin(np.pi * z / 16000.0)
    thpert = smooth_random(grid, 5)
    qvpert = 1e-3 * smooth_random(grid, 6)
    qw = 2e-3 * np.abs(smooth_random(grid, 7))
    return make_inputs(grid, uc, vc, wc, thpert, qvpert, qw)


def identity(a):
    return a


@pytest.mark.parametrize("variant", ["A", "B", "C", "H"])
def test_parts_sum_to_the_whole(grid, variant):
    inputs = flow_inputs(grid)
    dec = vforce.decompose(inputs, variant)
    _, error = vforce.split_buoyancy(inputs, dec, identity)
    assert error < 1e-10
    for kind in ("shear", "strain"):
        _, error = vforce.split_dynamic(inputs, dec, kind, identity)
        assert error < 1e-10


def test_split_sources_match_the_full_forcing_for_constant_density(grid):
    """With constant rho and discretely consistent winds the remainder is small."""
    inputs = flow_inputs(grid)
    rho = np.full(grid.nz, 1.0)
    sources = forcing.velocity_gradient_sources(
        inputs.uc, inputs.vc, inputs.wc, rho, grid, ("shear", "strain")
    )
    total_shear = sources["shear"] + sources["extension"]
    total_strain = sources["rotation"] + sources["strain"] + sources["density"]
    assert scale(total_shear - total_strain) < 1e-10 * scale(total_shear)


def test_variant_d_reproduces_model_pressure(grid):
    inputs = flow_inputs(grid)
    inputs.prspert = 50.0 * smooth_random(grid, 8)
    dec = vforce.decompose(inputs, "D")
    assert scale(dec.p_d + dec.p_b - inputs.prspert) < 1e-12 * scale(inputs.prspert)


def test_analyze_runs_every_variant_and_split(grid):
    inputs = flow_inputs(grid)
    inputs.prspert = 50.0 * smooth_random(grid, 8)
    roi = Box(12, NX - 12, 10, NY - 10)
    messages = []
    fields, diag, _ = vforce.analyze(inputs, roi, kmax=NZ - 4, splits=vforce.SPLITS, log=messages.append)
    assert vforce.failed_checks(diag) == []
    assert fields["dvpgf"].shape == (NZ + 1, roi.ny, roi.nx)
    assert fields["p_d"].shape == (NZ, roi.ny, roi.nx)
    for name in vforce.VARIANTS[1:]:
        assert np.isfinite(diag[f"sens_dvpgf_{name}"])
    assert diag["sens_B_D"] == 0.0  # D only changes p_d.
    assert np.all(diag["crop_buffer_cells"] > 0)
    assert np.all(np.isfinite(diag["crop_dvpgf_rel"]))
    assert "dvpgf_strain_remainder" in fields and "B_loading" in fields
