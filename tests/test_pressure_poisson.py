"""Poisson solver: residual and manufactured-solution convergence on a stretched z grid."""

import numpy as np
import pytest

from cm1analysis.pressure import poisson
from cm1analysis.pressure.grid import Grid

LX, LY, H = 60.0e3, 40.0e3, 12.0e3
MIN_ORDER = 1.8


def stretched_faces(nz, strength=0.35):
    """Smooth stretching: fine near the ground, coarse aloft, like CM1 stretch_z."""
    xi = np.linspace(0.0, 1.0, nz + 1)
    return H * (xi - strength * np.sin(np.pi * xi) / np.pi) / 1.0


def make_grid(r, nz_base=12):
    nx, ny, nz = 16 * r, 12 * r, nz_base * r
    return Grid(LX / nx, LY / ny, stretched_faces(nz)), nx, ny


def centers(grid, nx, ny):
    x = (np.arange(nx) + 0.5) * grid.dx
    y = (np.arange(ny) + 0.5) * grid.dy
    return np.meshgrid(grid.zc, y, x, indexing="ij")


def kprofile(z):
    """A variable coefficient like c_p rho0 theta0, in arbitrary units."""
    return 1.0 + 0.5 * np.exp(-z / 8000.0)


def dkprofile(z):
    return -0.5 / 8000.0 * np.exp(-z / 8000.0)


CASES = {
    # Exact solution phi(x, y, z) with zero vertical derivative on the top and
    # bottom, and the horizontal shape matching each lateral condition. The
    # periodic and Neumann cases include a horizontally uniform part so the
    # singular mode is exercised.
    "periodic": (
        lambda x, y: np.cos(2 * np.pi * x / LX) * np.sin(2 * np.pi * y / LY),
        -((2 * np.pi / LX) ** 2 + (2 * np.pi / LY) ** 2),
        True,
    ),
    "neumann": (
        lambda x, y: np.cos(np.pi * x / LX) * np.cos(2 * np.pi * y / LY),
        -((np.pi / LX) ** 2 + (2 * np.pi / LY) ** 2),
        True,
    ),
    "dirichlet": (
        lambda x, y: np.sin(np.pi * x / LX) * np.sin(np.pi * y / LY),
        -((np.pi / LX) ** 2 + (np.pi / LY) ** 2),
        False,
    ),
}


def manufactured(grid, nx, ny, lateral, variable_k):
    shape_xy, lam, uniform = CASES[lateral]
    z, y, x = centers(grid, nx, ny)
    m = np.pi / H
    vertical, dvertical, d2vertical = np.cos(m * z), -m * np.sin(m * z), -(m**2) * np.cos(m * z)
    phi = shape_xy(x, y) * vertical
    if uniform:
        phi = phi + 0.7 * np.cos(2 * m * z)
    if variable_k:
        k, dk = kprofile(z), dkprofile(z)
    else:
        k, dk = np.ones_like(z), np.zeros_like(z)
    rhs = k * lam * shape_xy(x, y) * vertical + shape_xy(x, y) * (dk * dvertical + k * d2vertical)
    if uniform:
        two = 2 * m
        rhs = rhs + 0.7 * (dk * -two * np.sin(two * z) + k * -(two**2) * np.cos(two * z))
    kc = kprofile(grid.zc) if variable_k else None
    kf = kprofile(grid.zf) if variable_k else None
    return phi, rhs, kc, kf


def volume_anomaly(phi, grid):
    weights = grid.dz[:, None, None] * np.ones_like(phi)
    return phi - np.sum(phi * weights) / np.sum(weights)


@pytest.mark.parametrize("lateral", poisson.LATERAL_CONDITIONS)
@pytest.mark.parametrize("variable_k", [False, True])
def test_residual_is_roundoff(lateral, variable_k):
    grid, nx, ny = make_grid(2)
    rng = np.random.default_rng(1)
    rhs = rng.normal(size=(grid.nz, ny, nx))
    kc = kprofile(grid.zc) if variable_k else None
    kf = kprofile(grid.zf) if variable_k else None
    phi, info = poisson.solve(rhs, grid, lateral, kc, kf)
    res = poisson.residual(phi, rhs, grid, lateral, kc, kf, info["mean_forcing_removed"])
    assert res < 1e-10


@pytest.mark.parametrize("lateral", poisson.LATERAL_CONDITIONS)
@pytest.mark.parametrize("variable_k", [False, True])
def test_manufactured_solution_converges_at_second_order(lateral, variable_k):
    errors = []
    for r in (1, 2, 4):
        grid, nx, ny = make_grid(r)
        exact, rhs, kc, kf = manufactured(grid, nx, ny, lateral, variable_k)
        phi, info = poisson.solve(rhs, grid, lateral, kc, kf)
        if lateral == "dirichlet":
            diff = phi - exact
        else:
            diff = volume_anomaly(phi, grid) - volume_anomaly(exact, grid)
        errors.append(np.max(np.abs(diff)) / np.max(np.abs(exact)))
    orders = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert np.all(orders > MIN_ORDER), (errors, orders)
    assert errors[-1] < 5e-3


def test_singular_mode_reports_unsolvable_forcing():
    grid, nx, ny = make_grid(1)
    rhs = np.ones((grid.nz, ny, nx))
    _, info = poisson.solve(rhs, grid, "periodic")
    assert info["solvability"] == pytest.approx(1.0)
    _, info = poisson.solve(rhs, grid, "dirichlet")
    assert info["mean_forcing_removed"] == 0.0


def test_lowest_level_mean_is_pinned_to_zero():
    grid, nx, ny = make_grid(1)
    rng = np.random.default_rng(2)
    rhs = rng.normal(size=(grid.nz, ny, nx))
    for lateral in ("periodic", "neumann"):
        phi, _ = poisson.solve(rhs, grid, lateral)
        assert abs(phi[0].mean()) < 1e-12 * np.max(np.abs(phi))
