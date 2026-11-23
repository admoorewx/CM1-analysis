"""Known-answer tests using winds sampled directly from closed-form expressions.

Nothing here builds its input with cm1analysis.wind.ops, so a mistake shared by the
forward operators and the solver cannot cancel. Each velocity is written out by
hand and sampled at the C-grid face positions.

The fields are not discretely nondivergent or irrotational, so the partition
only matches them to truncation error. The tests therefore require the error to
shrink at second order as the grid is refined.
"""

from dataclasses import dataclass

import numpy as np
import pytest

from cm1analysis.wind.partition import partition_wind

# A rectangular, anisotropic domain catches any swapped axis or spacing.
LX, LY = 96.0e3, 48.0e3
BASE_NX, BASE_NY = 96, 64
REFINEMENTS = (1, 2, 4)
SIGMA = 3000.0  # LY / 2 is 8 sigma, so the Gaussians are ~1e-14 on the boundary.
MIN_ORDER = 1.8
FINEST_ERROR_LIMIT = 5e-3
ROUNDOFF = 1e-10


@dataclass
class Grid:
    nx: int
    ny: int

    def __post_init__(self):
        self.dx, self.dy = LX / self.nx, LY / self.ny
        x_faces, x_centers = np.arange(self.nx + 1) * self.dx, (np.arange(self.nx) + 0.5) * self.dx
        y_faces, y_centers = np.arange(self.ny + 1) * self.dy, (np.arange(self.ny) + 0.5) * self.dy
        self.u_points = np.meshgrid(x_faces, y_centers)
        self.v_points = np.meshgrid(x_centers, y_faces)
        self.corners = np.meshgrid(x_faces, y_faces)
        self.centers = np.meshgrid(x_centers, y_centers)

    def sample(self, fields):
        """Sum velocity functions at the faces; an empty list gives zero wind."""
        u = np.zeros(self.u_points[0].shape)
        v = np.zeros(self.v_points[0].shape)
        for field in fields:
            u += field(*self.u_points)[0]
            v += field(*self.v_points)[1]
        return u, v


def refined(r):
    return Grid(BASE_NX * r, BASE_NY * r)


# ── Closed-form velocity fields, each returning (u, v) at (x, y) ───────────────


def vortex(omega, x0=0.5 * LX, y0=0.5 * LY):
    """Counterclockwise for omega > 0, with psi = -omega sigma^2 exp(-r^2 / 2 sigma^2)."""

    def field(x, y):
        a, b = x - x0, y - y0
        decay = np.exp(-(a**2 + b**2) / (2 * SIGMA**2))
        return -omega * b * decay, omega * a * decay

    return field


def outflow(rate, x0=0.5 * LX, y0=0.5 * LY):
    """Radially outward for rate > 0, with chi = -rate sigma^2 exp(-r^2 / 2 sigma^2)."""

    def field(x, y):
        a, b = x - x0, y - y0
        decay = np.exp(-(a**2 + b**2) / (2 * SIGMA**2))
        return rate * a * decay, rate * b * decay

    return field


def uniform(u0, v0):
    return lambda x, y: (np.full(x.shape, u0), np.full(x.shape, v0))


def harmonic_exponential(speed, k=2 * np.pi / LX, x0=0.4 * LX, y0=0.3 * LY):
    """Gradient of phi = (speed / k) exp(k (x - x0)) cos(k (y - y0)), a harmonic function."""

    def field(x, y):
        grow = speed * np.exp(k * (x - x0))
        return grow * np.cos(k * (y - y0)), -grow * np.sin(k * (y - y0))

    return field


def harmonic_polynomial(a, b, c):
    """Gradient of phi = a (x^2 - y^2) / 2 + b x y + c (x^3 - 3 x y^2), a harmonic function."""

    def field(x, y):
        return a * x + b * y + 3 * c * (x**2 - y**2), -a * y + b * x - 6 * c * x * y

    return field


def sine_psi(amplitude, m, n):
    """Rotational wind from psi = amplitude sin(m pi x / LX) sin(n pi y / LY), zero on the edges."""
    kx, ky = m * np.pi / LX, n * np.pi / LY

    def field(x, y):
        return (
            -amplitude * ky * np.sin(kx * x) * np.cos(ky * y),
            amplitude * kx * np.cos(kx * x) * np.sin(ky * y),
        )

    return field


def sine_chi(amplitude, m, n):
    """Divergent wind from chi = amplitude sin(m pi x / LX) sin(n pi y / LY), zero on the edges."""
    kx, ky = m * np.pi / LX, n * np.pi / LY

    def field(x, y):
        return (
            amplitude * kx * np.cos(kx * x) * np.sin(ky * y),
            amplitude * ky * np.sin(kx * x) * np.cos(ky * y),
        )

    return field


# ── Signs and physical values ──────────────────────────────────────────────────


def test_counterclockwise_vortex_has_positive_vorticity_and_negative_psi():
    omega = 5e-3
    grid = refined(4)
    p = partition_wind(*grid.sample([vortex(omega)]), grid.dx, grid.dy)

    # Corner (ny/2, nx/2) is the vortex center; zeta omits the outer ring of corners.
    assert p.zeta[grid.ny // 2 - 1, grid.nx // 2 - 1] == pytest.approx(2 * omega, rel=FINEST_ERROR_LIMIT)
    assert p.psi[grid.ny // 2, grid.nx // 2] == pytest.approx(-omega * SIGMA**2, rel=FINEST_ERROR_LIMIT)

    x, y = grid.corners
    psi_true = -omega * SIGMA**2 * np.exp(-((x - 0.5 * LX) ** 2 + (y - 0.5 * LY) ** 2) / (2 * SIGMA**2))
    assert np.abs(p.psi - psi_true).max() < FINEST_ERROR_LIMIT * np.abs(psi_true).max()


def test_radial_outflow_has_positive_divergence_and_negative_chi():
    rate = 5e-3
    grid = refined(4)
    p = partition_wind(*grid.sample([outflow(rate)]), grid.dx, grid.dy)

    # The outflow center is a corner, so the cell to its north-east is half a cell off.
    assert p.delta[grid.ny // 2, grid.nx // 2] == pytest.approx(2 * rate, rel=FINEST_ERROR_LIMIT)

    x, y = grid.centers
    chi_true = -rate * SIGMA**2 * np.exp(-((x - 0.5 * LX) ** 2 + (y - 0.5 * LY) ** 2) / (2 * SIGMA**2))
    assert p.chi[grid.ny // 2, grid.nx // 2] < 0.0
    assert np.abs(p.chi - chi_true).max() < FINEST_ERROR_LIMIT * np.abs(chi_true).max()


# ── Harmonic flow that the discrete operators annihilate exactly ───────────────


@pytest.mark.parametrize(
    "field",
    [
        harmonic_polynomial(a=2e-4, b=0.0, c=0.0),
        harmonic_polynomial(a=0.0, b=2e-4, c=0.0),
        harmonic_polynomial(a=2e-4, b=-1e-4, c=1e-9),
    ],
    ids=["strain", "shear_strain", "cubic_potential"],
)
def test_polynomial_harmonic_flow_is_all_harmonic(field):
    grid = refined(1)
    u, v = grid.sample([field, uniform(3.0, -1.0)])
    p = partition_wind(u, v, grid.dx, grid.dy)
    scale = max(np.abs(u).max(), np.abs(v).max())

    assert np.ptp(u) > 0.1 * scale  # The flow varies in space, not a uniform wind.
    assert max(np.abs(p.u_psi).max(), np.abs(p.v_psi).max()) < ROUNDOFF * scale
    assert max(np.abs(p.u_chi).max(), np.abs(p.v_chi).max()) < ROUNDOFF * scale
    assert max(np.abs(p.u_h - u).max(), np.abs(p.v_h - v).max()) < ROUNDOFF * scale


# ── Each part converges to its known answer ────────────────────────────────────

# Parts are lists of velocity functions. All of these have psi and chi equal to
# zero on the boundary apart from the harmonic part, so the partition is unique.
OFFSET_VORTEX = vortex(5e-3, x0=0.3 * LX, y0=0.55 * LY)
OFFSET_OUTFLOW = outflow(-4e-3, x0=0.7 * LX, y0=0.45 * LY)

CASES = {
    "vortex": {"psi": [vortex(5e-3)], "chi": [], "h": []},
    "outflow": {"psi": [], "chi": [outflow(5e-3)], "h": []},
    "varying_harmonic": {"psi": [], "chi": [], "h": [harmonic_exponential(1.0)]},
    "sine_modes": {"psi": [sine_psi(2e5, 2, 3)], "chi": [sine_chi(1.5e5, 3, 1)], "h": []},
    "compact_mix": {
        "psi": [OFFSET_VORTEX],
        "chi": [OFFSET_OUTFLOW],
        "h": [uniform(12.5, -3.0), harmonic_exponential(1.0)],
    },
    # Harmonic flow crosses every edge here, the situation expected in real data.
    "sine_modes_with_boundary_flow": {
        "psi": [sine_psi(2e5, 2, 3)],
        "chi": [sine_chi(1.5e5, 3, 1)],
        "h": [uniform(2.5, 1.0), harmonic_exponential(1.0)],
    },
}


def partition_errors(case, r):
    """Return each part's largest wind error, relative to the largest input wind."""
    grid = refined(r)
    u, v = grid.sample(case["psi"] + case["chi"] + case["h"])
    p = partition_wind(u, v, grid.dx, grid.dy)
    scale = max(np.abs(u).max(), np.abs(v).max())

    errors = {}
    for name, u_part, v_part in (
        ("psi", p.u_psi, p.v_psi),
        ("chi", p.u_chi, p.v_chi),
        ("h", p.u_h, p.v_h),
    ):
        u_true, v_true = grid.sample(case[name])
        errors[name] = max(np.abs(u_part - u_true).max(), np.abs(v_part - v_true).max()) / scale
    return errors


@pytest.mark.parametrize("name", CASES)
def test_parts_converge_to_known_answer(name):
    worst = [max(partition_errors(CASES[name], r).values()) for r in REFINEMENTS]
    order = np.log2(worst[-2] / worst[-1])
    assert worst[-1] < FINEST_ERROR_LIMIT, f"errors by refinement: {worst}"
    assert order > MIN_ORDER, f"order {order:.2f}, errors by refinement: {worst}"


def test_mixed_field_parts_are_not_trivially_small():
    """Guards the mixed cases: each part must carry real wind for the test to mean anything."""
    grid = refined(1)
    for fields in (CASES["compact_mix"]["psi"], CASES["compact_mix"]["chi"], CASES["compact_mix"]["h"]):
        u, v = grid.sample(fields)
        assert max(np.abs(u).max(), np.abs(v).max()) > 5.0
