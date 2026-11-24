"""Poisson solver: horizontal transform and tridiagonal solve in z (Klemp and Rotunno 1983).

Solves, at cell centers,

    Kc(z) lap_h(phi) + (1/dz) d/dz (Kf(z) dphi/dz) = rhs

with zero flux through the top and bottom faces. Inhomogeneous Neumann
conditions are carried by the flux-form right-hand sides in forcing.py, so the
solver itself only needs the homogeneous form. Kc = Kf = 1 gives the plain
Laplacian of the Xu et al. (2015a) form; Kc = c_p rho theta gives the Exner
operator of Rotunno and Klemp (variant H).

Lateral conditions, with the eigenvalue of the 1-D second difference:

    periodic   FFT      -4 sin^2(pi k / n) / d^2
    neumann    DCT-II   -4 sin^2(pi k / 2n) / d^2        zero gradient on the edge faces
    dirichlet  DST-II   -4 sin^2(pi (k + 1) / 2n) / d^2  phi = 0 on the edge faces

Periodic and Neumann leave the horizontally uniform mode singular. Its forcing
is made solvable by removing the volume-mean of rhs (reported as the
solvability residual) and its constant is pinned by setting the lowest-level
horizontal mean of phi to zero.
"""

import numpy as np
from scipy.fft import dctn, dstn, idctn, idstn, irfft2, rfft2

from ..parallel import fft_workers

LATERAL_CONDITIONS = ("periodic", "neumann", "dirichlet")


def _check_lateral(lateral):
    if lateral not in LATERAL_CONDITIONS:
        raise ValueError(f"unknown lateral condition {lateral!r}; expected {LATERAL_CONDITIONS}")


def _eigenvalues(n, d, lateral, half=False):
    k = np.arange(n // 2 + 1 if half else n, dtype=np.float64)
    if lateral == "periodic":
        arg = np.pi * k / n
    elif lateral == "neumann":
        arg = np.pi * k / (2.0 * n)
    else:
        arg = np.pi * (k + 1.0) / (2.0 * n)
    return -(4.0 / d**2) * np.sin(arg) ** 2


def _forward(rhs, lateral, workers):
    if lateral == "periodic":
        return rfft2(rhs, axes=(1, 2), workers=workers)
    if lateral == "neumann":
        return dctn(rhs, type=2, axes=(1, 2), norm="ortho", workers=workers)
    return dstn(rhs, type=2, axes=(1, 2), norm="ortho", workers=workers)


def _inverse(hat, lateral, shape, workers):
    if lateral == "periodic":
        return irfft2(hat, s=shape[1:], axes=(1, 2), workers=workers)
    if lateral == "neumann":
        return idctn(hat, type=2, axes=(1, 2), norm="ortho", workers=workers)
    return idstn(hat, type=2, axes=(1, 2), norm="ortho", workers=workers)


def _coefficients(grid, kc, kf):
    kc = np.ones(grid.nz) if kc is None else np.asarray(kc, dtype=np.float64)
    kf = np.ones(grid.nz + 1) if kf is None else np.asarray(kf, dtype=np.float64)
    if kc.shape != (grid.nz,) or kf.shape != (grid.nz + 1,):
        raise ValueError("kc must be a center profile and kf a face profile")
    return kc, kf


def solve(rhs, grid, lateral="periodic", kc=None, kf=None, workers=-1):
    """Return (phi, info). info holds 'solvability' and 'mean_forcing_removed'."""
    _check_lateral(lateral)
    workers = fft_workers(workers)
    rhs = np.asarray(rhs, dtype=np.float64)
    nz, ny, nx = rhs.shape
    if nz != grid.nz:
        raise ValueError(f"rhs has {nz} levels but the grid has {grid.nz}")
    kc, kf = _coefficients(grid, kc, kf)

    singular = lateral != "dirichlet"
    level_sums = rhs.sum(axis=(1, 2))
    total = float(np.sum(level_sums * grid.dz))
    scale = float(np.sum(np.abs(rhs).sum(axis=(1, 2)) * grid.dz))
    correction = total / (nx * ny * grid.dz.sum()) if singular else 0.0
    info = {
        "solvability": abs(total) / scale if scale > 0.0 else 0.0,
        "mean_forcing_removed": correction,
    }

    hat = _forward(rhs, lateral, workers)
    lam = (
        _eigenvalues(ny, grid.dy, lateral)[:, np.newaxis]
        + _eigenvalues(nx, grid.dx, lateral, half=lateral == "periodic")[np.newaxis, :]
    )

    if singular:
        norm = nx * ny if lateral == "periodic" else np.sqrt(nx * ny)
        hat[:, 0, 0] -= correction * norm
        uniform = hat[:, 0, 0].copy()
        lam[0, 0] = -1.0  # Placeholder; the uniform mode is solved exactly below.

    coupling = kf[1:-1] / grid.dzf[1:-1]
    lower = np.zeros(nz)
    upper = np.zeros(nz)
    lower[1:] = coupling / grid.dz[1:]
    upper[:-1] = coupling / grid.dz[:-1]

    # Vectorized Thomas algorithm over all horizontal modes, in place in hat.
    cprime = np.empty((nz,) + lam.shape)
    denom = -upper[0] + kc[0] * lam
    cprime[0] = upper[0] / denom
    hat[0] /= denom
    for k in range(1, nz):
        denom = -(lower[k] + upper[k]) + kc[k] * lam - lower[k] * cprime[k - 1]
        cprime[k] = upper[k] / denom
        hat[k] -= lower[k] * hat[k - 1]
        hat[k] /= denom
    for k in range(nz - 2, -1, -1):
        hat[k] -= cprime[k] * hat[k + 1]
    del cprime

    if singular:
        # Integrate the uniform mode: flux G[k+1] = G[k] + r[k] dz[k], G[0] = 0.
        flux = np.cumsum(uniform * grid.dz)
        phi0 = np.zeros(nz, dtype=uniform.dtype)
        phi0[1:] = np.cumsum(flux[:-1] * grid.dzf[1:-1] / kf[1:-1])
        hat[:, 0, 0] = phi0

    return _inverse(hat, lateral, rhs.shape, workers), info


def _lap_h(f, dx, dy, lateral):
    if lateral == "periodic":
        gx = np.roll(f, 1, axis=1) + np.roll(f, -1, axis=1)
        gy = np.roll(f, 1, axis=0) + np.roll(f, -1, axis=0)
    else:
        sign = 1.0 if lateral == "neumann" else -1.0
        px = np.empty((f.shape[0], f.shape[1] + 2))
        px[:, 1:-1] = f
        px[:, 0], px[:, -1] = sign * f[:, 0], sign * f[:, -1]
        py = np.empty((f.shape[0] + 2, f.shape[1]))
        py[1:-1] = f
        py[0], py[-1] = sign * f[0], sign * f[-1]
        gx = px[:, 2:] + px[:, :-2]
        gy = py[2:] + py[:-2]
    return (gx - 2.0 * f) / dx**2 + (gy - 2.0 * f) / dy**2


def apply_operator(phi, grid, lateral="periodic", kc=None, kf=None):
    """Apply the discrete operator that solve() inverts, level by level."""
    _check_lateral(lateral)
    kc, kf = _coefficients(grid, kc, kf)
    out = np.empty_like(phi, dtype=np.float64)
    for k in range(grid.nz):
        out[k] = kc[k] * _lap_h(phi[k], grid.dx, grid.dy, lateral)
    for k in range(1, grid.nz):
        flux = kf[k] * (phi[k] - phi[k - 1]) / grid.dzf[k]
        out[k - 1] += flux / grid.dz[k - 1]
        out[k] -= flux / grid.dz[k]
    return out


def residual(phi, rhs, grid, lateral="periodic", kc=None, kf=None, mean_forcing_removed=0.0):
    """max |L phi - rhs| / max |rhs|, using the rhs actually solved."""
    solved = np.asarray(rhs, dtype=np.float64) - mean_forcing_removed
    scale = float(np.max(np.abs(solved)))
    if scale == 0.0:
        return float(np.max(np.abs(phi)) > 0.0)
    error = apply_operator(phi, grid, lateral, kc, kf)
    error -= solved
    return float(np.max(np.abs(error))) / scale
