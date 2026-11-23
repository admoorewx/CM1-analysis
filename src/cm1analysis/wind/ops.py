"""C-grid difference operators for the wind partition.

Index convention (nx x ny scalar cells):
    u[j, i]    west face of cell (j, i), shape (ny, nx+1)
    v[j, i]    south face of cell (j, i), shape (ny+1, nx)
    chi[j, i]  cell center, shape (ny, nx)
    psi[j, i]  south-west corner of cell (j, i), shape (ny+1, nx+1)
"""

import numpy as np


def grid_shape(u, v):
    """Return (ny, nx) after checking that u and v have C-grid shapes."""
    if u.ndim != 2 or v.ndim != 2:
        raise ValueError(f"u and v must be 2-D, got {u.shape} and {v.shape}")
    ny, nx = u.shape[0], v.shape[1]
    if u.shape != (ny, nx + 1) or v.shape != (ny + 1, nx):
        raise ValueError(
            f"u {u.shape} and v {v.shape} do not fit a C-grid: expected u (ny, nx+1) and v (ny+1, nx)"
        )
    return ny, nx


def divergence(u, v, dx, dy):
    """Divergence at cell centers, shape (ny, nx)."""
    grid_shape(u, v)
    return (u[:, 1:] - u[:, :-1]) / dx + (v[1:, :] - v[:-1, :]) / dy


def vorticity(u, v, dx, dy):
    """Vertical vorticity at interior corners, shape (ny-1, nx-1)."""
    grid_shape(u, v)
    return (v[1:-1, 1:] - v[1:-1, :-1]) / dx - (u[1:, 1:-1] - u[:-1, 1:-1]) / dy


def rotational_wind(psi, dx, dy):
    """Return (u_psi, v_psi) = k x grad(psi) from psi on all corners."""
    return -(psi[1:, :] - psi[:-1, :]) / dy, (psi[:, 1:] - psi[:, :-1]) / dx


def divergent_wind(chi, dx, dy):
    """Return (u_chi, v_chi) = grad(chi), with chi = 0 on the boundary faces.

    The antisymmetric ghost cell (chi_ghost = -chi_edge) gives edge values of
    +/- 2 chi / dx and +/- 2 chi / dy.
    """
    ny, nx = chi.shape
    u_chi = np.empty((ny, nx + 1), dtype=np.float64)
    v_chi = np.empty((ny + 1, nx), dtype=np.float64)
    u_chi[:, 1:-1] = (chi[:, 1:] - chi[:, :-1]) / dx
    u_chi[:, 0] = 2.0 * chi[:, 0] / dx
    u_chi[:, -1] = -2.0 * chi[:, -1] / dx
    v_chi[1:-1, :] = (chi[1:, :] - chi[:-1, :]) / dy
    v_chi[0, :] = 2.0 * chi[0, :] / dy
    v_chi[-1, :] = -2.0 * chi[-1, :] / dy
    return u_chi, v_chi


def winds_to_centers(u, v):
    """Average face winds to cell centers, for ROI statistics and plotting."""
    return 0.5 * (u[:, 1:] + u[:, :-1]), 0.5 * (v[1:, :] + v[:-1, :])
