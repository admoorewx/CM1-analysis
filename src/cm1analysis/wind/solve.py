"""Zero-Dirichlet Poisson solvers for chi (cell centers) and psi (corners)."""

import numpy as np
import scipy.sparse as sp
from scipy.fft import dstn, idstn
from scipy.sparse.linalg import splu


def _eigenvalues(nmodes_x, nmodes_y, nx, ny, dx, dy):
    k = np.arange(1, nmodes_x + 1)
    ell = np.arange(1, nmodes_y + 1)
    lam_x = -(4.0 / dx**2) * np.sin(k * np.pi / (2.0 * nx)) ** 2
    lam_y = -(4.0 / dy**2) * np.sin(ell * np.pi / (2.0 * ny)) ** 2
    return lam_x + lam_y[:, np.newaxis]


def solve_chi(delta, dx, dy, workers=-1):
    """Solve lap(chi) = delta at cell centers using a type-II DST."""
    ny, nx = delta.shape
    lam = _eigenvalues(nx, ny, nx, ny, dx, dy)
    spectrum = dstn(delta, type=2, norm="ortho", workers=workers)
    return idstn(spectrum / lam, type=2, norm="ortho", workers=workers)


def solve_psi(zeta, dx, dy, workers=-1):
    """Solve lap(psi) = zeta at interior corners using a type-I DST.

    zeta has shape (ny-1, nx-1). The result has shape (ny+1, nx+1) and is
    zero on the outer ring.
    """
    ny, nx = zeta.shape[0] + 1, zeta.shape[1] + 1
    lam = _eigenvalues(nx - 1, ny - 1, nx, ny, dx, dy)
    spectrum = dstn(zeta, type=1, norm="ortho", workers=workers)
    psi = np.zeros((ny + 1, nx + 1), dtype=np.float64)
    psi[1:-1, 1:-1] = idstn(spectrum / lam, type=1, norm="ortho", workers=workers)
    return psi


def _second_difference(n, h, ghost_cell):
    main = -2.0 * np.ones(n)
    if ghost_cell:
        main[[0, -1]] = -3.0
    off = np.ones(n - 1)
    return sp.diags([off, main, off], [-1, 0, 1], format="csc") / h**2


def _laplacian(nx, ny, dx, dy, ghost_cell):
    tx = _second_difference(nx, dx, ghost_cell)
    ty = _second_difference(ny, dy, ghost_cell)
    return sp.kron(sp.identity(ny), tx, format="csc") + sp.kron(ty, sp.identity(nx), format="csc")


def solve_chi_sparse(delta, dx, dy):
    """Sparse direct reference for solve_chi."""
    ny, nx = delta.shape
    matrix = _laplacian(nx, ny, dx, dy, ghost_cell=True)
    return splu(matrix).solve(delta.ravel()).reshape(ny, nx)


def solve_psi_sparse(zeta, dx, dy):
    """Sparse direct reference for solve_psi."""
    ny, nx = zeta.shape[0] + 1, zeta.shape[1] + 1
    matrix = _laplacian(nx - 1, ny - 1, dx, dy, ghost_cell=False)
    psi = np.zeros((ny + 1, nx + 1), dtype=np.float64)
    psi[1:-1, 1:-1] = splu(matrix).solve(zeta.ravel()).reshape(ny - 1, nx - 1)
    return psi
