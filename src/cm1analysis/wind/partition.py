"""The Lynch (1989) partition algorithm and the A/B/C harmonic re-labeling variants."""

from dataclasses import dataclass

import numpy as np

from . import ops, solve


@dataclass
class Partition:
    psi: np.ndarray
    chi: np.ndarray
    u_psi: np.ndarray
    v_psi: np.ndarray
    u_chi: np.ndarray
    v_chi: np.ndarray
    u_h: np.ndarray
    v_h: np.ndarray
    delta: np.ndarray
    zeta: np.ndarray


def partition_wind(u, v, dx, dy, workers=-1):
    """Split (u, v) into rotational, divergent and harmonic parts.

    Uses psi = chi = 0 on the boundary, with the harmonic part as the residual
    (Lynch 1989). zeta holds interior corners only, shape (ny-1, nx-1).
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    ops.grid_shape(u, v)
    if not (np.all(np.isfinite(u)) and np.all(np.isfinite(v))):
        raise ValueError("u or v contains non-finite values")

    delta = ops.divergence(u, v, dx, dy)
    zeta = ops.vorticity(u, v, dx, dy)
    chi = solve.solve_chi(delta, dx, dy, workers=workers)
    psi = solve.solve_psi(zeta, dx, dy, workers=workers)
    u_psi, v_psi = ops.rotational_wind(psi, dx, dy)
    u_chi, v_chi = ops.divergent_wind(chi, dx, dy)
    return Partition(
        psi=psi,
        chi=chi,
        u_psi=u_psi,
        v_psi=v_psi,
        u_chi=u_chi,
        v_chi=v_chi,
        u_h=u - u_psi - u_chi,
        v_h=v - v_psi - v_chi,
        delta=delta,
        zeta=zeta,
    )


def apply_variant(p, variant):
    """Return (u_rot, v_rot, u_div, v_div) for variant A, B or C.

    A keeps the harmonic part separate, B adds it to the rotational part and
    C adds it to the divergent part.
    """
    if variant == "A":
        return p.u_psi, p.v_psi, p.u_chi, p.v_chi
    if variant == "B":
        return p.u_psi + p.u_h, p.v_psi + p.v_h, p.u_chi, p.v_chi
    if variant == "C":
        return p.u_psi, p.v_psi, p.u_chi + p.u_h, p.v_chi + p.v_h
    raise ValueError(f"unknown variant {variant!r}; expected A, B or C")
