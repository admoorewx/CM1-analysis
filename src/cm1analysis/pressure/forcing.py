"""Base state, buoyancy, advective forcing and Poisson right-hand sides.

Both Poisson problems are written in flux form, lap(p) = -div(F):

    dynamic    F = rho V.grad(V)          (cell centers, F_z averaged to z-faces)
    buoyancy   F = (0, 0, -rho b)         (z-faces, b averaged to the face)

The Neumann conditions of Klemp and Rotunno (1983) are dp/dn = -F.n, so a
Neumann boundary face contributes nothing to the right-hand side and the
solver only sees homogeneous conditions. On the top and bottom faces this gives
dp_d/dz = 0 (w = 0, so V.grad(w) = 0) and dp_b/dz = rho b, and the forcing is
exactly solvable: a layered b gives B = 0 to roundoff.
"""

import numpy as np

from .grid import column

# CM1 constants (cm1r18 include/constants.incl).
G = 9.81
RD = 287.04
RV = 461.5
CP = 1005.7
P00 = 1.0e5
REPS = RV / RD
VIRTUAL_COEFF = 0.61  # b = g (theta'/theta0 + 0.61 qv' - qw).


def exner(p):
    return (p / P00) ** (RD / CP)


def dry_density(p, th, qv):
    """CM1's density definition (base.F): p / (Rd theta pi (1 + qv Rv/Rd))."""
    return p / (RD * th * exner(p) * (1.0 + REPS * qv))


class BaseState:
    """Horizontally uniform reference profiles at cell centers, with rho at faces."""

    def __init__(self, grid, th, qv, p):
        self.grid = grid
        self.th = np.asarray(th, dtype=np.float64)
        self.qv = np.asarray(qv, dtype=np.float64)
        self.p = np.asarray(p, dtype=np.float64)
        for name in ("th", "qv", "p"):
            if getattr(self, name).shape != (grid.nz,):
                raise ValueError(f"base-state {name} must have {grid.nz} levels")
        self.rho = dry_density(self.p, self.th, self.qv)
        self.rho_f = grid.to_faces_log(self.rho)

    def hydrostatic_residuals(self):
        """Return (pressure form, Exner form) residuals, each relative to its gravity term.

        Pressure form: dp0/dz + g rho0 (1 + qv0) at interior faces.
        Exner form:    dpi0/dz + g / (c_p theta_v0), with theta_v0 averaged between
                       centers as CM1 does when it builds the base state (base.F).
        """
        g = self.grid
        rho_total = g.to_faces_log(self.rho * (1.0 + self.qv))[1:-1]
        dpdz = np.diff(self.p) / g.dzf[1:-1]
        p_form = np.max(np.abs(dpdz + G * rho_total)) / np.max(G * rho_total)
        thv = self.th * (1.0 + REPS * self.qv) / (1.0 + self.qv)
        gravity = G / (CP * 0.5 * (thv[1:] + thv[:-1]))
        dpidz = np.diff(exner(self.p)) / g.dzf[1:-1]
        exner_form = np.max(np.abs(dpidz + gravity)) / np.max(gravity)
        return float(p_form), float(exner_form)


def buoyancy_parts(th_pert, qv_pert, qw, th_ref):
    """Return (thermal, loading): g (theta'/theta0 + 0.61 qv') and -g qw."""
    thermal = th_pert / column(th_ref)
    thermal += VIRTUAL_COEFF * qv_pert
    thermal *= G
    return thermal, -G * qw


def density_buoyancy(rho_model, qv, qw, base):
    """Variant E: -g (rho - rho0) / rho0 with moist air plus condensate in both densities."""
    rho_total = rho_model * (1.0 + qv + qw)
    rho_ref = column(base.rho * (1.0 + base.qv))
    return -G * (rho_total - rho_ref) / rho_ref


def advective_acceleration(uc, vc, wc, grid, frame=(0.0, 0.0)):
    """Return (V.grad u, V.grad v, V.grad w) at cell centers.

    frame = (cx, cy) evaluates the horizontal advection in a frame moving at
    that velocity, for the reference-frame sensitivity test.
    """
    cx, cy = frame
    us = uc - cx if cx else uc
    vs = vc - cy if cy else vc
    out = []
    for q in (uc, vc, wc):
        a = us * grid.ddx(q)
        a += vs * grid.ddy(q)
        a += wc * grid.ddz(q)
        out.append(a)
    return tuple(out)


def _side(axis, index):
    idx = [slice(None)] * 3
    idx[axis] = index
    return tuple(idx)


def _add_horizontal_divergence(rhs, f, axis, d, lateral):
    """Subtract d(F)/dx in flux form, with F averaged from centers to faces."""
    faces = 0.5 * (f[_side(axis, slice(1, None))] + f[_side(axis, slice(None, -1))])
    faces /= d
    rhs[_side(axis, slice(None, -1))] -= faces
    rhs[_side(axis, slice(1, None))] += faces
    if lateral == "neumann":
        return  # dp/dn = -F.n: boundary faces drop out.
    if lateral == "periodic":
        west = east = 0.5 * (f[_side(axis, 0)] + f[_side(axis, -1)])  # Wrap-around face.
    else:
        west, east = f[_side(axis, 0)], f[_side(axis, -1)]
    rhs[_side(axis, 0)] += west / d
    rhs[_side(axis, -1)] -= east / d


def flux_divergence(grid, lateral, shape, fx=None, fy=None, fz=None):
    """Return -div(F) at cell centers.

    fx and fy are cell-centered; fz is on z-faces and only its interior faces
    are used, because the top and bottom are Neumann boundaries.
    """
    rhs = np.zeros(shape)
    if fx is not None:
        _add_horizontal_divergence(rhs, fx, 2, grid.dx, lateral)
    if fy is not None:
        _add_horizontal_divergence(rhs, fy, 1, grid.dy, lateral)
    if fz is not None:
        interior = fz[1:-1]
        rhs[:-1] -= interior / column(grid.dz[:-1])
        rhs[1:] += interior / column(grid.dz[1:])
    return rhs


def dynamic_rhs(acc, rho_c, grid, lateral):
    """-div(rho V.grad V) from the advective accelerations at centers."""
    ax, ay, az = acc
    rho = column(rho_c)
    fz = grid.to_faces(rho * az, "nearest")
    fz[0] = fz[-1] = 0.0
    return flux_divergence(grid, lateral, ax.shape, rho * ax, rho * ay, fz)


def buoyancy_rhs(b, rho_f, grid):
    """Return (d(rho b)/dz from z-face differences, b at z-faces)."""
    b_face = grid.to_faces(b, "nearest")
    flux = -column(rho_f) * b_face
    return flux_divergence(grid, "neumann", b.shape, fz=flux), b_face


def vertical_force(phi, grid, rho_f, kf=None, boundary=None):
    """-(Kf / rho) dphi/dz at z-faces.

    The boundary faces get the values in boundary (bottom, top), or zero. For
    p_b these are -b, because dp_b/dz = rho b there.
    """
    out = np.zeros((grid.nz + 1,) + phi.shape[1:])
    rho = column(rho_f)
    out[1:-1] = (phi[1:] - phi[:-1]) / column(grid.dzf[1:-1])
    out[1:-1] /= rho[1:-1]
    out[1:-1] *= -1.0 if kf is None else -column(kf)[1:-1]
    if boundary is not None:
        out[0], out[-1] = boundary
    return out


def model_exner_force(prspert, base, grid):
    """CM1's vertical pressure-gradient force -c_p theta_v0 dpi'/dz at z-faces (sound.F).

    pi' = pi(p0 + p') - pi(p0) and theta_v0 is averaged between centers, as in
    the model. The boundary faces are zero.
    """
    p0 = column(base.p)
    pi_pert = exner(p0 + prspert) - exner(p0)
    thv = base.th * (1.0 + REPS * base.qv) / (1.0 + base.qv)
    out = grid.ddz_faces(pi_pert)
    out[1:-1] *= column(-CP * 0.5 * (thv[1:] + thv[:-1]))
    return out


def velocity_gradient_sources(uc, vc, wc, rho_c, grid, kinds):
    """Source terms for the optional dynamic splits, at cell centers.

    kinds may contain "shear" (Rotunno and Klemp 1985):
        shear     -2 rho (v_x u_y + u_z w_x + v_z w_y)
        extension -rho (u_x^2 + v_y^2 + w_z^2 - (ln rho)'' w^2)
    and "strain" (Trapp and Weisman 2003):
        rotation  rho |omega|^2 / 2
        strain    -rho e_ij e_ij
        density   rho (ln rho)'' w^2
    """
    rho = column(rho_c)
    log_rho = np.log(rho_c) if np.ndim(rho_c) == 3 else column(np.log(rho_c))
    d2 = grid.ddz(grid.ddz(log_rho))
    shape = uc.shape
    want_shear, want_strain = "shear" in kinds, "strain" in kinds
    out = {}
    if want_shear:
        out["shear"], out["extension"] = np.zeros(shape), np.zeros(shape)
    if want_strain:
        out["rotation"], out["strain"] = np.zeros(shape), np.zeros(shape)

    for q, derivative in ((uc, grid.ddx), (vc, grid.ddy), (wc, grid.ddz)):
        square = derivative(q) ** 2
        if want_shear:
            out["extension"] -= square
        if want_strain:
            out["strain"] -= square
    pairs = (
        (uc, grid.ddy, vc, grid.ddx),  # u_y, v_x
        (uc, grid.ddz, wc, grid.ddx),  # u_z, w_x
        (vc, grid.ddz, wc, grid.ddy),  # v_z, w_y
    )
    for qa, da, qb, db in pairs:
        a, b = da(qa), db(qb)
        if want_shear:
            out["shear"] -= 2.0 * a * b
        if want_strain:
            out["rotation"] += 0.5 * (a - b) ** 2
            out["strain"] -= 0.5 * (a + b) ** 2
    density = d2 * wc**2
    if want_shear:
        out["extension"] += density
        out["extension"] *= rho
        out["shear"] *= rho
    if want_strain:
        out["rotation"] *= rho
        out["strain"] *= rho
        out["density"] = rho * density
    return out


def anelastic_residual(uc, vc, wc, rho_c, grid, cells):
    """RMS of div(rho V) over its three terms' RMS, inside cells (a slice tuple)."""
    rho = column(rho_c)
    terms = [grid.ddx(rho * uc)[cells], grid.ddy(rho * vc)[cells], grid.ddz(rho * wc)[cells]]
    divergence = terms[0] + terms[1] + terms[2]
    scale = np.sqrt(sum(np.mean(t**2) for t in terms))
    return float(np.sqrt(np.mean(divergence**2)) / scale) if scale > 0 else float("nan")
