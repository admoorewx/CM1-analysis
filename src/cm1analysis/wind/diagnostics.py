"""Validation numbers and boundary-sensitivity tests for one partition (Lynch 1989)."""

from dataclasses import dataclass

import numpy as np

from . import ops, solve
from .partition import partition_wind

TOLERANCE = 1e-10
SENSITIVITY_FLAG = 0.10
CROP_FRACTIONS = (0.1, 0.2, 0.3)
EDGE_BUFFER_CELLS = 20
ROI_PAD_KM = 25.0
ROI_TRIM_PERCENT = 1.0
SPARSE_CHECK_CELLS = 128


# ── Validation ─────────────────────────────────────────────────────────────────


def _max_abs(a):
    return float(np.max(np.abs(a)))


def _dot(a, b):
    return float(np.sum(a * b))


def _edge_weighted(u, v):
    """Give edge faces half weight, which makes the discrete parts orthogonal."""
    u_w, v_w = u.copy(), v.copy()
    u_w[:, [0, -1]] *= 0.5
    v_w[[0, -1], :] *= 0.5
    return u_w, v_w


def _kinetic_energy(u, v):
    return 0.5 * (_dot(u, u) + _dot(v, v))


def validate(p, u, v, dx, dy, workers=-1):
    """Return the validation table as a flat {name: float} dictionary.

    Operator residuals are scaled by the larger of max|delta| and max|zeta| so
    that a flow with no divergence or no vorticity still has a usable limit. A
    flow with neither, such as a uniform wind, falls back to max wind / grid
    spacing.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    wind = max(_max_abs(u), _max_abs(v), 1.0)
    fallback = wind / min(dx, dy)
    scale = max(_max_abs(p.delta), _max_abs(p.zeta))
    if scale <= 1e-14 * fallback:
        scale = fallback

    out = {
        "div_vpsi_rel": _max_abs(ops.divergence(p.u_psi, p.v_psi, dx, dy)) / scale,
        "vor_vchi_rel": _max_abs(ops.vorticity(p.u_chi, p.v_chi, dx, dy)) / scale,
        "div_vh_rel": _max_abs(ops.divergence(p.u_h, p.v_h, dx, dy)) / scale,
        "vor_vh_rel": _max_abs(ops.vorticity(p.u_h, p.v_h, dx, dy)) / scale,
    }

    # Boundary conditions.
    out["vpsi_normal_edge_max"] = max(_max_abs(p.u_psi[:, [0, -1]]), _max_abs(p.v_psi[[0, -1], :]))
    out["chi_ghost_rule_max"] = max(
        _max_abs(p.u_chi[:, 0] - 2.0 * p.chi[:, 0] / dx),
        _max_abs(p.u_chi[:, -1] + 2.0 * p.chi[:, -1] / dx),
        _max_abs(p.v_chi[0, :] - 2.0 * p.chi[0, :] / dy),
        _max_abs(p.v_chi[-1, :] + 2.0 * p.chi[-1, :] / dy),
    )

    # Net boundary flux of the harmonic part (per unit depth). It is scaled by
    # max wind x perimeter, because V_h's own flux is pure noise when V_h is ~0.
    ny, nx = p.delta.shape
    net = (p.u_h[:, -1].sum() - p.u_h[:, 0].sum()) * dy + (p.v_h[-1, :].sum() - p.v_h[0, :].sum()) * dx
    gross = (np.abs(p.u_h[:, [0, -1]]).sum()) * dy + (np.abs(p.v_h[[0, -1], :]).sum()) * dx
    out["vh_net_flux"] = float(net)
    out["vh_gross_flux"] = float(gross)
    out["vh_net_flux_rel"] = abs(float(net)) / (wind * 2.0 * (nx * dx + ny * dy))

    # Kinetic energy, summed over faces, and cross terms as fractions of total.
    total = max(_kinetic_energy(u, v), np.finfo(float).tiny)
    out["ke_total"] = _kinetic_energy(u, v)
    out["ke_frac_psi"] = _kinetic_energy(p.u_psi, p.v_psi) / total
    out["ke_frac_chi"] = _kinetic_energy(p.u_chi, p.v_chi) / total
    out["ke_frac_h"] = _kinetic_energy(p.u_h, p.v_h) / total

    pairs = {
        "psi_chi": ((p.u_psi, p.v_psi), (p.u_chi, p.v_chi)),
        "psi_h": ((p.u_psi, p.v_psi), (p.u_h, p.v_h)),
        "chi_h": ((p.u_chi, p.v_chi), (p.u_h, p.v_h)),
    }
    for name, ((ua, va), (ub, vb)) in pairs.items():
        out[f"ke_cross_{name}"] = (_dot(ua, ub) + _dot(va, vb)) / total
        ua_w, va_w = _edge_weighted(ua, va)
        out[f"ke_cross_{name}_edgewt"] = (_dot(ua_w, ub) + _dot(va_w, vb)) / total

    out["dst_vs_sparse_rel"] = _dst_vs_sparse(u, v, dx, dy, workers)
    return out


def _dst_vs_sparse(u, v, dx, dy, workers):
    """Compare the DST and sparse solvers on a central crop of the domain."""
    ny, nx = ops.grid_shape(u, v)
    size_x, size_y = min(SPARSE_CHECK_CELLS, nx), min(SPARSE_CHECK_CELLS, ny)
    i0, j0 = (nx - size_x) // 2, (ny - size_y) // 2
    uc = u[j0 : j0 + size_y, i0 : i0 + size_x + 1]
    vc = v[j0 : j0 + size_y + 1, i0 : i0 + size_x]
    delta = ops.divergence(uc, vc, dx, dy)
    zeta = ops.vorticity(uc, vc, dx, dy)

    def rel(a, b):
        return _max_abs(a - b) / max(_max_abs(b), np.finfo(float).tiny)

    return max(
        rel(solve.solve_chi(delta, dx, dy, workers), solve.solve_chi_sparse(delta, dx, dy)),
        rel(solve.solve_psi(zeta, dx, dy, workers), solve.solve_psi_sparse(zeta, dx, dy)),
    )


def failed_checks(v):
    """Return the names of the validation numbers that break their limits."""
    limits = {
        "div_vpsi_rel": TOLERANCE,
        "vor_vchi_rel": TOLERANCE,
        "div_vh_rel": TOLERANCE,
        "vor_vh_rel": TOLERANCE,
        "vpsi_normal_edge_max": TOLERANCE,
        "chi_ghost_rule_max": TOLERANCE,
        "vh_net_flux_rel": TOLERANCE,
        "dst_vs_sparse_rel": TOLERANCE,
        "ke_cross_psi_chi": 0.01,
        "ke_cross_psi_h": 0.01,
        "ke_cross_chi_h": 0.01,
        "ke_cross_psi_chi_edgewt": TOLERANCE,
        "ke_cross_psi_h_edgewt": TOLERANCE,
        "ke_cross_chi_h_edgewt": TOLERANCE,
    }
    return [name for name, limit in limits.items() if abs(v[name]) >= limit]


# ── Region of interest ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ROI:
    """Half-open cell index ranges: i0 <= i < i1 and j0 <= j < j1."""

    i0: int
    i1: int
    j0: int
    j1: int

    @property
    def cells(self):
        return (slice(self.j0, self.j1), slice(self.i0, self.i1))


def roi_from_reflectivity(cref, threshold, dx, dy):
    """Box around cref >= threshold, or None if the threshold is never reached.

    The extent uses the 1st to 99th percentile of the qualifying cell positions
    so isolated cells do not stretch the box. It is then padded by ROI_PAD_KM
    and kept outside the edge buffer.
    """
    ny, nx = cref.shape
    jj, ii = np.nonzero(np.asarray(cref) >= threshold)
    if ii.size == 0:
        return None
    pad_i = int(round(ROI_PAD_KM * 1000.0 / dx))
    pad_j = int(round(ROI_PAD_KM * 1000.0 / dy))
    low, high = ROI_TRIM_PERCENT, 100.0 - ROI_TRIM_PERCENT
    i0 = max(int(np.floor(np.percentile(ii, low))) - pad_i, EDGE_BUFFER_CELLS)
    i1 = min(int(np.ceil(np.percentile(ii, high))) + pad_i + 1, nx - EDGE_BUFFER_CELLS)
    j0 = max(int(np.floor(np.percentile(jj, low))) - pad_j, EDGE_BUFFER_CELLS)
    j1 = min(int(np.ceil(np.percentile(jj, high))) + pad_j + 1, ny - EDGE_BUFFER_CELLS)
    if i1 <= i0 or j1 <= j0:
        return None
    return ROI(i0, i1, j0, j1)


# ── Sensitivity tests ──────────────────────────────────────────────────────────


def _rms(a, b):
    return float(np.sqrt(np.mean(a**2 + b**2)))


def _ratio(a, b):
    return a / b if b > 0.0 else float("nan")


def sensitivity(p, u, v, dx, dy, roi, workers=-1):
    """Return bracket, harmonic-share and crop results for one partition.

    Scalars are floats and crop results are arrays ordered as CROP_FRACTIONS.
    Everything that needs an ROI is NaN when roi is None.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    nan_crop = np.full(len(CROP_FRACTIONS), np.nan)
    out = {
        "harmonic_ke_share_domain": _kinetic_energy(p.u_h, p.v_h)
        / max(_kinetic_energy(u, v), np.finfo(float).tiny),
        "harmonic_ke_share_roi": float("nan"),
        "bracket_a_b_raw": float("nan"),
        "bracket_a_c_raw": float("nan"),
        "bracket_a_b_mean_removed": float("nan"),
        "bracket_a_c_mean_removed": float("nan"),
        "bracket_flag": 0,
        "vh_roi_mean_u": float("nan"),
        "vh_roi_mean_v": float("nan"),
        "crop_evaluated": np.zeros(len(CROP_FRACTIONS), dtype=np.int32),
        "crop_change_vpsi": nan_crop.copy(),
        "crop_change_vchi": nan_crop.copy(),
        "crop_change_vh": nan_crop.copy(),
        "crop_flag": np.zeros(len(CROP_FRACTIONS), dtype=np.int32),
    }
    if roi is None:
        return out

    def in_roi(uf, vf):
        uc, vc = ops.winds_to_centers(uf, vf)
        return uc[roi.cells], vc[roi.cells]

    u_full, v_full = in_roi(u, v)
    u_psi, v_psi = in_roi(p.u_psi, p.v_psi)
    u_chi, v_chi = in_roi(p.u_chi, p.v_chi)
    u_h, v_h = in_roi(p.u_h, p.v_h)

    rms_psi, rms_chi = _rms(u_psi, v_psi), _rms(u_chi, v_chi)
    rms_h = _rms(u_h, v_h)
    mean_u, mean_v = float(u_h.mean()), float(v_h.mean())
    rms_h_removed = _rms(u_h - mean_u, v_h - mean_v)

    # V_rot(B) - V_rot(A) and V_div(C) - V_div(A) both equal V_h.
    out["harmonic_ke_share_roi"] = _ratio(rms_h**2, _rms(u_full, v_full) ** 2)
    out["bracket_a_b_raw"] = _ratio(rms_h, rms_psi)
    out["bracket_a_c_raw"] = _ratio(rms_h, rms_chi)
    out["bracket_a_b_mean_removed"] = _ratio(rms_h_removed, rms_psi)
    out["bracket_a_c_mean_removed"] = _ratio(rms_h_removed, rms_chi)
    out["bracket_flag"] = int(
        out["bracket_a_b_mean_removed"] > SENSITIVITY_FLAG
        or out["bracket_a_c_mean_removed"] > SENSITIVITY_FLAG
    )
    out["vh_roi_mean_u"] = mean_u
    out["vh_roi_mean_v"] = mean_v

    ny, nx = ops.grid_shape(u, v)
    for n, fraction in enumerate(CROP_FRACTIONS):
        mx, my = int(round(fraction * nx)), int(round(fraction * ny))
        if roi.i0 < mx or roi.i1 > nx - mx or roi.j0 < my or roi.j1 > ny - my:
            continue  # The crop would cut the ROI, so it is not evaluated.
        crop = partition_wind(
            u[my : ny - my, mx : nx - mx + 1],
            v[my : ny - my + 1, mx : nx - mx],
            dx,
            dy,
            workers=workers,
        )
        shifted = (slice(roi.j0 - my, roi.j1 - my), slice(roi.i0 - mx, roi.i1 - mx))

        def change(uf, vf, u_ref, v_ref, shifted=shifted):
            uc, vc = ops.winds_to_centers(uf, vf)
            return _ratio(_rms(uc[shifted] - u_ref, vc[shifted] - v_ref), _rms(u_ref, v_ref))

        out["crop_evaluated"][n] = 1
        out["crop_change_vpsi"][n] = change(crop.u_psi, crop.v_psi, u_psi, v_psi)
        out["crop_change_vchi"][n] = change(crop.u_chi, crop.v_chi, u_chi, v_chi)
        out["crop_change_vh"][n] = change(crop.u_h, crop.v_h, u_h, v_h)
        out["crop_flag"][n] = int(
            out["crop_change_vpsi"][n] > SENSITIVITY_FLAG or out["crop_change_vchi"][n] > SENSITIVITY_FLAG
        )
    return out
