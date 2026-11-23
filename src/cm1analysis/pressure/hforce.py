"""Horizontal pressure-gradient forces from the vertical decomposition's p_d and p_b.

    Du/Dt = -(1/rho0) dp_d/dx - (1/rho0) dp_b/dx + F_u     (and v with y)

HDPGF is the dynamic part and HBPGF the buoyancy part. No new Poisson solve is
needed. Forces sit on the x-faces and y-faces of the region of interest:
(p[i] - p[i-1]) / dx at face i, with rho0 of that level. In the Exner form
(variant H) the force is -c_p theta0 dpi'/dx, i.e. -(Kc / rho0) dphi/dx.

A face whose outer neighbour lies outside the solved crop is NaN. Constants
and functions of z alone drop out of horizontal gradients, so the KR83
undetermined-constant issue does not arise here.
"""

import numpy as np

from .grid import column

AXES = {"x": 2, "y": 1}
MIN_SPEED = 0.5  # m/s; the along-wind direction is undefined below this, so the value is NaN.


def _roi_index(roi, axis, start, stop):
    idx = [slice(None), slice(roi.j0, roi.j1), slice(roi.i0, roi.i1)]
    idx[axis] = slice(start, stop)
    return tuple(idx)


def face_force(phi, roi, axis, spacing, rho_c, kc=None):
    """-(K / rho) dphi/dn on the faces of roi normal to axis (2 = x, 1 = y).

    phi is on the crop and roi is crop-relative. rho_c is a profile or a 3-D
    crop field (variant G); kc is the variant H profile or None. Returns an
    array with one more point than roi along axis.
    """
    lo, hi = (roi.i0, roi.i1) if axis == 2 else (roi.j0, roi.j1)
    first, last = max(lo, 1), min(hi, phi.shape[axis] - 1)  # Faces with both neighbours in the crop.
    shape = [phi.shape[0], roi.ny, roi.nx]
    shape[axis] += 1
    out = np.full(shape, np.nan)
    if last < first:
        return out
    upper = _roi_index(roi, axis, first, last + 1)
    lower = _roi_index(roi, axis, first - 1, last)
    target = [slice(None)] * 3
    target[axis] = slice(first - lo, last - lo + 1)
    target = tuple(target)

    out[target] = phi[upper] - phi[lower]
    out[target] /= -spacing
    coef = 1.0 / np.asarray(rho_c, dtype=np.float64)
    if kc is not None:
        coef = coef * np.asarray(kc, dtype=np.float64)
    if coef.ndim == 3:
        out[target] *= 0.5 * (coef[upper] + coef[lower])
    else:
        out[target] *= column(coef)
    return out


def forces(p_d, p_b, roi, grid, rho_c, kc=None):
    """HDPGF and HBPGF, both components, on the faces of roi."""
    out = {}
    for name, phi in (("hdpgf", p_d), ("hbpgf", p_b)):
        out[f"{name}_x"] = face_force(phi, roi, AXES["x"], grid.dx, rho_c, kc)
        out[f"{name}_y"] = face_force(phi, roi, AXES["y"], grid.dy, rho_c, kc)
    return out


def model_forces(prspert, roi, grid, rho_c):
    """-(1/rho0) dp'/dx and dp'/dy from the model pressure, for the single-file comparison."""
    return {
        "model_hpgf_x": face_force(prspert, roi, AXES["x"], grid.dx, rho_c),
        "model_hpgf_y": face_force(prspert, roi, AXES["y"], grid.dy, rho_c),
    }


def along_wind(fx, fy, uc, vc):
    """(u F_x + v F_y) / |V_h| at cell centers, from face forces and center winds on the ROI."""
    cx = 0.5 * (fx[..., 1:] + fx[..., :-1])
    cy = 0.5 * (fy[:, 1:, :] + fy[:, :-1, :])
    speed = np.hypot(uc, vc)
    out = np.full(speed.shape, np.nan)
    moving = speed >= MIN_SPEED
    out[moving] = (uc[moving] * cx[moving] + vc[moving] * cy[moving]) / speed[moving]
    return out


# ── Statistics over both components ────────────────────────────────────────────


def _pairs(a_x, a_y, b_x, b_y, kmax):
    """Flattened finite values of both components on levels below face kmax."""
    a = np.concatenate([a_x[:kmax].ravel(), a_y[:kmax].ravel()])
    b = np.concatenate([b_x[:kmax].ravel(), b_y[:kmax].ravel()])
    keep = np.isfinite(a) & np.isfinite(b)
    return a[keep], b[keep]


def relative_rms(f, ref, name, kmax):
    """RMS(f - ref) / RMS(ref) over both components of the named force."""
    a, b = _pairs(f[f"{name}_x"], f[f"{name}_y"], ref[f"{name}_x"], ref[f"{name}_y"], kmax)
    den = np.sqrt(np.mean(b**2)) if b.size else 0.0
    return float(np.sqrt(np.mean((a - b) ** 2)) / den) if den > 0 else float("nan")


def comparison(f, model, kmax):
    """HDPGF + HBPGF against the model force: relative RMS, correlation and per-component bias."""
    net = {c: f[f"hdpgf_{c}"] + f[f"hbpgf_{c}"] for c in "xy"}
    a, b = _pairs(net["x"], net["y"], model["model_hpgf_x"], model["model_hpgf_y"], kmax)
    out = {"rel_rms": float("nan"), "corr": float("nan"), "bias_x": float("nan"), "bias_y": float("nan")}
    if b.size == 0:
        return out
    den = np.sqrt(np.mean(b**2))
    if den > 0:
        out["rel_rms"] = float(np.sqrt(np.mean((a - b) ** 2)) / den)
    da, db = a - a.mean(), b - b.mean()
    norm = np.sqrt(np.sum(da**2) * np.sum(db**2))
    if norm > 0:
        out["corr"] = float(np.sum(da * db) / norm)
    for c in "xy":
        diff = (net[c] - model[f"model_hpgf_{c}"])[:kmax]
        diff = diff[np.isfinite(diff)]
        out[f"bias_{c}"] = float(diff.mean()) if diff.size else float("nan")
    return out


def roi_rms(f, name, kmax):
    a, _ = _pairs(f[f"{name}_x"], f[f"{name}_y"], f[f"{name}_x"], f[f"{name}_y"], kmax)
    return float(np.sqrt(np.mean(a**2))) if a.size else float("nan")
