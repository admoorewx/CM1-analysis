"""DVPGF and effective buoyancy for one output time, with splits and sensitivity tests.

    Dw/Dt = -(1/rho0) dp_d/dz + B + F,   B = b - (1/rho0) dp_b/dz

decompose() does one variant on one crop. analyze() runs the chosen variant,
the optional splits, the single-file consistency check, the other variants and
the crop test, and returns fields cut to the region of interest plus a flat
dictionary of diagnostics. With horizontal=True it also writes the horizontal
forces HDPGF and HBPGF from the same p_d and p_b (hforce.py)
and repeats the comparisons for them.
"""

from dataclasses import dataclass, field, replace

import numpy as np

from . import forcing, hforce, poisson
from .grid import column, trimmed

VARIANTS = "ABCDEFGH"
VARIANT_NOTES = {
    "A": "Xu rho0 form, KR83 top/bottom conditions, periodic lateral (FFT)",
    "B": "lateral Neumann via cosine transform, mean forcing removed",
    "C": "lateral p' = 0 (Dirichlet, sine transform)",
    "D": "p_d := p'_model - p_b",
    "E": "density buoyancy b = -g (rho - rho0) / rho0",
    "F": "base state = analysis-crop mean profile",
    "G": "model rho in place of rho0",
    "H": "Exner form: div(c_p rho0 theta0 grad pi), force -c_p theta0 dpi/dz",
}
LATERAL = {"B": "neumann", "C": "dirichlet"}
SPLITS = ("loading", "shear", "strain")
SPLIT_PARTS = {
    "shear": ("shear", "extension", "remainder"),
    "strain": ("rotation", "strain", "density", "remainder"),
}
CROP_FRACTIONS = (0.1, 0.2, 0.3)
TOLERANCE = 1e-10


@dataclass
class Inputs:
    """Cell-centered float64 fields on the analysis crop, ordered (z, y, x)."""

    grid: object
    uc: np.ndarray
    vc: np.ndarray
    wc: np.ndarray
    thpert: np.ndarray
    qvpert: np.ndarray
    qw: np.ndarray
    base: forcing.BaseState
    prspert: np.ndarray = None
    rho_model: np.ndarray = None
    acc: tuple = field(default=None, repr=False)

    @property
    def shape(self):
        return self.uc.shape

    def acceleration(self):
        if self.acc is None:
            self.acc = forcing.advective_acceleration(self.uc, self.vc, self.wc, self.grid)
        return self.acc

    def sliced(self, box):
        """Views of every field inside box (crop-relative), sharing the cached advection."""
        cells = (slice(None),) + box.cells

        def cut(a):
            return None if a is None else a[cells]

        return replace(
            self,
            uc=cut(self.uc),
            vc=cut(self.vc),
            wc=cut(self.wc),
            thpert=cut(self.thpert),
            qvpert=cut(self.qvpert),
            qw=cut(self.qw),
            prspert=cut(self.prspert),
            rho_model=cut(self.rho_model),
            acc=None if self.acc is None else tuple(cut(a) for a in self.acc),
        )

    def crop_mean_base(self):
        """Variant F base state: the horizontal mean over this crop."""
        b = self.base
        p_mean = b.p + (self.prspert.mean(axis=(1, 2)) if self.prspert is not None else 0.0)
        return forcing.BaseState(
            self.grid,
            b.th + self.thpert.mean(axis=(1, 2)),
            b.qv + self.qvpert.mean(axis=(1, 2)),
            p_mean,
        )


def from_cgrid(grid, u, v, w, **fields):
    """Build Inputs from C-grid winds: u (nz, ny, nx+1), v (nz, ny+1, nx), w (nz+1, ny, nx)."""
    uc = 0.5 * (u[..., 1:] + u[..., :-1])
    vc = 0.5 * (v[:, 1:, :] + v[:, :-1, :])
    return Inputs(grid, uc, vc, grid.faces_to_centers(w), **fields)


@dataclass
class Setup:
    variant: str
    lateral: str
    base: forcing.BaseState
    rho_c: np.ndarray
    rho_f: np.ndarray
    kc: np.ndarray
    kf: np.ndarray
    b_thermal: np.ndarray
    b_loading: np.ndarray


def make_setup(inputs, variant):
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
    grid = inputs.grid
    base = inputs.crop_mean_base() if variant == "F" else inputs.base
    if variant == "F":
        th_pert = inputs.thpert - column(base.th - inputs.base.th)
        qv_pert = inputs.qvpert - column(base.qv - inputs.base.qv)
    else:
        th_pert, qv_pert = inputs.thpert, inputs.qvpert
    b_thermal, b_loading = forcing.buoyancy_parts(th_pert, qv_pert, inputs.qw, base.th)

    rho_c, rho_f = base.rho, base.rho_f
    kc, kf = None, None
    if variant == "E":
        if inputs.rho_model is None:
            raise ValueError("variant E needs the model density")
        total = forcing.density_buoyancy(inputs.rho_model, column(base.qv) + inputs.qvpert, inputs.qw, base)
        b_thermal = total - b_loading  # Keep the loading part so the split still works.
    if variant == "G":
        if inputs.rho_model is None:
            raise ValueError("variant G needs the model density")
        rho_c = inputs.rho_model
        rho_f = grid.to_faces(rho_c, "linear")
    if variant == "H":
        # theta0 (1 + qv0 Rv/Rd) makes c_p rho0 theta pi' the linearized p' for CM1's dry rho0.
        theta = base.th * (1.0 + forcing.REPS * base.qv)
        kc = forcing.CP * base.rho * theta
        kf = forcing.CP * base.rho_f * grid.to_faces(theta, "linear")
    return Setup(variant, LATERAL.get(variant, "periodic"), base, rho_c, rho_f, kc, kf, b_thermal, b_loading)


@dataclass
class Decomposition:
    setup: Setup
    p_d: np.ndarray  # Pa (variant H: Exner function)
    p_b: np.ndarray
    p_d_solved: np.ndarray  # Same as p_d except in variant D.
    dvpgf: np.ndarray  # z-faces, m s-2
    bpgf: np.ndarray
    b_face: np.ndarray
    B: np.ndarray
    info: dict

    def pressure(self, phi):
        """Pressure in Pa: phi itself, or c_p rho0 theta0 pi' for variant H."""
        return phi if self.setup.kc is None else column(self.setup.kc) * phi


def _solve(rhs, grid, setup, workers):
    phi, info = poisson.solve(rhs, grid, setup.lateral, setup.kc, setup.kf, workers)
    res = poisson.residual(phi, rhs, grid, setup.lateral, setup.kc, setup.kf, info["mean_forcing_removed"])
    return phi, res, info["solvability"]


def _buoyancy_force(b, grid, setup, workers):
    rhs, b_face = forcing.buoyancy_rhs(b, setup.rho_f, grid)
    p_b, res, solvability = _solve(rhs, grid, setup, workers)
    del rhs
    bpgf = forcing.vertical_force(p_b, grid, setup.rho_f, setup.kf, boundary=(-b_face[0], -b_face[-1]))
    return p_b, bpgf, b_face, res, solvability


def decompose(inputs, variant="A", workers=-1):
    """Split the vertical acceleration of one crop into DVPGF and B."""
    grid = inputs.grid
    setup = make_setup(inputs, variant)
    p_b, bpgf, b_face, res_b, solv_b = _buoyancy_force(
        setup.b_thermal + setup.b_loading, grid, setup, workers
    )
    rhs = forcing.dynamic_rhs(inputs.acceleration(), setup.rho_c, grid, setup.lateral)
    p_solved, res_d, solv_d = _solve(rhs, grid, setup, workers)
    del rhs
    p_d = p_solved
    if variant == "D":
        if inputs.prspert is None:
            raise ValueError("variant D needs the model pressure perturbation")
        p_d = inputs.prspert - p_b
    dvpgf = forcing.vertical_force(p_d, grid, setup.rho_f, setup.kf)
    info = {
        "poisson_residual_dynamic": res_d,
        "poisson_residual_buoyancy": res_b,
        "solvability_dynamic": solv_d,
        "solvability_buoyancy": solv_b,
    }
    return Decomposition(setup, p_d, p_b, p_solved, dvpgf, bpgf, b_face, b_face + bpgf, info)


# ── Optional splits ────────────────────────────────────────────────────────────


def _relative_max(a, ref):
    scale = float(np.max(np.abs(ref)))
    return float(np.max(np.abs(a - ref))) / scale if scale > 0 else float(np.max(np.abs(a)))


def split_buoyancy(inputs, dec, reduce, workers=-1):
    """Thermal and loading parts of p_b (Xu Fig. 22). Returns (fields, parts-sum error)."""
    grid, setup = inputs.grid, dec.setup
    fields, p_sum, f_sum = {}, np.zeros(inputs.shape), np.zeros(dec.bpgf.shape)
    for name, b in (("thermal", setup.b_thermal), ("loading", setup.b_loading)):
        p, bpgf, b_face, _, _ = _buoyancy_force(b, grid, setup, workers)
        p_sum += p
        f_sum += bpgf
        fields[f"bpgf_{name}"] = reduce(bpgf)
        fields[f"B_{name}"] = reduce(b_face + bpgf)
    error = max(_relative_max(p_sum, dec.p_b), _relative_max(f_sum, dec.bpgf))
    return fields, error


def split_dynamic(inputs, dec, kind, reduce, workers=-1):
    """Shear/extension or rotation/strain/density parts of p_d, plus the remainder.

    The mean vertical gradient of each part is undetermined (KR83 A6-A7), so the
    horizontal mean of each part is removed at every level. The parts therefore
    sum to p_d minus its level means. Returns (fields, parts-sum error).
    """
    grid, setup = inputs.grid, dec.setup
    sources = forcing.velocity_gradient_sources(inputs.uc, inputs.vc, inputs.wc, setup.rho_c, grid, (kind,))
    remainder = forcing.dynamic_rhs(inputs.acceleration(), setup.rho_c, grid, setup.lateral)
    for source in sources.values():
        remainder -= source
    sources["remainder"] = remainder
    fields, p_sum = {}, np.zeros(inputs.shape)
    for name in SPLIT_PARTS[kind]:
        p, _ = poisson.solve(sources.pop(name), grid, setup.lateral, setup.kc, setup.kf, workers)
        p -= p.mean(axis=(1, 2), keepdims=True)
        p_sum += p
        label = f"{kind}_remainder" if name == "remainder" else name
        fields[f"dvpgf_{label}"] = reduce(forcing.vertical_force(p, grid, setup.rho_f, setup.kf))
    reference = dec.p_d_solved - dec.p_d_solved.mean(axis=(1, 2), keepdims=True)
    return fields, _relative_max(p_sum, reference)


# ── Comparison statistics ──────────────────────────────────────────────────────


def relative_rms(a, ref):
    den = np.sqrt(np.mean(ref**2))
    return float(np.sqrt(np.mean((a - ref) ** 2)) / den) if den > 0 else float("nan")


def correlation(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    den = np.sqrt(np.sum(a**2) * np.sum(b**2))
    return float(np.sum(a * b) / den) if den > 0 else float("nan")


def _level_anomaly(a):
    return a - a.mean(axis=(1, 2), keepdims=True)


# ── Full analysis of one time ──────────────────────────────────────────────────


def analyze(
    inputs,
    roi,
    kmax,
    variant="A",
    splits=(),
    sensitivity=VARIANTS[1:],
    crop_fractions=CROP_FRACTIONS,
    frame_test=True,
    horizontal=False,
    along_wind=False,
    workers=-1,
    log=print,
):
    """Return (fields, diagnostics, base state used) for one crop.

    roi is a Box relative to the crop. Fields are cut to it horizontally and
    keep every level. Statistics use z-faces 1..kmax (the bottom face, where
    w = 0, and everything above the damping layer are left out) and cell
    centers below face kmax. Horizontal forces (horizontal=True) sit on the
    ROI's x- and y-faces at cell-center levels; along_wind adds their
    projection on the horizontal wind at cell centers.
    """
    grid = inputs.grid
    faces = (slice(1, kmax + 1),) + roi.cells
    centers = (slice(0, kmax),) + roi.cells

    def reduce(a):
        return np.ascontiguousarray(a[(slice(None),) + roi.cells])

    diag = {"kmax": kmax}
    p_form, exner_form = inputs.base.hydrostatic_residuals()
    diag["hydrostatic_residual_p"] = p_form
    diag["hydrostatic_residual_exner"] = exner_form

    log(f"    variant {variant}: solving")
    dec = decompose(inputs, variant, workers)
    diag.update(dec.info)
    fields = {
        "p_d": reduce(dec.p_d),
        "p_b": reduce(dec.p_b),
        "dvpgf": reduce(dec.dvpgf),
        "bpgf": reduce(dec.bpgf),
        "b": reduce(dec.b_face),
        "B": reduce(dec.B),
    }
    net = dec.dvpgf[faces] + dec.B[faces]

    def horizontal_forces(d):
        return hforce.forces(d.p_d, d.p_b, roi, grid, d.setup.rho_c, d.setup.kc)

    h_fields = model_h = None
    if horizontal:
        h_fields = horizontal_forces(dec)
        fields.update(h_fields)
        if along_wind:
            uc, vc = inputs.uc[(slice(None),) + roi.cells], inputs.vc[(slice(None),) + roi.cells]
            for name in ("hdpgf", "hbpgf"):
                fields[f"{name}_s"] = hforce.along_wind(h_fields[f"{name}_x"], h_fields[f"{name}_y"], uc, vc)
        for name in VARIANTS:
            for stat in ("rel_rms", "corr", "bias_x", "bias_y"):
                diag[f"hcons_{name}_{stat}"] = float("nan")
        if inputs.prspert is not None:
            # The anelastic -(1/rho0) dp'/dx with the input base state for every variant.
            model_h = hforce.model_forces(inputs.prspert, roi, grid, inputs.base.rho)
            fields.update(model_h)

    def record_hcons(name, h):
        if model_h is not None:
            for stat, value in hforce.comparison(h, model_h, kmax).items():
                diag[f"hcons_{name}_{stat}"] = value

    if horizontal:
        record_hcons(variant, h_fields)

    # Single-file consistency check against the model pressure, in two forms:
    # the anelastic -(1/rho0) dp'/dz, and CM1's own -c_p theta_v0 dpi'/dz. They
    # differ by about (c_v/c_p) g p'/p0, the compressible part the anelastic
    # p' form leaves out.
    nan_keys = (
        "consistency_force_rel_rms",
        "consistency_force_corr",
        "consistency_exner_rel_rms",
        "consistency_exner_corr",
        "consistency_pressure_rel_rms",
    )
    diag.update(dict.fromkeys(nan_keys, float("nan")))
    if inputs.prspert is not None:
        for name, key, model_force in (
            ("model_vpgf", "force", forcing.vertical_force(inputs.prspert, grid, dec.setup.rho_f)),
            ("model_vpgf_exner", "exner", forcing.model_exner_force(inputs.prspert, inputs.base, grid)),
        ):
            fields[name] = reduce(model_force)
            model_net = model_force[faces] + dec.b_face[faces]
            del model_force
            diag[f"consistency_{key}_rel_rms"] = relative_rms(net, model_net)
            diag[f"consistency_{key}_corr"] = correlation(net, model_net)
            del model_net
        p_sum = dec.pressure(dec.p_d + dec.p_b)[centers]
        diag["consistency_pressure_rel_rms"] = relative_rms(
            _level_anomaly(p_sum), _level_anomaly(inputs.prspert[centers])
        )

    diag["anelastic_residual"] = forcing.anelastic_residual(
        inputs.uc, inputs.vc, inputs.wc, dec.setup.rho_c, grid, centers
    )

    for kind in SPLITS:
        diag[f"parts_sum_{kind}"] = float("nan")
    for kind in splits:
        log(f"    split: {kind}")
        if kind == "loading":
            parts, error = split_buoyancy(inputs, dec, reduce, workers)
        else:
            parts, error = split_dynamic(inputs, dec, kind, reduce, workers)
        fields.update(parts)
        diag[f"parts_sum_{kind}"] = error

    # Sensitivity to the variant, always measured against variant A.
    ref = dec if variant == "A" else None
    for name in VARIANTS[1:]:
        diag[f"sens_dvpgf_{name}"] = float("nan")
        diag[f"sens_B_{name}"] = float("nan")
        if horizontal:
            diag[f"sens_hdpgf_{name}"] = float("nan")
            diag[f"sens_hbpgf_{name}"] = float("nan")
    if sensitivity:
        if ref is None:
            log("    sensitivity reference: variant A")
            ref = decompose(inputs, "A", workers)
        ref_dvpgf, ref_B = ref.dvpgf[faces].copy(), ref.B[faces].copy()
        if horizontal:
            ref_h = h_fields if variant == "A" else horizontal_forces(ref)
            if variant != "A":
                record_hcons("A", ref_h)
        del ref
        for name in sensitivity:
            if name == "A":
                continue
            if name == variant:
                other = dec
            else:
                log(f"    sensitivity: variant {name}")
                other = decompose(inputs, name, workers)
            diag[f"sens_dvpgf_{name}"] = relative_rms(other.dvpgf[faces], ref_dvpgf)
            diag[f"sens_B_{name}"] = relative_rms(other.B[faces], ref_B)
            if horizontal:
                other_h = h_fields if name == variant else horizontal_forces(other)
                diag[f"sens_hdpgf_{name}"] = hforce.relative_rms(other_h, ref_h, "hdpgf", kmax)
                diag[f"sens_hbpgf_{name}"] = hforce.relative_rms(other_h, ref_h, "hbpgf", kmax)
                record_hcons(name, other_h)
                del other_h
            del other

    # Reference frame: horizontal advection in a frame moving with the ROI-mean wind.
    diag["frame_u"] = diag["frame_v"] = diag["frame_dvpgf_rel"] = float("nan")
    if horizontal:
        diag["frame_hdpgf_rel"] = float("nan")
    if frame_test:
        log("    frame test")
        cx, cy = float(inputs.uc[centers].mean()), float(inputs.vc[centers].mean())
        acc = forcing.advective_acceleration(inputs.uc, inputs.vc, inputs.wc, grid, (cx, cy))
        rhs = forcing.dynamic_rhs(acc, dec.setup.rho_c, grid, dec.setup.lateral)
        del acc
        p, _ = poisson.solve(rhs, grid, dec.setup.lateral, dec.setup.kc, dec.setup.kf, workers)
        del rhs
        shifted = forcing.vertical_force(p, grid, dec.setup.rho_f, dec.setup.kf)
        solved = forcing.vertical_force(dec.p_d_solved, grid, dec.setup.rho_f, dec.setup.kf)
        diag["frame_u"], diag["frame_v"] = cx, cy
        diag["frame_dvpgf_rel"] = relative_rms(shifted[faces], solved[faces])
        if horizontal:
            setup = dec.setup
            moved = hforce.forces(p, p, roi, grid, setup.rho_c, setup.kc)
            still = hforce.forces(dec.p_d_solved, dec.p_d_solved, roi, grid, setup.rho_c, setup.kc)
            diag["frame_hdpgf_rel"] = hforce.relative_rms(moved, still, "hdpgf", kmax)

    # Crop test: trim the buffer and re-solve.
    n = len(crop_fractions)
    diag["crop_buffer_cells"] = np.full(n, -1, dtype=np.int32)
    diag["crop_dvpgf_rel"] = np.full(n, np.nan)
    diag["crop_B_rel"] = np.full(n, np.nan)
    if horizontal:
        for key in ("crop_hdpgf_rel", "crop_hbpgf_rel", "crop_hcons_rel_rms"):
            diag[key] = np.full(n, np.nan)
    whole = type(roi)(0, inputs.shape[2], 0, inputs.shape[1])
    margins = whole.margins(roi)
    diag["buffer_cells_min"] = int(min(margins))
    dvpgf_roi, B_roi = dec.dvpgf[faces].copy(), dec.B[faces].copy()
    base = dec.setup.base
    del dec
    for m, fraction in enumerate(crop_fractions):
        sub = trimmed(whole, roi, fraction)
        log(f"    crop test: buffer trimmed {fraction:.0%}")
        sliced = inputs.sliced(sub)
        small = decompose(sliced, variant, workers)
        inner_roi = roi.relative_to(sub)
        inner = (faces[0],) + inner_roi.cells
        diag["crop_buffer_cells"][m] = int(min(sub.margins(roi)))
        diag["crop_dvpgf_rel"][m] = relative_rms(small.dvpgf[inner], dvpgf_roi)
        diag["crop_B_rel"][m] = relative_rms(small.B[inner], B_roi)
        if horizontal:
            small_h = hforce.forces(small.p_d, small.p_b, inner_roi, grid, small.setup.rho_c, small.setup.kc)
            diag["crop_hdpgf_rel"][m] = hforce.relative_rms(small_h, h_fields, "hdpgf", kmax)
            diag["crop_hbpgf_rel"][m] = hforce.relative_rms(small_h, h_fields, "hbpgf", kmax)
            if model_h is not None:
                diag["crop_hcons_rel_rms"][m] = hforce.comparison(small_h, model_h, kmax)["rel_rms"]
            del small_h
        del small, sliced

    diag["roi_rms_dvpgf"] = float(np.sqrt(np.mean(dvpgf_roi**2)))
    diag["roi_rms_B"] = float(np.sqrt(np.mean(B_roi**2)))
    if horizontal:
        diag["roi_rms_hdpgf"] = hforce.roi_rms(h_fields, "hdpgf", kmax)
        diag["roi_rms_hbpgf"] = hforce.roi_rms(h_fields, "hbpgf", kmax)
        if model_h is not None:
            diag["roi_rms_model_hpgf"] = hforce.roi_rms(
                {k.replace("model_hpgf", "m"): v for k, v in model_h.items()}, "m", kmax
            )
    return fields, diag, base


def failed_checks(diag):
    """Names of the must-pass checks (residuals, solvability, parts sums) at or above TOLERANCE."""
    names = [
        "poisson_residual_dynamic",
        "poisson_residual_buoyancy",
        "solvability_dynamic",
        "solvability_buoyancy",
    ] + [f"parts_sum_{kind}" for kind in SPLITS]
    return [n for n in names if np.isfinite(diag.get(n, np.nan)) and diag[n] >= TOLERANCE]
