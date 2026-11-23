"""Decompose CM1 output files into DVPGF and effective buoyancy and write ``wforce_*.nc``.

    from cm1analysis.pressure import compute_file
    compute_file("cm1out_000120.nc", "derived/", roi_dbz=30.0, horizontal=True)

One output file is written per input file, so batch jobs can process times
separately. The region of interest (ROI) is, by default, the box around
composite reflectivity >= ``roi_dbz`` (padded and kept off the domain edge, see
:func:`grid.roi_from_reflectivity`); ``roi_km`` fixes it instead. The Poisson
problems are solved on the ROI grown by ``buffer_km`` on every side, and the
fields are written on the ROI only.

With ``section_y_km`` the ROI is one x-z row. The problems are still solved in
3-D, on a strip ``buffer_km`` wide on each side of the row, so the section
matches the full-ROI answer to the accuracy of the crop test.

With ``horizontal`` the same p'_d and p'_b also give the horizontal forces
HDPGF and HBPGF on the ROI's x- and y-faces (no extra Poisson solve), plus
their comparison with the model pressure gradient. ``along_wind`` adds their
component along the horizontal wind.
"""

from __future__ import annotations

import json
import logging
import time as clock
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

from ..io import CM1File
from . import vforce
from .grid import Box, buffered, roi_from_reflectivity
from .source import SolverSource
from .writer import WForceWriter

log = logging.getLogger(__name__)

DEFAULT_ROI_DBZ = 25.0
DEFAULT_BUFFER_KM = 50.0
DEFAULT_ZTOP_KM = 15.0  # Top of the statistics; keep it below the model's Rayleigh damping layer.
# Peak memory measured on a 1-km derecho domain with every split: about 1.3 GB fixed
# (HDF5 chunk caches for full-level chunks, Python) plus 30 crop-sized float64 arrays.
FIXED_GB, LIVE_FIELDS = 1.3, 30
# Options that only add output; a file computed with more of them can serve a request for fewer.
ADDITIVE = ("horizontal", "along_wind")


@dataclass(frozen=True)
class ComputeOptions:
    """Settings of one decomposition run (the legacy run_wforce.py command-line options).

    variant       variant written to the output, one of :data:`vforce.VARIANTS` (see VARIANT_NOTES)
    splits        optional splits: "loading" (thermal/loading), "shear" (RK85), "strain" (T&W), or "all"
    roi_km        fixed ROI [xmin, xmax, ymin, ymax] in km; None uses the reflectivity box
    section_y_km  x-z section mode: the ROI is the row nearest this y
    section_x_km  x limits [xmin, xmax] of the section in km (default: the reflectivity box)
    roi_dbz       composite reflectivity threshold of the automatic ROI
    buffer_km     buffer around the ROI that is solved but not written
    ztop_km       top of the statistics, below the damping layer
    sensitivity   variants compared with A (empty to skip)
    horizontal    also write HDPGF and HBPGF and their checks
    along_wind    also write HDPGF and HBPGF along the horizontal wind (implies horizontal)
    crop_test     run the buffer-trim test
    frame_test    run the reference-frame test
    double        store fields as float64 (default float32)
    workers       scipy.fft workers (-1: all cores)
    """

    variant: str = "A"
    splits: tuple[str, ...] = ()
    roi_km: tuple[float, float, float, float] | None = None
    section_y_km: float | None = None
    section_x_km: tuple[float, float] | None = None
    roi_dbz: float = DEFAULT_ROI_DBZ
    buffer_km: float = DEFAULT_BUFFER_KM
    ztop_km: float = DEFAULT_ZTOP_KM
    sensitivity: tuple[str, ...] = tuple(vforce.VARIANTS[1:])
    horizontal: bool = False
    along_wind: bool = False
    crop_test: bool = True
    frame_test: bool = True
    double: bool = False
    workers: int = -1

    def __post_init__(self):
        set_ = object.__setattr__
        if self.variant not in vforce.VARIANTS:
            raise ValueError(f"unknown variant {self.variant!r}; expected one of {vforce.VARIANTS}")
        splits = [self.splits] if isinstance(self.splits, str) else list(self.splits or ())
        bad = set(splits) - set(vforce.SPLITS) - {"all"}
        if bad:
            raise ValueError(f"unknown split(s) {sorted(bad)}; expected {vforce.SPLITS} or 'all'")
        set_(self, "splits", tuple(vforce.SPLITS) if "all" in splits else tuple(dict.fromkeys(splits)))
        sensitivity = [self.sensitivity] if isinstance(self.sensitivity, str) else self.sensitivity
        sensitivity = tuple(sensitivity or ())
        bad = set(sensitivity) - set(vforce.VARIANTS[1:])
        if bad:
            raise ValueError(f"unknown sensitivity variant(s) {sorted(bad)}; expected {vforce.VARIANTS[1:]}")
        set_(self, "sensitivity", sensitivity)
        for name, size in (("roi_km", 4), ("section_x_km", 2)):
            value = getattr(self, name)
            if value is not None:
                if len(value) != size:
                    raise ValueError(f"{name} needs {size} numbers, got {value}")
                set_(self, name, tuple(float(v) for v in value))
        set_(self, "horizontal", bool(self.horizontal or self.along_wind))
        if self.section_y_km is not None and self.roi_km is not None:
            raise ValueError("roi_km and section_y_km cannot be combined; use section_x_km for the x limits")
        if self.section_x_km is not None and self.section_y_km is None:
            raise ValueError("section_x_km needs section_y_km")

    @classmethod
    def from_mapping(cls, values) -> ComputeOptions:
        """Options from a mapping, ignoring keys that are not compute options."""
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in values.items() if k in names})

    def signature(self) -> dict:
        """The options that determine the output contents (not workers), as JSON-friendly values."""
        values = asdict(self)
        values.pop("workers")
        return json.loads(json.dumps(values))

    def satisfied_by(self, stored: dict) -> bool:
        """True if a file written with the stored signature holds everything these options ask for."""
        wanted = self.signature()
        for key, value in wanted.items():
            if key in ADDITIVE:
                if value and not stored.get(key):
                    return False
            elif stored.get(key) != value:
                return False
        return True


def output_path(input_path, output_dir) -> Path:
    """wforce_<number>.nc for cm1out_<number>.nc, in output_dir."""
    name = Path(input_path).name
    stem = name[len("cm1out_") :] if name.startswith("cm1out_") else name
    return Path(output_dir) / f"wforce_{stem}"


def stored_options(path) -> dict | None:
    """The compute-option signature saved in a wforce file, or None if it has none."""
    from netCDF4 import Dataset

    with Dataset(path) as ds:
        text = getattr(ds, "compute_options", None)
    return None if text is None else json.loads(text)


def compute_file(
    path,
    output_dir,
    options: ComputeOptions | None = None,
    *,
    time_range: Sequence[float] | None = None,
    overwrite: bool = False,
    target=None,
    **kwargs,
) -> Path | None:
    """Decompose one CM1 file and return the path of the written ``wforce_*.nc``.

    ``options`` or keyword arguments (not both) set :class:`ComputeOptions`.
    ``time_range`` = (start, end) keeps only model times in that inclusive
    range, in seconds; None is returned if no time is in range. ``target``
    overrides the output file name. Raises FileExistsError if the output
    exists and ``overwrite`` is false.
    """
    if options is not None and kwargs:
        raise TypeError("pass either options or keyword options, not both")
    options = options or ComputeOptions(**kwargs)
    target = Path(target) if target is not None else output_path(path, output_dir)
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} exists; pass overwrite=True to replace it")
    with CM1File(path) as cm1:
        indices = _time_indices(cm1.times, time_range)
        if not indices:
            log.info("%s: no times in range, skipped", Path(path).name)
            return None
        compute(cm1, target, options, indices)
    return target


def compute(cm1: CM1File, target, options: ComputeOptions, time_indices: Sequence[int] | None = None) -> int:
    """Decompose the given times of an open file into target. Returns the number of failed checks.

    The ROI and crop are fixed by the first time, so every time in a file shares one grid.
    """
    source = SolverSource(cm1)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    indices = list(range(len(source.times))) if time_indices is None else list(time_indices)
    grid = source.grid
    kmax = grid.face_index_below(options.ztop_km * 1000.0)
    log.info(
        "%s: %d x %d x %d, dx = %g m, dz = %g-%g m, ROI top %g km (face %d); hydrometeors %s",
        source.path.name,
        source.nx,
        source.ny,
        grid.nz,
        source.dx,
        grid.dz.min(),
        grid.dz.max(),
        grid.zf[kmax] / 1000,
        kmax,
        ", ".join(source.hydrometeors) or "none",
    )

    roi = select_roi(source, options, indices[0])
    buffer_i = int(round(options.buffer_km * 1000.0 / source.dx))
    buffer_j = int(round(options.buffer_km * 1000.0 / source.dy))
    crop = buffered(roi, buffer_i, buffer_j, source.nx, source.ny)
    margins = crop.margins(roi)
    points = crop.nx * crop.ny * grid.nz
    log.info(
        "ROI x %d-%d, y %d-%d (%d x %d cells); crop %d x %d, buffer W/E/S/N %s cells; ~%.1f GB peak",
        roi.i0,
        roi.i1,
        roi.j0,
        roi.j1,
        roi.nx,
        roi.ny,
        crop.nx,
        crop.ny,
        margins,
        FIXED_GB + points * 8 * LIVE_FIELDS / 1e9,
    )
    if min(margins) < min(buffer_i, buffer_j):
        log.warning("the buffer is clipped by the domain edge on at least one side")

    attrs = {
        "variant": options.variant,
        "variant_description": vforce.VARIANT_NOTES[options.variant],
        "splits": ",".join(options.splits),
        "roi_source": roi_source(options),
        "section": int(options.section_y_km is not None),
        "buffer_km": options.buffer_km,
        "ztop_roi_m": float(grid.zf[kmax]),
        "kmax_face": kmax,
        "horizontal": int(options.horizontal),
        "along_wind": int(options.along_wind),
        "compute_options": json.dumps(options.signature()),
    }
    roi_in_crop = roi.relative_to(crop)
    failures_total = 0
    with WForceWriter(target, source, roi, crop, len(indices), attrs, options.double) as out:
        for out_index, t in enumerate(indices):
            start = clock.time()
            log.info("t = %.0f s: reading", source.times[t])
            inputs, w, spreads = source.read(t, crop)
            fields_, diag, base = vforce.analyze(
                inputs,
                roi_in_crop,
                kmax,
                variant=options.variant,
                splits=options.splits,
                sensitivity=options.sensitivity,
                crop_fractions=vforce.CROP_FRACTIONS if options.crop_test else (),
                frame_test=options.frame_test,
                horizontal=options.horizontal,
                along_wind=options.along_wind,
                workers=options.workers,
                log=log.info,
            )
            fields_["w"] = np.ascontiguousarray(w[(slice(None),) + roi_in_crop.cells])
            diag.update(spreads)
            if not options.crop_test:
                # Keep the crop diagnostics' shape, so every file has the same layout.
                n = len(vforce.CROP_FRACTIONS)
                for key in [k for k in diag if k.startswith("crop_")]:
                    diag[key] = (
                        np.full(n, -1, dtype=np.int32) if key == "crop_buffer_cells" else np.full(n, np.nan)
                    )
            failures = vforce.failed_checks(diag)
            diag["validation_failures"] = len(failures)
            for line in describe(diag, failures):
                log.info("%s", line)
            if failures:
                log.warning("t = %.0f s: validation failed: %s", source.times[t], ", ".join(failures))
            out.write(
                out_index, source.times[t], fields_, base, source.read_cref(t, roi), diag, options.variant
            )
            failures_total += len(failures)
            log.info("t = %.0f s done in %.0f s", source.times[t], clock.time() - start)
            del inputs, fields_, w
    log.info("wrote %s", target)
    return failures_total


def _time_indices(times, time_range):
    if time_range is None:
        return list(range(len(times)))
    start, end = time_range
    return [n for n, t in enumerate(times) if start <= t <= end]


def select_roi(source: SolverSource, options: ComputeOptions, time_index: int) -> Box:
    if options.roi_km is not None:
        return source.roi_from_limits(*options.roi_km)
    if options.section_y_km is not None:
        j = source.nearest_row(options.section_y_km)
        if options.section_x_km is not None:
            limits = source.roi_from_limits(options.section_x_km[0], options.section_x_km[1], -np.inf, np.inf)
        else:
            limits = automatic_roi(source, options, time_index)
        return Box(limits.i0, limits.i1, j, j + 1)
    return automatic_roi(source, options, time_index)


def automatic_roi(source: SolverSource, options: ComputeOptions, time_index: int) -> Box:
    cref = source.read_cref(time_index)
    if cref is None:
        raise ValueError(f"{source.path.name} has no cref; set roi_km")
    roi = roi_from_reflectivity(cref, options.roi_dbz, source.dx, source.dy)
    if roi is None:
        raise ValueError(f"cref never reaches {options.roi_dbz:g} dBZ; set roi_km or lower roi_dbz")
    return roi


def roi_source(options: ComputeOptions) -> str:
    if options.roi_km is not None:
        return "manual"
    if options.section_y_km is not None:
        x = "manual x" if options.section_x_km is not None else f"x from cref >= {options.roi_dbz:g} dBZ"
        return f"x-z section at y = {options.section_y_km:g} km, {x}"
    return f"cref >= {options.roi_dbz:g} dBZ"


def describe(diag, failures) -> list[str]:
    """Human-readable summary lines of one time's diagnostics."""
    lines = [
        "Poisson residual d/b: "
        f"{diag['poisson_residual_dynamic']:.1e} / {diag['poisson_residual_buoyancy']:.1e}; "
        f"solvability d/b: {diag['solvability_dynamic']:.1e} / {diag['solvability_buoyancy']:.1e}",
        "Base state: hydrostatic residual p/Exner "
        f"{diag['hydrostatic_residual_p']:.1e} / {diag['hydrostatic_residual_exner']:.1e}, "
        f"th0 spread {diag['base_th0_spread']:.1e} K",
        f"RMS in ROI: DVPGF {diag['roi_rms_dvpgf']:.3e}, B {diag['roi_rms_B']:.3e} m s-2; "
        f"anelastic residual {diag['anelastic_residual']:.3f}",
        "Consistency with model p': rel RMS (corr) vs -(1/rho0) dp'/dz + b "
        f"{diag['consistency_force_rel_rms']:.3f} ({diag['consistency_force_corr']:.3f}), "
        f"vs CM1 -c_p thv0 dpi'/dz + b {diag['consistency_exner_rel_rms']:.3f} "
        f"({diag['consistency_exner_corr']:.3f}); pressure {diag['consistency_pressure_rel_rms']:.3f}",
    ]
    parts = [
        f"{kind} {diag[f'parts_sum_{kind}']:.1e}"
        for kind in vforce.SPLITS
        if np.isfinite(diag[f"parts_sum_{kind}"])
    ]
    if parts:
        lines.append(f"Parts sum: {', '.join(parts)}")
    sens = [
        f"{v} {diag[f'sens_dvpgf_{v}']:.3f}/{diag[f'sens_B_{v}']:.3f}"
        for v in vforce.VARIANTS[1:]
        if np.isfinite(diag[f"sens_dvpgf_{v}"])
    ]
    if sens:
        lines.append(f"Variants vs A, rel RMS DVPGF/B: {', '.join(sens)}")
    if np.isfinite(diag["frame_dvpgf_rel"]):
        lines.append(
            f"Frame shift ({diag['frame_u']:.1f}, {diag['frame_v']:.1f}) m/s: "
            f"DVPGF rel change {diag['frame_dvpgf_rel']:.3f}"
        )
    for n, fraction in enumerate(vforce.CROP_FRACTIONS):
        if np.isfinite(diag["crop_dvpgf_rel"][n]):
            lines.append(
                f"Buffer trimmed {fraction:.0%} ({diag['crop_buffer_cells'][n]} cells left): "
                f"DVPGF {diag['crop_dvpgf_rel'][n]:.3f}, B {diag['crop_B_rel'][n]:.3f}"
            )
    if "roi_rms_hdpgf" in diag:
        lines += _describe_horizontal(diag)
    lines.append(f"Validation: {'FAILED ' + ', '.join(failures) if failures else 'passed'}")
    return lines


def _describe_horizontal(diag) -> list[str]:
    model = diag.get("roi_rms_model_hpgf", float("nan"))
    lines = [
        f"Horizontal RMS in ROI: HDPGF {diag['roi_rms_hdpgf']:.3e}, HBPGF {diag['roi_rms_hbpgf']:.3e}, "
        f"model -(1/rho0) grad p' {model:.3e} m s-2"
    ]
    rows = [
        f"{v} {diag[f'hcons_{v}_rel_rms']:.3f} ({diag[f'hcons_{v}_corr']:.3f}; "
        f"bias {diag[f'hcons_{v}_bias_x']:.1e}/{diag[f'hcons_{v}_bias_y']:.1e})"
        for v in vforce.VARIANTS
        if np.isfinite(diag[f"hcons_{v}_rel_rms"])
    ]
    if rows:
        lines.append(f"HDPGF + HBPGF vs model, rel RMS (corr; bias x/y m s-2): {', '.join(rows)}")
        if np.isfinite(diag.get("hcons_D_rel_rms", np.nan)):
            lines.append("  (variant D matches the model by construction; it is a bound, not a result)")
    sens = [
        f"{v} {diag[f'sens_hdpgf_{v}']:.3f}/{diag[f'sens_hbpgf_{v}']:.3f}"
        for v in vforce.VARIANTS[1:]
        if np.isfinite(diag.get(f"sens_hdpgf_{v}", np.nan))
    ]
    if sens:
        lines.append(f"Variants vs A, rel RMS HDPGF/HBPGF: {', '.join(sens)}")
    if np.isfinite(diag.get("frame_hdpgf_rel", np.nan)):
        lines.append(f"Frame shift: HDPGF rel change {diag['frame_hdpgf_rel']:.3f}")
    for n, fraction in enumerate(vforce.CROP_FRACTIONS):
        if np.isfinite(diag["crop_hdpgf_rel"][n]):
            lines.append(
                f"Buffer trimmed {fraction:.0%}: HDPGF {diag['crop_hdpgf_rel'][n]:.3f}, "
                f"HBPGF {diag['crop_hbpgf_rel'][n]:.3f}, vs model {diag['crop_hcons_rel_rms'][n]:.3f}"
            )
    return lines


__all__ = [
    "ComputeOptions",
    "compute",
    "compute_file",
    "output_path",
    "stored_options",
]
