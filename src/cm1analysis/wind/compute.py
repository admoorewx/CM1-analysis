"""Partition CM1 winds on one level and write the result to netCDF.

:func:`write_partition` partitions chosen times of one open file into one
output file; :func:`partition_file` adds the legacy command-line conventions
(output named ``wind_decomp_<suffix>.nc``, time range, refusal to overwrite).
Both validate every partition and return how many validation checks failed,
which is zero for a correct solve.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..io import CM1File
from . import diagnostics
from .diagnostics import ROI
from .output import PartitionWriter
from .partition import apply_variant, partition_wind
from .source import WindLevel

log = logging.getLogger(__name__)

DEFAULT_ROI_DBZ = 25.0
VARIANTS = ("A", "B", "C")


def output_path(input_path: str | Path, output_dir: str | Path) -> Path:
    """``cm1out_000120.nc`` -> ``<output_dir>/wind_decomp_000120.nc``."""
    name = Path(input_path).name
    name = "wind_decomp_" + (name[len("cm1out_") :] if name.startswith("cm1out_") else name)
    return Path(output_dir) / name


def time_indices(times: Sequence[float], time_range: Sequence[float] | None) -> list[int]:
    """Indices of the times inside the inclusive range [start, end] in s (all if None)."""
    if time_range is None:
        return list(range(len(times)))
    start, end = time_range
    return [n for n, t in enumerate(times) if start <= t <= end]


def roi_source(roi_km: Sequence[float] | None, roi_dbz: float) -> str:
    """How the region of interest is chosen, as recorded in the output's ``roi_source``."""
    return "manual" if roi_km is not None else f"cref >= {roi_dbz:g} dBZ"


def select_roi(source: WindLevel, time_index: int, roi_km: Sequence[float] | None, roi_dbz: float):
    """The fixed ROI if limits are given, else the box around cref >= roi_dbz (None if impossible)."""
    if roi_km is not None:
        return source.roi_from_limits(*roi_km)
    cref = source.read_cref(time_index)
    if cref is None:
        log.warning("%s has no cref; ROI tests skipped", source.path.name)
        return None
    return diagnostics.roi_from_reflectivity(cref, roi_dbz, source.dx, source.dy)


def roi_values(roi: ROI | None) -> dict[str, int]:
    if roi is None:
        return {"roi_i0": -1, "roi_i1": -1, "roi_j0": -1, "roi_j1": -1}
    return {"roi_i0": roi.i0, "roi_i1": roi.i1, "roi_j0": roi.j0, "roi_j1": roi.j1}


def describe(time: float, roi: ROI | None, checks: Mapping, sens: Mapping, failures: Sequence[str]) -> str:
    """Multi-line summary of one partition: energy shares, brackets, crop tests, validation."""
    box = "none" if roi is None else f"x {roi.i0}-{roi.i1} y {roi.j0}-{roi.j1} (cells)"
    lines = [
        f"  t = {time:.0f} s, ROI {box}",
        "    KE psi/chi/h: "
        f"{checks['ke_frac_psi']:.3f} / {checks['ke_frac_chi']:.3f} / {checks['ke_frac_h']:.3f}"
        f", V_h share in ROI: {sens['harmonic_ke_share_roi']:.3f}",
        "    Bracket A-B, A-C raw: "
        f"{sens['bracket_a_b_raw']:.3f}, {sens['bracket_a_c_raw']:.3f}; mean removed: "
        f"{sens['bracket_a_b_mean_removed']:.3f}, {sens['bracket_a_c_mean_removed']:.3f}"
        f" [{'FLAG' if sens['bracket_flag'] else 'ok'}]",
    ]
    for n, fraction in enumerate(diagnostics.CROP_FRACTIONS):
        if sens["crop_evaluated"][n]:
            lines.append(
                f"    Crop {fraction:.0%}: V_psi {sens['crop_change_vpsi'][n]:.3f}, "
                f"V_chi {sens['crop_change_vchi'][n]:.3f}, V_h {sens['crop_change_vh'][n]:.3f}"
                f" [{'FLAG' if sens['crop_flag'][n] else 'ok'}]"
            )
        else:
            lines.append(f"    Crop {fraction:.0%}: not evaluated (would cut the ROI)")
    lines.append(f"    Validation: {'FAILED ' + ', '.join(failures) if failures else 'passed'}")
    return "\n".join(lines)


def write_partition(
    cm1: CM1File,
    target: str | Path,
    indices: Sequence[int] | None = None,
    *,
    level: int = 0,
    height_km: float | None = None,
    variant: str = "A",
    roi_km: Sequence[float] | None = None,
    roi_dbz: float = DEFAULT_ROI_DBZ,
    workers: int = -1,
) -> int:
    """Partition the winds at the given time indices (default: all) of one file into ``target``.

    The level is the index ``level``, or the level nearest ``height_km`` if
    given. ``roi_km = [xmin, xmax, ymin, ymax]`` fixes the region of interest
    for the sensitivity tests; otherwise it is the box around
    ``cref >= roi_dbz``. Variants B and C also write ``u_rot``, ``v_rot``,
    ``u_div`` and ``v_div``. Returns the number of failed validation checks.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected A, B or C")
    source = WindLevel(cm1, level, height_km)
    indices = list(range(cm1.ntimes)) if indices is None else list(indices)
    if not indices:
        raise ValueError("no times to partition")
    requested = "" if height_km is None else f", nearest to requested {height_km * 1000.0:g} m"
    log.info(
        "%s: level %d (%g m%s), dx = %g m, dy = %g m",
        cm1.path.name,
        source.level,
        source.level_height_m,
        requested,
        source.dx,
        source.dy,
    )
    attrs: dict[str, Any] = {"roi_dbz_threshold": roi_dbz} if roi_km is None else {}
    if height_km is not None:
        attrs["requested_height_m"] = height_km * 1000.0
    failures_total = 0
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with PartitionWriter(target, source, len(indices), variant, roi_source(roi_km, roi_dbz), attrs) as out:
        for out_index, t in enumerate(indices):
            u, v = source.read_wind(t)
            p = partition_wind(u, v, source.dx, source.dy, workers=workers)
            checks = diagnostics.validate(p, u, v, source.dx, source.dy, workers)
            failures = diagnostics.failed_checks(checks)
            roi = select_roi(source, t, roi_km, roi_dbz)
            sens = diagnostics.sensitivity(p, u, v, source.dx, source.dy, roi, workers=workers)
            log.info("%s", describe(source.times[t], roi, checks, sens, failures))
            if failures:
                log.warning("t = %g s: validation failed: %s", source.times[t], ", ".join(failures))
            values = {**checks, **sens, **roi_values(roi), "validation_failures": len(failures)}
            out.write(out_index, source.times[t], p, values, apply_variant(p, variant))
            failures_total += len(failures)
    log.info("wrote %s", target)
    return failures_total


def partition_file(
    path: str | Path,
    output_dir: str | Path,
    *,
    time_range: Sequence[float] | None = None,
    overwrite: bool = False,
    **options: Any,
) -> tuple[Path | None, int]:
    """Partition one CM1 file into ``<output_dir>/wind_decomp_<suffix>.nc``.

    ``options`` are those of :func:`write_partition`. Returns (output path, or
    None if no time is inside ``time_range``; number of failed checks). Raises
    FileExistsError if the output exists and ``overwrite`` is false.
    """
    target = output_path(path, output_dir)
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} exists; pass overwrite to replace it")
    with CM1File(path) as cm1:
        indices = time_indices(cm1.times, time_range)
        if not indices:
            log.info("%s: no times in range, skipped", Path(path).name)
            return None, 0
        return target, write_partition(cm1, target, indices, **options)
