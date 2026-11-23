"""Helmholtz wind-partition plots (Lynch 1989), ported from the legacy ``wind_decomposition/`` scripts.

``wind_partition`` draws one 2x3 plan view per output time: rotational wind
``V_psi`` with psi contours, divergent wind ``V_chi`` with chi contours and the
total wind with vectors (top row); vertical vorticity, divergence and the
pressure perturbation at the partitioned level (bottom row). Composite
reflectivity is contoured on every panel.

Compute and caching: the partition of each time is a netCDF file under
``<output>/<run>/wind/derived/``, named after the run, level, variant and
time. If a file with matching settings exists it is reused; otherwise the
partition is computed from the CM1 file's native ``u`` and ``v`` and written
there first. Each time has its own file, so batch jobs can split a run with
``--times`` and share the cache. Set ``recompute: true`` to rebuild it.

Options (``plots.wind.wind_partition`` in the config):

- ``height_km``: partition level (nearest model level); null = lowest level.
- ``variant``: A, B or C (see :func:`cm1analysis.wind.apply_variant`).
- ``roi_km`` / ``roi_dbz``: fixed region of interest ``[xmin, xmax, ymin,
  ymax]``, or the box around ``cref >= roi_dbz``; it drives the sensitivity
  tests and is the default view.
- ``cref_dbz``: reflectivity contour level.
- ``full_domain``: view the whole domain inside the edge buffer instead of the ROI.
- ``extent_km`` (common to the category): an explicit view, overriding both.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.lines import Line2D
from netCDF4 import Dataset

from ..pipeline import TimeContext
from ..registry import Category, common_options, register
from ..wind.compute import roi_source, write_partition
from ..wind.diagnostics import EDGE_BUFFER_CELLS
from ..wind.ops import winds_to_centers
from ._common import WIND, time_label

log = logging.getLogger(__name__)

WIND_CATEGORY = Category.WIND
DERIVED_DIR = "derived"
TIME_MATCH_S = 0.5

POTENTIAL_SCALE = 1e5  # psi and chi are plotted in 1e5 m2 s-1
VORTICITY_SCALE = 1e-3  # zeta and delta are plotted in 1e-3 s-1
PA_PER_HPA = 100.0
POTENTIAL_LEVELS = 17
CREF_COLOR = "white"  # over the wind-speed colormap (top row)
CREF_COLOR_LIGHT_BG = "#111111"  # over the RdBu_r panels, whose near-zero values are white
CONTOUR_COLOR = "gray"

common_options(WIND_CATEGORY, {"extent_km": None})


# ------------------------------------------------------------------ compute
def _level(ctx: TimeContext) -> int:
    height = ctx.options["height_km"]
    return 0 if height is None else ctx.cm1.level_index(float(height))


def _level_suffix(ctx: TimeContext) -> str:
    height = ctx.options["height_km"]
    return "lowest" if height is None else f"z{float(height):g}km"


def partition_path(ctx: TimeContext) -> Path:
    """Where the partition of this time is cached."""
    name = f"wind_decomp_{ctx.tag}_k{_level(ctx):03d}_{ctx.options['variant']}_{round(ctx.seconds):06d}s.nc"
    return ctx.outdir.parent / DERIVED_DIR / name


def _cache_matches(path: Path, ctx: TimeContext) -> bool:
    """True if the cached file was computed with this time, level, variant and ROI rule."""
    opts = ctx.options
    try:
        with Dataset(path) as ds:
            same = (
                int(ds.level_index) == _level(ctx)
                and ds.variant == opts["variant"]
                and ds.roi_source == roi_source(opts["roi_km"], opts["roi_dbz"])
                and Path(ds.source_file).name == ctx.cm1.path.name
                and ds["time"].size == 1
                and abs(float(ds["time"][0]) - ctx.seconds) <= TIME_MATCH_S
            )
            if same and opts["roi_km"] is None:
                same = float(ds.roi_dbz_threshold) == float(opts["roi_dbz"])
            return bool(same)
    except (OSError, AttributeError, KeyError, IndexError):
        return False  # unreadable or from an older layout: recompute


def ensure_partition(ctx: TimeContext) -> Path:
    """Path of this time's partition file, computing and writing it if needed."""
    path = partition_path(ctx)
    if ctx.options["recompute"] or not path.exists() or not _cache_matches(path, ctx):
        opts = ctx.options
        roi_km = None if opts["roi_km"] is None else [float(v) for v in opts["roi_km"]]
        failures = write_partition(
            ctx.cm1,
            path,
            [ctx.time],
            level=_level(ctx),
            variant=opts["variant"],
            roi_km=roi_km,
            roi_dbz=float(opts["roi_dbz"]),
            workers=int(opts["workers"]),
        )
        if failures:
            # As in the legacy command line, a failed check is reported but the partition is kept:
            # on small or edge-dominated domains the unweighted KE cross terms can exceed their limit.
            log.warning("%d validation check(s) failed; see the diagnostics group of %s", failures, path)
    return path


# --------------------------------------------------------------------- view
def _coordinate_km(ds: Dataset, name: str) -> np.ndarray:
    var = ds[name]
    scale = {"km": 1.0, "m": 1e-3}[getattr(var, "units", "km")]
    return np.asarray(var[:], dtype=np.float64) * scale


def _view(ctx: TimeContext, ds: Dataset) -> tuple[int, int, int, int]:
    """(i0, i1, j0, j1) cell ranges to draw: extent_km, else the ROI, else inside the edge buffer."""
    ny, nx = ds["chi"].shape[1:]
    if ctx.options["extent_km"] is not None:
        ys, xs = ctx.window
        return xs.start, xs.stop, ys.start, ys.stop
    roi = [int(ds["diagnostics"][name][0]) for name in ("roi_i0", "roi_i1", "roi_j0", "roi_j1")]
    if not ctx.options["full_domain"] and roi[0] >= 0:
        return tuple(roi)
    # Keep at least two cells per axis on small grids.
    bx, by = min(EDGE_BUFFER_CELLS, (nx - 2) // 2), min(EDGE_BUFFER_CELLS, (ny - 2) // 2)
    return bx, nx - bx, by, ny - by


def _rounded_up(value: float, step: float) -> float:
    return max(step, step * np.ceil(value / step))


def _colorbar(fig, axes, mappable, label, fonts, extend="neither", aspect=45, ticks=None):
    bar = fig.colorbar(
        mappable, ax=axes, location="bottom", shrink=0.7, aspect=aspect, extend=extend, ticks=ticks
    )
    if ticks is not None:
        bar.ax.set_xticklabels([f"{t:g}" for t in ticks])
    bar.set_label(label, fontsize=fonts["cbar_label"])
    bar.ax.tick_params(labelsize=fonts["cbar_tick"], length=2)
    bar.outline.set_linewidth(0.4)


def _potential_contours(ax, x, y, field) -> None:
    """17 evenly spaced contours symmetric about zero, without the zero line (negative dashed)."""
    limit = float(np.nanmax(np.abs(field))) if np.isfinite(field).any() else 0.0
    if limit <= 0.0:
        return
    levels = [lv for lv in np.linspace(-limit, limit, POTENTIAL_LEVELS) if abs(lv) > 1e-9 * limit]
    ax.contour(x, y, field, levels=levels, colors=CONTOUR_COLOR, linewidths=0.5)


def _prspert_hpa(ctx: TimeContext, level: int, cells) -> np.ndarray | None:
    if not ctx.cm1.has("prspert"):
        return None
    data = ctx.cm1.read("prspert", ctx.time, level=level, dtype=np.float64)[cells]
    return data / PA_PER_HPA if ctx.cm1.units("prspert") == "Pa" else data


# --------------------------------------------------------------------- plot
@register(
    "wind_partition",
    WIND_CATEGORY,
    requires=["u", "v"],
    options={
        "height_km": None,
        "variant": "A",
        "roi_km": None,
        "roi_dbz": 25.0,
        "cref_dbz": 35.0,
        "full_domain": False,
        "vector_count": 20,
        "recompute": False,
        "workers": -1,
    },
)
def wind_partition(ctx: TimeContext) -> None:
    """Rotational, divergent and total wind, vorticity, divergence and p' (Lynch 1989 partition)."""
    path = ensure_partition(ctx)
    opts, fonts = ctx.options, ctx.fonts
    with Dataset(path) as ds:
        i0, i1, j0, j1 = _view(ctx, ds)
        cells = (slice(j0, j1), slice(i0, i1))
        corners = (slice(j0, j1 + 1), slice(i0, i1 + 1))
        x, y = _coordinate_km(ds, "xh")[i0:i1], _coordinate_km(ds, "yh")[j0:j1]
        xc, yc = _coordinate_km(ds, "xf")[i0 : i1 + 1], _coordinate_km(ds, "yf")[j0 : j1 + 1]

        def center_winds(part):
            uc, vc = winds_to_centers(
                np.asarray(ds[f"u_{part}"][0], dtype=np.float64),
                np.asarray(ds[f"v_{part}"][0], dtype=np.float64),
            )
            return uc[cells], vc[cells]

        u_psi, v_psi = center_winds("psi")
        u_chi, v_chi = center_winds("chi")
        u_h, v_h = center_winds("h")
        psi = np.asarray(ds["psi"][0][corners]) / POTENTIAL_SCALE
        chi = np.asarray(ds["chi"][0][cells]) / POTENTIAL_SCALE
        delta = np.asarray(ds["delta"][0][cells]) / VORTICITY_SCALE
        zeta = np.ma.filled(ds["zeta"][0][corners], np.nan) / VORTICITY_SCALE
        level, level_height_m, variant = int(ds.level_index), float(ds.level_height_m), ds.variant
    u_total, v_total = u_psi + u_chi + u_h, v_psi + v_chi + v_h  # the parts sum to the CM1 wind

    cref = ctx.cm1.read("cref", ctx.time, dtype=np.float64)[cells] if ctx.cm1.has("cref") else None
    prspert = _prspert_hpa(ctx, level, cells)

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 11.5), sharex=True, sharey=True, layout="constrained")

    speed_mesh = None
    titles = ("Rotational wind $V_\\psi$", "Divergent wind $V_\\chi$", "Total wind $V$")
    winds = ((u_psi, v_psi), (u_chi, v_chi), (u_total, v_total))
    for ax, (uf, vf), title in zip(axes[0], winds, titles, strict=True):
        speed_mesh = ax.pcolormesh(
            x, y, np.hypot(uf, vf), cmap=WIND.cmap, norm=WIND.norm, alpha=WIND.alpha,
            shading="nearest", rasterized=True,
        )  # fmt: skip
        ax.set_title(title, fontsize=fonts["title"], loc="left")
    _colorbar(fig, list(axes[0]), speed_mesh, WIND.label, fonts, extend=WIND.extend, ticks=WIND.ticks)

    _potential_contours(axes[0, 0], xc, yc, psi)
    _potential_contours(axes[0, 1], x, y, chi)

    count = max(1, int(opts["vector_count"]))
    stride = max(1, max(i1 - i0, j1 - j0) // count)
    sample = (slice(stride // 2, None, stride), slice(stride // 2, None, stride))
    # Scale to the 90th percentile, not the max, so isolated gusts don't shrink every arrow to a head.
    vector_speed = _rounded_up(float(np.nanpercentile(np.hypot(u_total, v_total), 90)), 5.0)
    vectors = axes[0, 2].quiver(
        x[sample[1]], y[sample[0]], u_total[sample], v_total[sample],
        color="white", pivot="mid", width=0.003, headwidth=3.5, headlength=4, headaxislength=3.5,
        angles="xy", scale_units="width", scale=vector_speed * count,  # vector_speed spans one stride
    )  # fmt: skip
    axes[0, 2].quiverkey(
        vectors, 0.84, 1.03, vector_speed, f"{vector_speed:g} m s$^{{-1}}$",
        labelpos="E", fontproperties={"size": fonts["legend"] - 4},
    )  # fmt: skip

    vort_limit = _rounded_up(
        max(float(np.nanpercentile(np.abs(zeta), 99.5)), float(np.percentile(np.abs(delta), 99.5))), 0.5
    )
    diverging = {"cmap": "RdBu_r", "vmin": -vort_limit, "vmax": vort_limit, "rasterized": True}
    axes[1, 0].pcolormesh(xc, yc, zeta, shading="nearest", **diverging)
    axes[1, 0].set_title("Vertical vorticity $\\zeta$", fontsize=fonts["title"], loc="left")
    vort_mesh = axes[1, 1].pcolormesh(x, y, delta, shading="nearest", **diverging)
    axes[1, 1].set_title("Divergence $\\delta$", fontsize=fonts["title"], loc="left")
    _colorbar(fig, [axes[1, 0], axes[1, 1]], vort_mesh, "$\\zeta$, $\\delta$ ($10^{-3}$ s$^{-1}$)", fonts)

    axes[1, 2].set_title("Pressure perturbation $p'$", fontsize=fonts["title"], loc="left")
    if prspert is not None and np.isfinite(prspert).any():
        p_limit = _rounded_up(float(np.nanpercentile(np.abs(prspert), 99.5)), 0.5)
        p_mesh = axes[1, 2].pcolormesh(
            x, y, prspert, cmap="RdBu_r", vmin=-p_limit, vmax=p_limit, shading="nearest", rasterized=True
        )
        # One panel wide, so halve the aspect to match the two-panel bar's thickness.
        _colorbar(fig, [axes[1, 2]], p_mesh, "$p'$ (hPa)", fonts, extend="both", aspect=22)
    else:
        axes[1, 2].text(0.5, 0.5, "no prspert in file", transform=axes[1, 2].transAxes, ha="center")

    cref_dbz = float(opts["cref_dbz"])
    draw_cref = cref is not None and np.isfinite(cref).any() and np.nanmax(cref) >= cref_dbz
    for row, row_axes in enumerate(axes):
        for ax in row_axes:
            if draw_cref:
                color = CREF_COLOR if row == 0 else CREF_COLOR_LIGHT_BG
                ax.contour(x, y, cref, levels=[cref_dbz], colors=color, linewidths=0.5)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=fonts["tick"], length=2)
    for ax in axes[1]:
        ax.set_xlabel("x (km)", fontsize=fonts["axis_label"])
    for ax in axes[:, 0]:
        ax.set_ylabel("y (km)", fontsize=fonts["axis_label"])
    axes[0, 0].set_xlim(x[0], x[-1])
    axes[0, 0].set_ylim(y[0], y[-1])

    handles = [
        Line2D(
            [], [], color=CREF_COLOR, linewidth=0.8,
            path_effects=[patheffects.withStroke(linewidth=1.6, foreground=CREF_COLOR_LIGHT_BG)],
            label=f"Composite reflectivity {cref_dbz:g} dBZ",
        ),
        Line2D(
            [], [], color=CONTOUR_COLOR, linewidth=0.5,
            label=f"$\\psi$, $\\chi$ contours ({POTENTIAL_LEVELS} even levels; negative dashed)",
        ),
    ]  # fmt: skip
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False, fontsize=fonts["legend"])
    fig.suptitle(
        f"Wind partition, variant {variant}: {ctx.run.display_name}, {time_label(ctx.seconds)}, "
        f"z = {level_height_m:g} m",
        fontsize=fonts["suptitle"],
    )
    ctx.save(fig, ctx.seconds, suffix=_level_suffix(ctx))
