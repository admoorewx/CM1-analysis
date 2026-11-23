"""Vertical momentum forcing: dynamic pressure-gradient force (DVPGF) and effective buoyancy (B).

Ported from the legacy ``vertical_momentum/plot_wforce.py``. The figures are
drawn from the derived ``wforce`` file of each output time, which
:mod:`cm1analysis.pressure` computes (Poisson solves for p'_d and p'_b). The
file is cached at::

    <output_dir>/<run>/pressure/derived/wforce_<run>_<time>s.nc

A plot reuses the cached file when it was computed with the same options (a
file that also holds the horizontal forces serves a request without them) and
otherwise computes and writes it first, so ``cm1plot run pressure RUN --times
...`` works end to end and batch jobs can be split by time. ``recompute: true``
forces a fresh computation (once per file and session).

The compute options (variant, ROI, buffer, tests, ...) are common to every
plot in the category, so both figures share one derived file; see
:class:`cm1analysis.pressure.ComputeOptions`. Unlike the stand-alone compute
function, ``horizontal`` defaults to true here so the horizontal figure needs
no second solve.

Files computed in section mode (``section_y_km``) hold a single x-z row; for
those the figures are stacked sections along that row instead of plan views.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from netCDF4 import Dataset

from ..pipeline import TimeContext
from ..pressure import REQUIRED, ComputeOptions, compute, stored_options, vforce
from ..registry import Category, common_options, register
from ._common import time_label

log = logging.getLogger(__name__)

PRESSURE = Category.PRESSURE

common_options(
    PRESSURE,
    {
        # Decomposition (ComputeOptions); changing any of these recomputes the derived file.
        "variant": "A",
        "splits": [],  # "loading", "shear", "strain" or "all"
        "roi_km": None,  # [xmin, xmax, ymin, ymax]; None: box around cref >= roi_dbz
        "roi_dbz": 25.0,
        "section_y_km": None,  # x-z section mode: solve and plot only the row nearest this y
        "section_x_km": None,  # [xmin, xmax] of the section; None: x range of the reflectivity box
        "buffer_km": 50.0,  # solved around the ROI but not written
        "ztop_km": 15.0,  # top of the statistics, below the damping layer
        "sensitivity": list(vforce.VARIANTS[1:]),  # variants compared with A; [] skips
        "horizontal": True,  # also compute HDPGF and HBPGF
        "along_wind": False,
        "crop_test": True,
        "frame_test": True,
        "double": False,
        "workers": -1,
        "recompute": False,  # ignore a cached derived file
        # Figures
        "height_km": 1.0,  # height of the plan views
        "y_km": None,  # row of the x-z sections; None: the strongest downdraft at height_km
        "cref_dbz": 35.0,  # reflectivity contour
        "x_range_km": None,  # [xmin, xmax] of the plots; None: the whole ROI
    },
)

FORCE_SCALE = 1e-2  # plotted in 10^-2 m s^-2
W_LEVELS = (-8.0, -4.0, -2.0, 2.0, 4.0, 8.0)
INK = "#1f2933"
CREF_COLOR = "#111111"
W_COLOR = "#3e4c59"
SECTION_COLOR = "#7b8794"
DASHED = (0, (4, 3))
W_LABEL = "w = ±2, 4, 8 m s$^{-1}$ (negative dashed)"
V_TITLES = ("Dynamic VPGF", "Effective buoyancy $B$", "DVPGF + $B$")
H_TITLES = ("Dynamic HPGF", "Buoyancy HPGF", "HDPGF + HBPGF", "Model $-\\rho_0^{-1}\\nabla_h p'$")
VECTOR_COUNT = 22

_FRESH: set[Path] = set()  # derived files computed in this session (so recompute runs once per file)


# ------------------------------------------------------------ derived file
def derived_path(ctx: TimeContext) -> Path:
    run = ctx.run.name if ctx.run is not None else ctx.cm1.path.parent.name
    return (
        ctx.settings.output_dir / run / "pressure" / "derived" / f"wforce_{run}_{round(ctx.seconds):06d}s.nc"
    )


def derived_file(ctx: TimeContext, horizontal: bool = False) -> Path:
    """The derived wforce file of this time: cached if it matches the options, else computed now."""
    options = ComputeOptions.from_mapping(ctx.options)
    if horizontal and not options.horizontal:
        options = replace(options, horizontal=True)
    path = derived_path(ctx)
    if path.exists() and (path in _FRESH or not ctx.options["recompute"]):
        stored = stored_options(path)
        if stored is not None and options.satisfied_by(stored):
            log.info("using cached %s", path)
            return path
        log.info("options differ from %s; recomputing", path.name)
    failures = compute(ctx.cm1, path, options, [ctx.time])
    if failures:
        log.warning("%s: %d validation check(s) failed; see its diagnostics group", path.name, failures)
    _FRESH.add(path)
    return path


# ------------------------------------------------------------------ plots
@register("vertical_forcing", PRESSURE, requires=list(REQUIRED))
def vertical_forcing(ctx: TimeContext) -> None:
    """DVPGF, effective buoyancy B and their sum: plan views at height_km and x-z sections."""
    with Dataset(derived_file(ctx)) as ds:
        fig = _plot_section(ds, ctx) if _is_section(ds) else _plot_time(ds, ctx)
    ctx.save(fig, ctx.seconds)


@register("horizontal_forcing", PRESSURE, requires=list(REQUIRED))
def horizontal_forcing(ctx: TimeContext) -> None:
    """HDPGF, HBPGF, their sum and the model's -(1/rho0) grad_h p' (x-component fill, vectors)."""
    with Dataset(derived_file(ctx, horizontal=True)) as ds:
        fig = _plot_horizontal_section(ds, ctx) if _is_section(ds) else _plot_horizontal(ds, ctx)
    ctx.save(fig, ctx.seconds)


# ---------------------------------------------------------------- helpers
def _km(ds, dim) -> np.ndarray:
    """Values of the coordinate of a dimension, in km."""
    var = next(ds[n] for n in ds.variables if ds[n].dimensions == (dim,))
    scale = 1e-3 if getattr(var, "units", "km") == "m" else 1.0
    return np.asarray(var[:], dtype=np.float64) * scale


def _rounded_up(value, step):
    return max(step, step * np.ceil(value / step))


def _is_section(ds) -> bool:
    return bool(getattr(ds, "section", 0)) or ds["dvpgf"].shape[2] == 1


def _style(ax, fonts) -> None:
    ax.tick_params(labelsize=fonts["tick"], colors=INK, length=2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)


def _title(ax, text, fonts) -> None:
    ax.set_title(text, color=INK, fontsize=fonts["title"], loc="left")


def _label(ax, fonts, x=None, y=None) -> None:
    if x:
        ax.set_xlabel(x, color=INK, fontsize=fonts["axis_label"])
    if y:
        ax.set_ylabel(y, color=INK, fontsize=fonts["axis_label"])


def _colorbar(fig, mesh, axes, label, fonts) -> None:
    bar = fig.colorbar(mesh, ax=axes, location="right", shrink=0.6, aspect=40)
    bar.set_label(label, color=INK, fontsize=fonts["cbar_label"])
    bar.ax.tick_params(labelsize=fonts["cbar_tick"], colors=INK, length=2)
    bar.outline.set_linewidth(0.4)


def _colors(limit) -> dict:
    return {"cmap": "RdBu_r", "vmin": -limit, "vmax": limit, "shading": "nearest", "rasterized": True}


def _suptitle(fig, ds, ctx, what: str) -> None:
    run = ctx.run.display_name if ctx.run is not None else ctx.cm1.path.parent.name
    fig.suptitle(
        f"{what}, variant {ds.variant}\n{run}, {time_label(ctx.seconds)}",
        color=INK,
        fontsize=ctx.fonts["suptitle"],
    )


def _legend(fig, handles, fonts) -> None:
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=len(handles),
        frameon=False,
        fontsize=fonts["legend"],
    )


def _plan_handles(cref_dbz) -> list:
    return [
        Line2D([], [], color=W_COLOR, linewidth=0.5, label=W_LABEL),
        Line2D([], [], color=CREF_COLOR, linewidth=0.7, label=f"Composite reflectivity {cref_dbz:g} dBZ"),
        Line2D([], [], color=SECTION_COLOR, linewidth=0.8, linestyle=DASHED, label="Cross-section row"),
    ]


def _section_handles(ds, cref_dbz) -> list:
    handles = [Line2D([], [], color=W_COLOR, linewidth=0.6, label=W_LABEL)]
    if "cref" in ds.variables:
        handles.append(
            Line2D([], [], color=SECTION_COLOR, linewidth=0.8, linestyle=DASHED, label=f"{cref_dbz:g} dBZ")
        )
    return handles


def _section_row(ds, ctx, y, zf_km) -> int:
    """Row of the x-z sections: y_km, or the strongest downdraft at the plan height."""
    if ctx.options["y_km"] is not None:
        return int(np.argmin(np.abs(y - ctx.options["y_km"])))
    k_plan = int(np.argmin(np.abs(zf_km - ctx.options["height_km"])))
    w = np.asarray(ds["w"][0, k_plan], dtype=np.float64)
    return int(np.unravel_index(np.argmin(w), w.shape)[0])


def _columns(ctx, x):
    """Columns inside x_range_km (all if unset), for the section-mode figures."""
    x_range = ctx.options["x_range_km"]
    if x_range is None:
        return slice(None)
    columns = (x >= x_range[0]) & (x <= x_range[1])
    if not columns.any():
        raise ValueError(f"x_range_km {list(x_range)} is outside the section ({x[0]:g}-{x[-1]:g} km)")
    return columns


def _cref_strip(fig, cell, ds, ctx, x, y, columns):
    """Composite reflectivity along the section row, as a line plot above the sections."""
    fonts = ctx.fonts
    top = fig.add_subplot(cell)
    if "cref" in ds.variables:
        cref = np.asarray(ds["cref"][0, 0], dtype=np.float64)[columns]
        top.plot(x, cref, color=INK, linewidth=1.2)
        top.axhline(ctx.options["cref_dbz"], color=SECTION_COLOR, linewidth=0.8, linestyle=DASHED)
        top.set_ylim(bottom=0)
    _label(top, fonts, y="cref (dBZ)")
    _title(top, f"Composite reflectivity along y = {y[0]:.0f} km", fonts)
    top.tick_params(labelbottom=False)
    _style(top, fonts)
    return top


# --------------------------------------------------------- vertical forces
V_LABEL = "Vertical acceleration ($10^{-2}$ m s$^{-2}$)"


def _plot_time(ds, ctx):
    """Plan views (top) and x-z sections (bottom) of DVPGF, B and DVPGF + B."""
    fonts, opts = ctx.fonts, ctx.options
    face_dims = ds["dvpgf"].dimensions
    x, y, zf_km = _km(ds, face_dims[3]), _km(ds, face_dims[2]), _km(ds, face_dims[1])
    k_top = int(ds.kmax_face)
    k_plan = int(np.argmin(np.abs(zf_km - opts["height_km"])))

    dvpgf = np.asarray(ds["dvpgf"][0], dtype=np.float64) / FORCE_SCALE
    B = np.asarray(ds["B"][0], dtype=np.float64) / FORCE_SCALE
    w = np.asarray(ds["w"][0], dtype=np.float64)
    fields = (dvpgf, B, dvpgf + B)
    j_section = _section_row(ds, ctx, y, zf_km)
    plan = [f[k_plan] for f in fields]
    section = [f[1 : k_top + 1, j_section, :] for f in fields]
    colors = _colors(_rounded_up(max(np.percentile(np.abs(a), 99.5) for a in plan + section), 0.5))
    cref = np.asarray(ds["cref"][0], dtype=np.float64) if "cref" in ds.variables else None

    fig = plt.figure(figsize=(18, 11.5), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.75])
    top_axes = [fig.add_subplot(grid[0, n]) for n in range(3)]
    for ax in top_axes[1:]:
        ax.sharex(top_axes[0])
        ax.sharey(top_axes[0])
    mesh = None
    for ax, field, title in zip(top_axes, plan, V_TITLES, strict=True):
        mesh = ax.pcolormesh(x, y, field, **colors)
        ax.contour(x, y, w[k_plan], levels=W_LEVELS, colors=W_COLOR, linewidths=0.5)
        if cref is not None and np.nanmax(cref) >= opts["cref_dbz"]:
            ax.contour(x, y, cref, levels=[opts["cref_dbz"]], colors=CREF_COLOR, linewidths=0.7)
        ax.axhline(y[j_section], color=SECTION_COLOR, linewidth=0.8, linestyle=DASHED)
        _title(ax, f"{title}, z = {zf_km[k_plan]:.2f} km", fonts)
        ax.set_aspect("equal")
        _label(ax, fonts, x="x (km)")
        _style(ax, fonts)
    _label(top_axes[0], fonts, y="y (km)")
    if opts["x_range_km"] is not None:
        top_axes[0].set_xlim(*opts["x_range_km"])

    bottom_axes = [fig.add_subplot(grid[1, n], sharex=top_axes[n]) for n in range(3)]
    for ax in bottom_axes[1:]:
        ax.sharey(bottom_axes[0])
    z_section = zf_km[1 : k_top + 1]
    for ax, field, title in zip(bottom_axes, section, V_TITLES, strict=True):
        ax.pcolormesh(x, z_section, field, **colors)
        ax.contour(
            x, z_section, w[1 : k_top + 1, j_section, :], levels=W_LEVELS, colors=W_COLOR, linewidths=0.5
        )
        _title(ax, f"{title}, y = {y[j_section]:.0f} km", fonts)
        _label(ax, fonts, x="x (km)")
        _style(ax, fonts)
    _label(bottom_axes[0], fonts, y="z (km)")

    _colorbar(fig, mesh, top_axes + bottom_axes, V_LABEL, fonts)
    _legend(fig, _plan_handles(opts["cref_dbz"]), fonts)
    _suptitle(fig, ds, ctx, "Vertical momentum forcing")
    return fig


def _plot_section(ds, ctx):
    """Section-mode file: reflectivity along the row, then DVPGF, B and their sum as x-z panels."""
    fonts = ctx.fonts
    face_dims = ds["dvpgf"].dimensions
    x, y, zf_km = _km(ds, face_dims[3]), _km(ds, face_dims[2]), _km(ds, face_dims[1])
    k_top = int(ds.kmax_face)
    z = zf_km[1 : k_top + 1]
    columns = _columns(ctx, x)
    x = x[columns]

    def row(name):
        return np.asarray(ds[name][0, 1 : k_top + 1, 0], dtype=np.float64)[:, columns]

    dvpgf, B = row("dvpgf") / FORCE_SCALE, row("B") / FORCE_SCALE
    w = row("w")
    fields = (dvpgf, B, dvpgf + B)
    colors = _colors(_rounded_up(max(np.percentile(np.abs(f), 99.5) for f in fields), 0.5))

    fig = plt.figure(figsize=(14, 12), layout="constrained")
    grid = fig.add_gridspec(4, 1, height_ratios=[0.3, 1.0, 1.0, 1.0])
    top = _cref_strip(fig, grid[0], ds, ctx, x, y, columns)
    axes = [fig.add_subplot(grid[n + 1], sharex=top) for n in range(3)]
    mesh = None
    for ax, field, title in zip(axes, fields, V_TITLES, strict=True):
        mesh = ax.pcolormesh(x, z, field, **colors)
        ax.contour(x, z, w, levels=W_LEVELS, colors=W_COLOR, linewidths=0.6)
        _title(ax, title, fonts)
        _label(ax, fonts, y="z (km)")
        _style(ax, fonts)
        if ax is not axes[-1]:
            ax.tick_params(labelbottom=False)
    _label(axes[-1], fonts, x="x (km)")
    top.set_xlim(x[0], x[-1])
    _colorbar(fig, mesh, axes, V_LABEL, fonts)
    _legend(fig, _section_handles(ds, ctx.options["cref_dbz"]), fonts)
    _suptitle(fig, ds, ctx, f"Vertical momentum forcing along y = {y[0]:.0f} km")
    return fig


# ------------------------------------------------------- horizontal forces
H_LABEL = "Eastward (x) acceleration ($10^{-2}$ m s$^{-2}$)"


def _horizontal_fields(ds):
    """(x, y) components at cell centers for the four columns; None where the model p' is absent."""

    def centers(name):
        fx = np.asarray(ds[f"{name}_x"][0], dtype=np.float64)
        fy = np.asarray(ds[f"{name}_y"][0], dtype=np.float64)
        return 0.5 * (fx[..., 1:] + fx[..., :-1]) / FORCE_SCALE, 0.5 * (fy[:, 1:] + fy[:, :-1]) / FORCE_SCALE

    d, b = centers("hdpgf"), centers("hbpgf")
    model = centers("model_hpgf") if "model_hpgf_x" in ds.variables else None
    return [d, b, (d[0] + b[0], d[1] + b[1]), model]


def _h_limit(arrays):
    values = np.concatenate([a[np.isfinite(a)].ravel() for a in arrays if a is not None])
    return _rounded_up(np.percentile(np.abs(values), 99.5), 0.5) if values.size else 1.0


def _empty_panel(ax, title, fonts):
    ax.text(0.5, 0.5, "model p' not in file", transform=ax.transAxes, ha="center", va="center", color=INK)
    _title(ax, title, fonts)
    _style(ax, fonts)


def _w_centers(w):
    """w from z-faces to cell centers (centers are mid-cell in CM1)."""
    return 0.5 * (w[1:] + w[:-1])


def _plot_horizontal(ds, ctx):
    """Plan views (fill: x-component, arrows: vector) and x-z sections of the horizontal forces."""
    fonts, opts = ctx.fonts, ctx.options
    face_dims = ds["dvpgf"].dimensions
    x, y, zf_km = _km(ds, face_dims[3]), _km(ds, face_dims[2]), _km(ds, face_dims[1])
    zc_km = _km(ds, ds["hdpgf_x"].dimensions[1])
    k_top = int(ds.kmax_face)  # cell centers below face kmax
    k_plan = int(np.argmin(np.abs(zc_km - opts["height_km"])))
    j_section = _section_row(ds, ctx, y, zf_km)
    fields = _horizontal_fields(ds)

    plan = [None if f is None else (f[0][k_plan], f[1][k_plan]) for f in fields]
    section = [None if f is None else f[0][:k_top, j_section, :] for f in fields]
    colors = _colors(_h_limit([p[0] for p in plan if p is not None] + section))
    cref = np.asarray(ds["cref"][0], dtype=np.float64) if "cref" in ds.variables else None
    w = np.asarray(ds["w"][0], dtype=np.float64)
    w_plan = w[int(np.argmin(np.abs(zf_km - zc_km[k_plan])))]

    stride = max(1, max(x.size, y.size) // VECTOR_COUNT)
    sample = (slice(stride // 2, None, stride), slice(stride // 2, None, stride))
    speeds = np.concatenate([np.hypot(*p).ravel() for p in plan if p is not None])
    speeds = speeds[np.isfinite(speeds)]
    vector_speed = _rounded_up(np.percentile(speeds, 90), 0.5) if speeds.size else 1.0
    spacing = stride * (x[1] - x[0]) if x.size > 1 else 1.0

    fig = plt.figure(figsize=(23, 11.5), layout="constrained")
    grid = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.75])
    top_axes = [fig.add_subplot(grid[0, n]) for n in range(4)]
    for ax in top_axes[1:]:
        ax.sharex(top_axes[0])
        ax.sharey(top_axes[0])
    mesh = vectors = None
    for ax, field, title in zip(top_axes, plan, H_TITLES, strict=True):
        title = f"{title}, z = {zc_km[k_plan]:.2f} km"
        if field is None:
            _empty_panel(ax, title, fonts)
            continue
        mesh = ax.pcolormesh(x, y, field[0], **colors)
        vectors = ax.quiver(
            x[sample[1]],
            y[sample[0]],
            field[0][sample],
            field[1][sample],
            color=INK,
            pivot="mid",
            width=0.003,
            headwidth=3.5,
            headlength=4,
            headaxislength=3.5,
            angles="xy",
            scale_units="xy",
            scale=vector_speed / spacing,
        )
        ax.contour(x, y, w_plan, levels=W_LEVELS, colors=W_COLOR, linewidths=0.5)
        if cref is not None and np.nanmax(cref) >= opts["cref_dbz"]:
            ax.contour(x, y, cref, levels=[opts["cref_dbz"]], colors=CREF_COLOR, linewidths=0.7)
        ax.axhline(y[j_section], color=SECTION_COLOR, linewidth=0.8, linestyle=DASHED)
        _title(ax, title, fonts)
        ax.set_aspect("equal")
        _label(ax, fonts, x="x (km)")
        _style(ax, fonts)
    _label(top_axes[0], fonts, y="y (km)")
    if opts["x_range_km"] is not None:
        top_axes[0].set_xlim(*opts["x_range_km"])
    if vectors is not None:
        top_axes[-1].quiverkey(
            vectors,
            0.80,
            1.12,
            vector_speed,
            f"{vector_speed:g} $\\times 10^{{-2}}$ m s$^{{-2}}$",
            labelpos="E",
            color=INK,
            labelcolor=INK,
            fontproperties={"size": fonts["tick"]},
        )

    bottom_axes = [fig.add_subplot(grid[1, n], sharex=top_axes[n]) for n in range(4)]
    for ax in bottom_axes[1:]:
        ax.sharey(bottom_axes[0])
    z_section = zc_km[:k_top]
    w_section = _w_centers(w[:, j_section, :])[:k_top]
    for ax, field, title in zip(bottom_axes, section, H_TITLES, strict=True):
        title = f"{title}, y = {y[j_section]:.0f} km"
        if field is None:
            _empty_panel(ax, title, fonts)
            continue
        ax.pcolormesh(x, z_section, field, **colors)
        ax.contour(x, z_section, w_section, levels=W_LEVELS, colors=W_COLOR, linewidths=0.5)
        _title(ax, title, fonts)
        _label(ax, fonts, x="x (km)")
        _style(ax, fonts)
    _label(bottom_axes[0], fonts, y="z (km)")
    _colorbar(fig, mesh, top_axes + bottom_axes, H_LABEL, fonts)
    _legend(fig, _plan_handles(opts["cref_dbz"]), fonts)
    _suptitle(fig, ds, ctx, "Horizontal pressure-gradient forces")
    return fig


def _plot_horizontal_section(ds, ctx):
    """Section-mode file: reflectivity along the row, then the four x-component panels."""
    fonts = ctx.fonts
    face_dims = ds["dvpgf"].dimensions
    x, y = _km(ds, face_dims[3]), _km(ds, face_dims[2])
    zc_km = _km(ds, ds["hdpgf_x"].dimensions[1])
    k_top = int(ds.kmax_face)
    z = zc_km[:k_top]
    columns = _columns(ctx, x)
    x = x[columns]
    fields = [None if f is None else f[0][:k_top, 0][:, columns] for f in _horizontal_fields(ds)]
    w = _w_centers(np.asarray(ds["w"][0], dtype=np.float64)[:, 0][:, columns])[:k_top]
    colors = _colors(_h_limit(fields))

    fig = plt.figure(figsize=(14, 15), layout="constrained")
    grid = fig.add_gridspec(5, 1, height_ratios=[0.3, 1.0, 1.0, 1.0, 1.0])
    top = _cref_strip(fig, grid[0], ds, ctx, x, y, columns)
    axes = [fig.add_subplot(grid[n + 1], sharex=top) for n in range(4)]
    mesh = None
    for ax, field, title in zip(axes, fields, H_TITLES, strict=True):
        if field is None:
            _empty_panel(ax, title, fonts)
        else:
            mesh = ax.pcolormesh(x, z, field, **colors)
            ax.contour(x, z, w, levels=W_LEVELS, colors=W_COLOR, linewidths=0.6)
            _title(ax, title, fonts)
            _style(ax, fonts)
        _label(ax, fonts, y="z (km)")
        if ax is not axes[-1]:
            ax.tick_params(labelbottom=False)
    _label(axes[-1], fonts, x="x (km)")
    top.set_xlim(x[0], x[-1])
    _colorbar(fig, mesh, axes, H_LABEL, fonts)
    _legend(fig, _section_handles(ds, ctx.options["cref_dbz"]), fonts)
    _suptitle(fig, ds, ctx, f"Horizontal pressure-gradient forces (x-component) along y = {y[0]:.0f} km")
    return fig
