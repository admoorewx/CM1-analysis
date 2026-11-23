"""Plan-view plots.

Ported from the legacy ``planar/`` scripts (cm1_plot, cm1_four_panel,
swath_refl_contour, derecho_lasher_output, derecho_exp_compare,
cm1_derecho_compare, swath_coldpool_compare) and the plan views that lived in
``cross_section/`` (hydrometeor_flux_planar, swath_max_downdraft). Duplicated figures are merged and
every panel of a given field uses the same style (``_common.FIELD_STYLES``).

Every plot accepts the common option ``extent_km: [xmin, xmax, ymin, ymax]``
to draw (and read) only part of the domain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..pipeline import ExperimentContext, Frame, RunContext, RunInput, TimeContext
from ..registry import Category, Scope, common_options, register, satisfied
from ..style import SEVERE_WIND, SIG_SEVERE_WIND
from ._common import (
    FIELD_STYLES,
    FLUX_DOWN_LEVELS,
    SPECIES_COLORS,
    SPECIES_NAMES,
    WIND,
    FieldStyle,
    colorbar,
    flux_style,
    km_axes,
    mark_maximum,
    panel_grid,
    planview,
    same_grid,
    species_style,
    time_label,
)

PLANAR = Category.PLANAR
TIME_MATCH_S = 0.5

common_options(PLANAR, {"extent_km": None})

COOLING_TERMS = {
    "cbudevp": "Rain evaporation",
    "cbudmltg": "Graupel melting",
    "cbudmlth": "Hail melting",
    "cbudsubg": "Graupel sublimation",
    "cbudsubh": "Hail sublimation",
}
HYDROMETEOR_FIELDS = ("qc", "qr", "qi", "qs", "qg", "qhl", "qh")


# ----------------------------------------------------------------- helpers
def _level_label(frame: Frame, k: int) -> str:
    z = frame.cm1.z_km[k]
    where = f"{z * 1000:.0f} m" if z < 1.0 else f"{z:.2f} km"
    return f"lowest level ({where})" if k == 0 else where


def _title(ctx: TimeContext, what: str) -> str:
    return f"{what}\n{ctx.run.display_name}, {time_label(ctx.seconds)}"


def _single(ctx: TimeContext, data: np.ndarray, style: FieldStyle, title: str):
    fig, axs = panel_grid(1, 1, panel=(8.5, 7.5))
    ax = axs[0, 0]
    colorbar(fig, planview(ax, ctx.x, ctx.y, data, style), ax, style, ctx.fonts)
    km_axes(ax, ctx.fonts)
    ax.set_title(_title(ctx, title), fontsize=ctx.fonts["title"])
    return fig, ax


def _panel(fig, ax, frame: Frame, data, style: FieldStyle, title: str, fonts, xlabel=True, ylabel=True):
    mappable = planview(ax, frame.x, frame.y, data, style)
    colorbar(fig, mappable, ax, style, fonts)
    km_axes(ax, fonts, xlabel, ylabel)
    ax.set_title(title, fontsize=fonts["title"])
    return mappable


def _heights(ctx: TimeContext) -> list[tuple[int, str]]:
    """(level index, file-name suffix) for each entry of the heights_km option.

    0 means the lowest model level. Each figure is named after the requested
    height (z1km, ...) so names match across runs with different vertical
    grids; the title gives the actual height of the nearest level. Heights
    that land on the same level are drawn once.
    """
    heights = ctx.options["heights_km"]
    if isinstance(heights, int | float):
        heights = [heights]
    levels, seen = [], set()
    for h in heights:
        h = float(h)
        k = 0 if h == 0.0 else ctx.level(h)
        if k not in seen:
            seen.add(k)
            levels.append((k, "lowest" if h == 0.0 else f"z{h:g}km"))
    return levels


def _at_heights(ctx: TimeContext, field, style: FieldStyle, what: str) -> None:
    """One single-panel figure per requested height; field(k) returns the 2-D field at level k."""
    for k, suffix in _heights(ctx):
        fig, _ = _single(ctx, field(k), style, f"{what}, {_level_label(ctx, k)}")
        ctx.save(fig, ctx.seconds, suffix=suffix)


def _darken(color: str, factor: float = 0.55) -> tuple[float, float, float]:
    """A darker shade of a matplotlib color, for lines drawn over a fill of that color."""
    from matplotlib.colors import to_rgb

    return tuple(factor * c for c in to_rgb(color))


def _hide_unused(axs, used: int) -> None:
    for ax in axs.flat[used:]:
        ax.set_visible(False)


def _on_interval(seconds: float, interval_s: float) -> bool:
    return abs(seconds - interval_s * round(seconds / interval_s)) <= TIME_MATCH_S


# ------------------------------------------------------ single-time panels
@register("cref", PLANAR, requires=["cref"])
def cref(ctx: TimeContext) -> None:
    """Composite reflectivity."""
    fig, _ = _single(ctx, ctx.read("cref"), FIELD_STYLES["cref"], "Composite reflectivity")
    ctx.save(fig, ctx.seconds)


@register("wind10", PLANAR, requires=["low_level_wind"], options={"arrow_stride": 25})
def wind10(ctx: TimeContext) -> None:
    """Low-level (10-m when available) wind speed with direction arrows."""
    wind = ctx.wind()
    fig, ax = _single(ctx, wind.speed, WIND, f"{wind.label} wind")
    s = max(1, int(ctx.options["arrow_stride"]))
    ax.quiver(ctx.x[::s], ctx.y[::s], wind.u[::s, ::s], wind.v[::s, ::s], color="k", width=0.0015)
    ctx.save(fig, ctx.seconds)


@register("thpert", PLANAR, requires=["thpert"], options={"heights_km": [0.0, 1.0, 3.0]})
def thpert(ctx: TimeContext) -> None:
    """Potential temperature perturbation at each height in heights_km (0 = lowest level)."""
    _at_heights(
        ctx,
        lambda k: ctx.read("thpert", level=k),
        FIELD_STYLES["thpert"],
        "Potential temperature perturbation",
    )


@register("prspert", PLANAR, requires=["prspert"], options={"heights_km": [0.0, 1.0, 3.0, 5.0]})
def prspert(ctx: TimeContext) -> None:
    """Pressure perturbation at each height in heights_km (0 = lowest level)."""
    _at_heights(
        ctx, lambda k: ctx.read("prspert", level=k) / 100.0, FIELD_STYLES["prspert"], "Pressure perturbation"
    )


@register("sws", PLANAR, requires=["sws"])
def sws(ctx: TimeContext) -> None:
    """CM1's own maximum lowest-level wind speed (sws), with the severe-wind contour."""
    data = ctx.read("sws")
    fig, ax = _single(ctx, data, WIND, ctx.cm1.long_name("sws").capitalize())
    if np.nanmax(data) >= SEVERE_WIND:
        ax.contour(ctx.x, ctx.y, data, levels=[SEVERE_WIND], colors="k", linewidths=0.5)
    ctx.save(fig, ctx.seconds)


@register("max_w_5km", PLANAR, requires=["sus"])
def max_w_5km(ctx: TimeContext) -> None:
    """CM1's maximum vertical velocity at 5 km AGL (sus)."""
    fig, _ = _single(ctx, ctx.read("sus"), FIELD_STYLES["sus"], ctx.cm1.long_name("sus").capitalize())
    ctx.save(fig, ctx.seconds)


@register("sfc_pressure", PLANAR, requires=["surface_pressure"])
def sfc_pressure(ctx: TimeContext) -> None:
    """Surface pressure (psfc, or the lowest-level pressure)."""
    from ..diagnostics import surface_pressure

    data, which = surface_pressure(ctx.cm1, ctx.time, ctx.window)
    fig, _ = _single(ctx, data, FIELD_STYLES["pressure"], f"{which} pressure")
    ctx.save(fig, ctx.seconds)


@register("buoyancy", PLANAR, requires=["buoyancy"], options={"heights_km": [0.0, 1.0, 3.0]})
def buoyancy(ctx: TimeContext) -> None:
    """Buoyancy from CM1's base state, including water loading, at each height in heights_km."""
    from ..diagnostics import buoyancy as compute

    _at_heights(
        ctx, lambda k: compute(ctx.cm1, ctx.time, k, ctx.window), FIELD_STYLES["buoyancy"], "Buoyancy"
    )


# ------------------------------------------------------- multi-panel times
@register("surface", PLANAR, requires=["cref", "low_level_wind", "surface_pressure", "t2"])
def surface(ctx: TimeContext) -> None:
    """Four panels: reflectivity, low-level wind, surface pressure, 2-m temperature."""
    from ..diagnostics import surface_pressure

    wind = ctx.wind()
    pressure, which = surface_pressure(ctx.cm1, ctx.time, ctx.window)
    panels = [
        (ctx.read("cref"), FIELD_STYLES["cref"], "Composite reflectivity"),
        (wind.speed, WIND, f"{wind.label} wind"),
        (pressure, FIELD_STYLES["pressure"], f"{which} pressure"),
        (ctx.read("t2"), FIELD_STYLES["t2"], "2-m temperature"),
    ]
    fig, axs = panel_grid(2, 2)
    for n, (ax, (data, style, title)) in enumerate(zip(axs.flat, panels, strict=True)):
        _panel(fig, ax, ctx, data, style, title, ctx.fonts, xlabel=n >= 2, ylabel=n % 2 == 0)
    fig.suptitle(f"{ctx.run.display_name}, {time_label(ctx.seconds)}", fontsize=ctx.fonts["suptitle"])
    ctx.save(fig, ctx.seconds)


@register(
    "hydrometeors",
    PLANAR,
    requires=["cref", ("qv", *HYDROMETEOR_FIELDS)],
    options={"top_km": None, "include_vapor": True},
)
def hydrometeors(ctx: TimeContext) -> None:
    """Column-maximum mixing ratio of every species in the file, beside reflectivity and wind."""
    top = ctx.options["top_km"]
    panels = [(ctx.read("cref"), FIELD_STYLES["cref"], "Composite reflectivity")]
    if satisfied(ctx.cm1, "low_level_wind"):
        wind = ctx.wind()
        panels.append((wind.speed, WIND, f"{wind.label} wind"))
    if ctx.cm1.has("cpc"):
        panels.append((ctx.read("cpc"), FIELD_STYLES["cpc"], "Cold pool intensity"))
    species = (["qv"] if ctx.options["include_vapor"] and ctx.cm1.has("qv") else []) + ctx.cm1.hydrometeors
    for name in species:
        panels.append(
            (
                1000.0 * ctx.column_max(name, top),
                species_style(name),
                f"{SPECIES_NAMES.get(name, name)} ({name}), column max",
            )
        )

    ncols = 3
    nrows = -(-len(panels) // ncols)
    fig, axs = panel_grid(nrows, ncols)
    for n, (ax, (data, style, title)) in enumerate(zip(axs.flat, panels, strict=False)):
        _panel(
            fig,
            ax,
            ctx,
            data,
            style,
            title,
            ctx.fonts,
            xlabel=n >= len(panels) - ncols,
            ylabel=n % ncols == 0,
        )
    _hide_unused(axs, len(panels))
    fig.suptitle(f"{ctx.run.display_name}, {time_label(ctx.seconds)}", fontsize=ctx.fonts["suptitle"])
    ctx.save(fig, ctx.seconds)


@register(
    "hydrometeor_flux",
    PLANAR,
    requires=["cref", "air_density", "w_scalar", HYDROMETEOR_FIELDS],
    options={"top_km": None, "refl_dbz": 35.0},
)
def hydrometeor_flux(ctx: TimeContext) -> None:
    """Column-max upward (filled) and column-min downward (dashed) mass flux rho*q*w, one figure per species.

    Transport by the air motion only: sedimentation is not included because CM1
    does not write fall speeds by default, so below cloud base the true
    downward flux is larger than shown.
    """
    from matplotlib.lines import Line2D

    from ..diagnostics import column_mass_flux_extremes

    species = ctx.cm1.hydrometeors
    fmax, fmin = column_mass_flux_extremes(ctx.cm1, ctx.time, species, ctx.window, ctx.options["top_km"])
    cref_field, dbz = ctx.read("cref"), ctx.options["refl_dbz"]
    for name in species:
        color = SPECIES_COLORS.get(name, "gray")
        title = f"{SPECIES_NAMES.get(name, name)} ({name}) vertical mass flux"
        fig, ax = _single(ctx, np.maximum(fmax[name], 0.0), flux_style(name), title)
        handles = []
        down = np.minimum(fmin[name], 0.0)
        if np.nanmin(down) <= FLUX_DOWN_LEVELS[-1]:
            dark = _darken(color)
            lines = ax.contour(
                ctx.x, ctx.y, down, levels=FLUX_DOWN_LEVELS, colors=[dark], linestyles="--", linewidths=0.8
            )
            ax.clabel(lines, fmt=lambda v: f"{abs(v):g}", fontsize=ctx.fonts["contour_label"])
            handles.append(Line2D([], [], color=dark, ls="--", label="Downward flux (g m$^{-2}$ s$^{-1}$)"))
        if np.nanmax(cref_field) >= dbz:
            ax.contour(ctx.x, ctx.y, cref_field, levels=[dbz], colors="0.45", linewidths=1.0)
            handles.append(Line2D([], [], color="0.45", label=f"{dbz:g} dBZ"))
        if handles:
            ax.legend(handles=handles, loc="upper left", fontsize=ctx.fonts["legend"], framealpha=0.8)
        ctx.save(fig, ctx.seconds, suffix=name)


# ------------------------------------------------------------- run scope
@register("overview", PLANAR, scope=Scope.RUN, requires=["cref", "low_level_wind"])
def overview(ctx: RunContext) -> None:
    """Four panels per time: reflectivity, cold pool, low-level wind, and the wind swath so far."""
    swath = None
    for frame in ctx.frames():
        wind = frame.wind()
        swath = wind.speed if swath is None else np.fmax(swath, wind.speed)
        if not ctx.selected(frame.seconds):
            continue
        if frame.cm1.has("cpc"):
            cold = (frame.read("cpc"), FIELD_STYLES["cpc"], "Cold pool intensity")
        else:
            cold = (frame.read("thpert", level=0), FIELD_STYLES["thpert"], r"Lowest-level $\theta'$")
        panels = [
            (frame.read("cref"), FIELD_STYLES["cref"], "Composite reflectivity"),
            cold,
            (wind.speed, WIND, f"{wind.label} wind"),
            (swath, WIND, f"{wind.label} wind swath"),
        ]
        fig, axs = panel_grid(2, 2)
        for n, (ax, (data, style, title)) in enumerate(zip(axs.flat, panels, strict=True)):
            _panel(fig, ax, frame, data, style, title, ctx.fonts, xlabel=n >= 2, ylabel=n % 2 == 0)
        mark_maximum(axs[1, 1], frame.x, frame.y, swath, ctx.fonts, "m/s")
        fig.suptitle(f"{ctx.run.display_name}, {time_label(frame.seconds)}", fontsize=ctx.fonts["suptitle"])
        ctx.save(fig, frame.seconds)


@register(
    "wind_swath",
    PLANAR,
    scope=Scope.RUN,
    requires=["low_level_wind", "cref"],
    options={"refl_dbz": 45.0, "interval_s": 7200.0},
)
def wind_swath(ctx: RunContext) -> None:
    """Accumulated low-level wind swath with reflectivity outlines at a fixed interval."""
    swath, outlines, label = None, [], ""
    for frame in ctx.frames():
        wind = frame.wind()
        swath = wind.speed if swath is None else np.fmax(swath, wind.speed)
        label = wind.label
        if _on_interval(frame.seconds, ctx.options["interval_s"]):
            outlines.append(frame.read("cref"))
        x, y = frame.x, frame.y

    fig, axs = panel_grid(1, 1, panel=(9.0, 8.5))
    ax = axs[0, 0]
    colorbar(fig, planview(ax, x, y, swath, WIND), ax, WIND, ctx.fonts)
    for cref_field in outlines:
        if np.nanmax(cref_field) >= ctx.options["refl_dbz"]:
            ax.contour(
                x,
                y,
                cref_field,
                levels=[ctx.options["refl_dbz"]],
                colors="k",
                linestyles="dashed",
                linewidths=1.5,
            )
    km_axes(ax, ctx.fonts)
    hours = ctx.options["interval_s"] / 3600.0
    ax.set_title(
        f"Accumulated {label} maximum wind swath\nwith {hours:g}-hourly {ctx.options['refl_dbz']:g} dBZ "
        f"outlines, {ctx.run.display_name}",
        fontsize=ctx.fonts["title"],
    )
    ctx.save(fig)


@register(
    "refl_progression",
    PLANAR,
    scope=Scope.RUN,
    requires=["cref"],
    options={"min_dbz": 35.0, "interval_s": 7200.0},
)
def refl_progression(ctx: RunContext) -> None:
    """Reflectivity above a threshold at a fixed interval, overlaid to show the system's track."""
    snapshots = []
    for frame in ctx.frames():
        if _on_interval(frame.seconds, ctx.options["interval_s"]):
            snapshots.append((frame.seconds, frame.read("cref")))
        x, y = frame.x, frame.y
    if not snapshots:
        raise ValueError(f"no output times fall on the {ctx.options['interval_s']:g} s interval")

    fig, axs = panel_grid(1, 1, panel=(9.0, 8.0))
    ax = axs[0, 0]
    mappable = None
    for _, field in reversed(snapshots):  # earliest drawn last, so it is on top
        masked = np.ma.masked_less(field, ctx.options["min_dbz"])
        mappable = planview(ax, x, y, masked, FIELD_STYLES["cref"])
    colorbar(fig, mappable, ax, FIELD_STYLES["cref"], ctx.fonts)
    km_axes(ax, ctx.fonts)
    hours = ", ".join(f"{t / 3600:g} h" for t, _ in snapshots)
    ax.set_title(
        f"Composite reflectivity progression, {ctx.run.display_name}\n({hours})", fontsize=ctx.fonts["title"]
    )
    ctx.save(fig)


@register(
    "cooling_budget",
    PLANAR,
    scope=Scope.RUN,
    requires=[tuple(COOLING_TERMS), "cref", "low_level_wind"],
    options={"top_km": None, "wind_contour": SIG_SEVERE_WIND},
)
def cooling_budget(ctx: RunContext) -> None:
    """Accumulated column-minimum cold-pool cooling terms (cbud*) with the wind swath outlined."""
    top, contour = ctx.options["top_km"], ctx.options["wind_contour"]
    acc: dict[str, np.ndarray] = {}
    swath = coldest = None
    for frame in ctx.frames():
        terms = [t for t in COOLING_TERMS if frame.cm1.has(t)]
        for term in terms:
            value = frame.column_min(term, top)
            acc[term] = value if term not in acc else np.fmin(acc[term], value)
        wind = frame.wind()
        swath = wind.speed if swath is None else np.fmax(swath, wind.speed)
        if frame.cm1.has("thpert"):
            low = frame.read("thpert", level=0)
            coldest = low if coldest is None else np.fmin(coldest, low)
        if not ctx.selected(frame.seconds):
            continue

        panels = [
            (frame.read("cref"), FIELD_STYLES["cref"], "Composite reflectivity", False),
            (swath, WIND, f"{wind.label} wind swath", False),
        ]
        if coldest is not None:
            panels.append((coldest, FIELD_STYLES["coldpool_min"], r"Lowest-level minimum $\theta'$", True))
        panels += [(acc[t], FIELD_STYLES["cooling"], f"{COOLING_TERMS[t]} ({t})", True) for t in terms]

        ncols = 3
        nrows = -(-len(panels) // ncols)
        fig, axs = panel_grid(nrows, ncols)
        for n, (ax, (data, style, title, outline)) in enumerate(zip(axs.flat, panels, strict=False)):
            _panel(
                fig,
                ax,
                frame,
                data,
                style,
                title,
                ctx.fonts,
                xlabel=n >= len(panels) - ncols,
                ylabel=n % ncols == 0,
            )
            if outline and contour is not None and np.nanmax(swath) >= contour:
                ax.contour(frame.x, frame.y, swath, levels=[contour], colors="cyan", linewidths=0.5)
        _hide_unused(axs, len(panels))
        fig.suptitle(
            f"Accumulated column-minimum cooling, {ctx.run.display_name}, {time_label(frame.seconds)}",
            fontsize=ctx.fonts["suptitle"],
        )
        ctx.save(fig, frame.seconds)


@register(
    "downdraft_swath",
    PLANAR,
    scope=Scope.RUN,
    requires=["w_scalar", "cref"],
    options={"top_km": 5.0, "refl_dbz": 35.0, "interval_s": 3600.0, "frames": True},
)
def downdraft_swath(ctx: RunContext) -> None:
    """Running minimum of column-minimum w (the strongest downdraft so far), per time and final.

    ``top_km`` caps the column search to keep upper-level compensating
    subsidence out (None searches the whole column). Frames show the
    current reflectivity outline; the final figure shows outlines every
    ``interval_s``.
    """
    top, dbz = ctx.options["top_km"], ctx.options["refl_dbz"]
    depth = "full column" if top is None else f"below {top:g} km"
    style = FIELD_STYLES["downdraft"]
    swath, outlines, last = None, [], 0.0

    def draw(x, y, outline_fields, seconds, extra=""):
        fig, axs = panel_grid(1, 1, panel=(9.0, 8.0))
        ax = axs[0, 0]
        colorbar(fig, planview(ax, x, y, swath, style), ax, style, ctx.fonts)
        for field in outline_fields:
            if np.nanmax(field) >= dbz:
                ax.contour(x, y, field, levels=[dbz], colors="k", linestyles="dashed", linewidths=1.2)
        km_axes(ax, ctx.fonts)
        ax.set_title(
            f"Accumulated strongest downdraft ({depth}), {ctx.run.display_name}\n"
            f"through {time_label(seconds)}, min w = {np.nanmin(swath):.1f} m s$^{{-1}}${extra}",
            fontsize=ctx.fonts["title"],
        )
        return fig

    for frame in ctx.frames():
        name = "winterp" if frame.cm1.has("winterp") else "w"
        swath = frame.column_min(name, top) if swath is None else np.fmin(swath, frame.column_min(name, top))
        cref_field = frame.read("cref")
        if _on_interval(frame.seconds, ctx.options["interval_s"]):
            outlines.append(cref_field)
        x, y, last = frame.x, frame.y, frame.seconds
        if ctx.options["frames"] and ctx.selected(frame.seconds):
            ctx.save(draw(x, y, [cref_field], frame.seconds), frame.seconds)

    hours = ctx.options["interval_s"] / 3600.0
    note = f"\n{dbz:g} dBZ outlines every {hours:g} h" if outlines else ""
    ctx.save(draw(x, y, outlines, last, note))


# ------------------------------------------------------ experiment scope
@dataclass(frozen=True)
class _Row:
    style: str
    diff_style: str
    needs: str
    title: str


ROWS = {
    "cref": _Row("cref", "cref_diff", "cref", "Composite reflectivity"),
    "cpc": _Row("cpc", "wind_diff", "cpc", "Cold pool intensity"),
    "thpert": _Row("thpert", "thpert_diff", "thpert", r"Lowest-level $\theta'$"),
    "wind": _Row("wind", "wind_diff", "low_level_wind", "Low-level wind"),
    "wind_swath": _Row("wind", "wind_diff", "low_level_wind", "Low-level wind swath"),
    "coldpool_swath": _Row("coldpool_min", "thpert_diff", "thpert", r"Minimum lowest-level $\theta'$"),
}


class _RunState:
    """Running swaths for one run; updated once per time, read by the rows."""

    def __init__(self) -> None:
        self.wind_max = None
        self.thpert_min = None

    def update(self, frame: Frame, rows) -> dict[str, np.ndarray]:
        """Advance the swaths and return this time's field for each row."""
        fields = {}
        wind = frame.wind() if {"wind", "wind_swath"} & set(rows) else None
        low = frame.read("thpert", level=0) if {"thpert", "coldpool_swath"} & set(rows) else None
        if wind is not None:
            self.wind_max = wind.speed if self.wind_max is None else np.fmax(self.wind_max, wind.speed)
        if low is not None:
            self.thpert_min = low if self.thpert_min is None else np.fmin(self.thpert_min, low)
        for row in rows:
            fields[row] = {
                "cref": lambda: frame.read("cref"),
                "cpc": lambda: frame.read("cpc"),
                "thpert": lambda: low,
                "wind": lambda: wind.speed,
                "wind_swath": lambda: self.wind_max,
                "coldpool_swath": lambda: self.thpert_min,
            }[row]()
        return fields


def _check_rows(ctx: ExperimentContext, rows, runs: tuple[RunInput, ...]) -> None:
    unknown = [r for r in rows if r not in ROWS]
    if unknown:
        raise ValueError(f"unknown row(s) {unknown}; choose from {sorted(ROWS)}")
    from ..io import CM1File

    for run in runs:
        with CM1File(run.files[0]) as cm1:
            missing = [r for r in rows if not satisfied(cm1, ROWS[r].needs)]
        if missing:
            raise ValueError(f"run {run.name} cannot provide row(s) {missing}")


def _frames_at(ctx: ExperimentContext, entries, run_names):
    """Open each run's file for one common time; yields (name, frame) and closes the files."""
    from ..io import CM1File

    for name in run_names:
        path, _ = entries[name]
        with CM1File(path) as cm1:
            yield name, ctx.open_frame(entries[name], cm1)


@register(
    "compare",
    PLANAR,
    scope=Scope.EXPERIMENT,
    requires=["cref", "low_level_wind"],
    options={"rows": ["cref", "cpc", "wind_swath"]},
)
def compare(ctx: ExperimentContext) -> None:
    """One column per run (reference first), one row per field, at every common time."""
    rows = list(ctx.options["rows"])
    runs = ctx.all_runs
    _check_rows(ctx, rows, runs)
    states = {run.name: _RunState() for run in runs}
    labels = {run.name: run.display_name for run in runs}

    for seconds, entries in ctx.common_times(runs):
        plot = ctx.selected(seconds)
        if plot:
            fig, axs = panel_grid(len(rows), len(runs), panel=(4.6, 4.0))
            row_mappables = {}
        for c, (name, frame) in enumerate(_frames_at(ctx, entries, [r.name for r in runs])):
            fields = states[name].update(frame, rows)
            if not plot:
                continue
            for r, row in enumerate(rows):
                ax = axs[r, c]
                row_mappables[row] = planview(
                    ax, frame.x, frame.y, fields[row], FIELD_STYLES[ROWS[row].style]
                )
                km_axes(ax, ctx.fonts, xlabel=r == len(rows) - 1, ylabel=c == 0)
                if c == 0:
                    ax.set_ylabel(f"{ROWS[row].title}\ny (km)", fontsize=ctx.fonts["axis_label"])
            axs[0, c].set_title(labels[name], fontsize=ctx.fonts["title"])
        if plot:
            for r, row in enumerate(rows):
                style = FIELD_STYLES[ROWS[row].style]
                colorbar(fig, row_mappables[row], list(axs[r, :]), style, ctx.fonts)
            fig.suptitle(f"{ctx.tag}, {time_label(seconds)}", fontsize=ctx.fonts["suptitle"])
            ctx.save(fig, seconds)


def _require_reference(ctx: ExperimentContext) -> RunInput:
    if ctx.reference is None:
        raise ValueError(f"{ctx.spec.name} needs a reference run (config 'reference' or --reference)")
    if not ctx.runs:
        raise ValueError(f"{ctx.spec.name} needs at least one run besides the reference")
    return ctx.reference


@register(
    "compare_diff",
    PLANAR,
    scope=Scope.EXPERIMENT,
    requires=["cref", "low_level_wind", "thpert"],
    options={"rows": ["cref", "wind_swath", "thpert"]},
)
def compare_diff(ctx: ExperimentContext) -> None:
    """Each run beside the reference and their difference (run minus reference), per time."""
    ref = _require_reference(ctx)
    rows = list(ctx.options["rows"])
    for run in ctx.runs:
        pair = (ref, run)
        _check_rows(ctx, rows, pair)
        states = {ref.name: _RunState(), run.name: _RunState()}
        for seconds, entries in ctx.common_times(pair):
            frames, fields = {}, {}
            for name, frame in _frames_at(ctx, entries, [ref.name, run.name]):
                fields[name] = states[name].update(frame, rows)
                frames[name] = frame
            if not ctx.selected(seconds):
                continue
            same_grid(frames[ref.name], frames[run.name])
            frame = frames[ref.name]
            fig, axs = panel_grid(len(rows), 3, panel=(4.8, 4.1))
            for r, row in enumerate(rows):
                spec = ROWS[row]
                a, b = fields[ref.name][row], fields[run.name][row]
                last = r == len(rows) - 1
                _panel(
                    fig,
                    axs[r, 0],
                    frame,
                    a,
                    FIELD_STYLES[spec.style],
                    f"{ref.display_name}: {spec.title}",
                    ctx.fonts,
                    xlabel=last,
                )
                _panel(
                    fig,
                    axs[r, 1],
                    frame,
                    b,
                    FIELD_STYLES[spec.style],
                    f"{run.display_name}: {spec.title}",
                    ctx.fonts,
                    xlabel=last,
                    ylabel=False,
                )
                _panel(
                    fig,
                    axs[r, 2],
                    frame,
                    b - a,
                    FIELD_STYLES[spec.diff_style],
                    f"{run.display_name} minus {ref.display_name}",
                    ctx.fonts,
                    xlabel=last,
                    ylabel=False,
                )
            fig.suptitle(
                f"{ctx.tag}: {run.display_name} vs {ref.display_name}, {time_label(seconds)}",
                fontsize=ctx.fonts["suptitle"],
            )
            ctx.save(fig, seconds, suffix=run.name)


@register("swath_diff", PLANAR, scope=Scope.EXPERIMENT, requires=["low_level_wind", "thpert"])
def swath_diff(ctx: ExperimentContext) -> None:
    """Final wind and cold-pool swaths of each run beside the reference, with differences."""
    ref = _require_reference(ctx)
    rows = ["wind_swath", "coldpool_swath"]
    for run in ctx.runs:
        pair = (ref, run)
        states = {ref.name: _RunState(), run.name: _RunState()}
        frames = {}
        times = ctx.common_times(pair)
        if not times:
            raise ValueError(f"{ref.name} and {run.name} share no output times")
        for _, entries in times:
            for name, frame in _frames_at(ctx, entries, [ref.name, run.name]):
                states[name].update(frame, rows)
                frames[name] = frame
        same_grid(frames[ref.name], frames[run.name])
        frame = frames[ref.name]
        fig, axs = panel_grid(2, 3, panel=(4.8, 4.1))
        for r, (row, attr) in enumerate((("wind_swath", "wind_max"), ("coldpool_swath", "thpert_min"))):
            spec = ROWS[row]
            a, b = getattr(states[ref.name], attr), getattr(states[run.name], attr)
            _panel(
                fig,
                axs[r, 0],
                frame,
                a,
                FIELD_STYLES[spec.style],
                f"{ref.display_name}: {spec.title}",
                ctx.fonts,
                xlabel=r == 1,
            )
            _panel(
                fig,
                axs[r, 1],
                frame,
                b,
                FIELD_STYLES[spec.style],
                f"{run.display_name}: {spec.title}",
                ctx.fonts,
                xlabel=r == 1,
                ylabel=False,
            )
            _panel(
                fig,
                axs[r, 2],
                frame,
                b - a,
                FIELD_STYLES[spec.diff_style],
                f"{run.display_name} minus {ref.display_name}",
                ctx.fonts,
                xlabel=r == 1,
                ylabel=False,
            )
        fig.suptitle(
            f"{ctx.tag}: accumulated swaths, {run.display_name} vs {ref.display_name} "
            f"(through {time_label(times[-1][0])})",
            fontsize=ctx.fonts["suptitle"],
        )
        ctx.save(fig, suffix=run.name)
