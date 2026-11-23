"""West-east vertical cross sections through the storm.

Ported from the legacy ``cross_section/`` scripts (derecho_cross_section_v3/v4
and the single-field scripts they duplicate, derecho_buoyancy_cross_section,
derecho_cmt, derecho_hydrometeors, derecho_hydrometeor_flux, derecho_mp_tend,
compare_coldpool_cross_sections).

Every figure has three panels: composite reflectivity in plan view with the
section line, a second panel chosen per plot, and the section itself. Where
the section sits, and the storm motion used for storm-relative winds, are
set by the common options and explained in :mod:`cm1analysis.sections`.
Heights are physical (km above the surface), not level indices.
"""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .. import diagnostics
from ..io import CM1File
from ..pipeline import ExperimentContext, RunContext
from ..registry import Category, Scope, common_options, register
from ..sections import Placement, Section, placements, plan_window
from ._common import (
    FIELD_STYLES,
    SECTION_STYLES,
    SPECIES_COLORS,
    SPECIES_NAMES,
    FieldStyle,
    colorbar,
    km_axes,
    opaque,
    planview,
    time_label,
)

SECTION = Category.CROSS_SECTION

common_options(
    SECTION,
    {
        "y_km": None,  # pin the section row; None places it automatically (auto_row)
        "auto_row": "apex",  # "apex" (forward-most cold pool point) or "max_wind"
        "window_km": 100.0,  # section length
        "trailing_fraction": 0.75,  # share of the window behind the gust front
        "top_km": 12.0,
        "cp_threshold_k": -2.0,  # lowest-level theta' that defines the cold pool
        "storm_u": None,  # c_x in m/s; None estimates it from the leading edge's motion
        "leading_edge": "cold_pool",  # edge tracked for c_x: "cold_pool" or "reflectivity" (>= edge_dbz)
        "edge_dbz": 40.0,
        "motion_window_s": 3600.0,  # c_x is a line fit to the edge position over this window
        "average_km": 40.0,  # sections are averaged over this width across the line (0: one row)
        "smooth_km": 20.0,  # smoothing for auto_row: max_wind
        "plan_zoom": 0.9,  # plan-view half-width, as a fraction of the section length
        "refl_dbz": 35.0,  # reflectivity outline on the section
        "vector_dx_km": 2.5,
        "vector_dz_km": 0.5,
        "vector_scale": 500.0,  # larger draws shorter arrows
    },
)

BASE = ["cref", "thpert", ("uinterp", "u"), "w_scalar"]
COOLING_TERMS = {
    "cbudevp": ("Rain evaporation", "green", "-"),
    "cbudmltg": ("Graupel melting", "purple", "-"),
    "cbudsubg": ("Graupel sublimation", "purple", "--"),
    "cbudmlth": ("Hail melting", "red", "-"),
    "cbudsubh": ("Hail sublimation", "red", "--"),
}


# ----------------------------------------------------------------- helpers
def _sections(ctx: RunContext, options=None) -> Iterator[Section]:
    """Sections at the selected times of the run, each placed per sections.placements."""
    options = options or ctx.options
    places = placements(ctx.run.files, options)
    for frame in ctx.frames():
        if ctx.selected(frame.seconds):
            yield _section(frame.cm1, places[(frame.cm1.path, frame.time)], options)


def _section(cm1: CM1File, place: Placement, options) -> Section:
    return Section(cm1, place, options["top_km"], options["average_km"])


def _draw_line(ax, sec: Section, lw: float) -> None:
    """The section on a plan view: its center line and, when averaged, the band it covers."""
    ax.plot([sec.x[0], sec.x[-1]], [sec.y_km, sec.y_km], color="k", lw=lw)
    if sec.averaged:
        half = 0.5 * sec.band_km
        ax.add_patch(
            plt.Rectangle(
                (sec.x[0], sec.y_km - half),
                sec.x[-1] - sec.x[0],
                sec.band_km,
                fill=False,
                ec="k",
                lw=0.8,
                ls="--",
            )
        )


def _figure():
    fig, axd = plt.subplot_mosaic(
        [["plan", "sec", "sec"], ["aux", "sec", "sec"]], figsize=(19, 10), layout="constrained"
    )
    return fig, axd


def _plan_reflectivity(ax, fig, ctx, sec: Section) -> tuple[slice, slice]:
    """Composite reflectivity around the section, with the section drawn on it. Returns the window."""
    window = plan_window(sec.cm1, sec.place, ctx.options["plan_zoom"])
    ys, xs = window
    x, y = sec.cm1.x_km[xs], sec.cm1.y_km[ys]
    cref = sec.cm1.read("cref", sec.place.time, y=ys, x=xs)
    colorbar(fig, planview(ax, x, y, cref, FIELD_STYLES["cref"]), ax, FIELD_STYLES["cref"], ctx.fonts)
    _draw_line(ax, sec, 2.0)
    km_axes(ax, ctx.fonts)
    ax.set_title("Composite reflectivity", fontsize=ctx.fonts["title"])
    return window


def _plan_field(ax, fig, ctx, sec: Section, window, data, style: FieldStyle, title: str) -> None:
    ys, xs = window
    x, y = sec.cm1.x_km[xs], sec.cm1.y_km[ys]
    colorbar(fig, planview(ax, x, y, data, style), ax, style, ctx.fonts)
    _draw_line(ax, sec, 1.5)
    km_axes(ax, ctx.fonts)
    ax.set_title(title, fontsize=ctx.fonts["title"])


def _plan_updraft(ax, fig, ctx, sec: Section, window) -> None:
    """Second panel for the basic sections: 5-km updraft (sus), else cold pool C (cpc), else wind."""
    ys, xs = window
    t = sec.place.time
    if sec.cm1.has("sus"):
        data, style, title = (
            sec.cm1.read("sus", t, y=ys, x=xs),
            FIELD_STYLES["sus"],
            "Max w at 5 km AGL (sus)",
        )
    elif sec.cm1.has("cpc"):
        data, style, title = sec.cm1.read("cpc", t, y=ys, x=xs), FIELD_STYLES["cpc"], "Cold pool intensity"
    else:
        wind = diagnostics.low_level_wind(sec.cm1, t, window)
        data, style, title = wind.speed, FIELD_STYLES["wind"], f"{wind.label} wind"
    _plan_field(ax, fig, ctx, sec, window, data, style, title)


def _fill(ax, fig, ctx, sec: Section, data, style: FieldStyle):
    kwargs = {"norm": style.norm} if style.norm is not None else {}
    mappable = ax.contourf(
        sec.x,
        sec.z,
        data,
        levels=style.levels,
        cmap=style.cmap,
        extend=style.extend,
        alpha=style.alpha,
        **kwargs,
    )
    colorbar(fig, mappable, ax, style, ctx.fonts)
    return mappable


def _outline_reflectivity(ax, ctx, sec: Section, handles: list) -> None:
    if not sec.cm1.has("dbz"):
        return
    dbz = sec.field("dbz")
    level = ctx.options["refl_dbz"]
    if np.nanmax(dbz) >= level:
        ax.contour(sec.x, sec.z, dbz, levels=[level], colors="k", linewidths=1.2, linestyles="--", alpha=0.8)
        handles.append(Line2D([], [], color="k", ls="--", lw=1.2, label=f"{level:g} dBZ"))


def _outline_cold_pool(ax, ctx, sec: Section, handles: list) -> None:
    level = ctx.options["cp_threshold_k"]
    th = sec.field("thpert")
    if np.nanmin(th) <= level:
        ax.contour(sec.x, sec.z, th, levels=[level], colors="k", linewidths=2.0, linestyles="solid")
        handles.append(Line2D([], [], color="k", lw=2.0, label=f"Cold pool ($\\theta'$ = {level:g} K)"))


def _vectors(ax, ctx, sec: Section, color: str = "k", winds=None) -> None:
    """(u, w) arrows, thinned to about vector_dx_km by vector_dz_km: storm-relative unless winds is given.

    ``winds`` is (u, w, frame label) to draw other components, e.g. the perturbation wind.
    """
    if winds is None:
        u_rel, frame = sec.u_relative()
        w = sec.w()
    else:
        u_rel, w, frame = winds
    dx = max(1, round(ctx.options["vector_dx_km"] / max(np.median(np.diff(sec.x)), 1e-6)))
    cols = np.arange(0, sec.x.size, dx)
    heights = np.arange(ctx.options["vector_dz_km"], sec.z[-1], ctx.options["vector_dz_km"])
    rows = np.unique([int(np.argmin(np.abs(sec.z - h))) for h in heights])
    q = ax.quiver(
        sec.x[cols],
        sec.z[rows],
        u_rel[np.ix_(rows, cols)],
        w[np.ix_(rows, cols)],
        color=color,
        alpha=0.75,
        scale=ctx.options["vector_scale"],
        width=0.0018,
    )
    ax.quiverkey(
        q,
        0.99,
        1.012,
        20.0,
        f"20 m s$^{{-1}}$, {frame}",
        labelpos="W",
        coordinates="axes",
        fontproperties={"size": ctx.fonts["legend"] - 2},
    )


def _finish_section(ax, ctx, sec: Section, title: str, handles: list) -> None:
    ax.set_xlim(sec.x[0], sec.x[-1])
    ax.set_ylim(0.0, sec.z[-1])
    ax.set_xlabel("x (km)", fontsize=ctx.fonts["axis_label"])
    ax.set_ylabel("Height (km)", fontsize=ctx.fonts["axis_label"])
    ax.tick_params(labelsize=ctx.fonts["tick"])
    where = f"y = {sec.y_km:.1f} km" + (f" (mean over {sec.band_km:.0f} km)" if sec.averaged else "")
    ax.set_title(f"{title}, {where}", fontsize=ctx.fonts["title"], pad=22)
    if handles:
        ax.legend(handles=handles, loc="upper left", fontsize=ctx.fonts["legend"], framealpha=0.8)


def _suptitle(fig, ctx, sec: Section) -> None:
    fig.suptitle(f"{ctx.run.display_name}, {time_label(sec.seconds)}", fontsize=ctx.fonts["suptitle"])


def _basic(ctx: RunContext, field, style: FieldStyle, title: str, extra=None, vector_color="k") -> None:
    """The standard three-panel figure: reflectivity and updraft plan views, one section field."""
    for sec in _sections(ctx):
        fig, axd = _figure()
        window = _plan_reflectivity(axd["plan"], fig, ctx, sec)
        _plan_updraft(axd["aux"], fig, ctx, sec, window)
        ax = axd["sec"]
        _fill(ax, fig, ctx, sec, field(sec), style)
        handles: list = []
        if extra:
            extra(ax, sec, handles)
        _outline_reflectivity(ax, ctx, sec, handles)
        _vectors(ax, ctx, sec, vector_color)
        _finish_section(ax, ctx, sec, title, handles)
        _suptitle(fig, ctx, sec)
        ctx.save(fig, sec.seconds)


# ---------------------------------------------------------- basic sections
@register("w_section", SECTION, scope=Scope.RUN, requires=BASE)
def w_section(ctx: RunContext) -> None:
    """Vertical velocity section with storm-relative wind vectors."""
    _basic(ctx, lambda s: s.w(), SECTION_STYLES["w"], "Vertical velocity")


@register("thpert_section", SECTION, scope=Scope.RUN, requires=BASE)
def thpert_section(ctx: RunContext) -> None:
    """Potential temperature perturbation section with the cold pool outlined."""
    _basic(
        ctx,
        lambda s: s.field("thpert"),
        opaque(FIELD_STYLES["thpert"]),
        "Potential temperature perturbation",
        extra=lambda ax, sec, h: _outline_cold_pool(ax, ctx, sec, h),
    )


@register("prspert_section", SECTION, scope=Scope.RUN, requires=[*BASE, "prspert"])
def prspert_section(ctx: RunContext) -> None:
    """Pressure perturbation section (hPa)."""
    _basic(
        ctx, lambda s: s.field("prspert") / 100.0, opaque(FIELD_STYLES["prspert"]), "Pressure perturbation"
    )


@register("sr_wind_section", SECTION, scope=Scope.RUN, requires=[*BASE, "prspert"])
def sr_wind_section(ctx: RunContext) -> None:
    """Storm-relative wind speed in the section plane, with negative pressure perturbations contoured."""

    def speed(sec: Section) -> np.ndarray:
        u_rel, _ = sec.u_relative()
        return np.hypot(u_rel, sec.w())

    def lows(ax, sec: Section, handles: list) -> None:
        p = sec.field("prspert") / 100.0
        levels = np.arange(-15.0, 0.0, 1.0)
        if np.nanmin(p) <= levels[-1]:
            ax.contour(sec.x, sec.z, p, levels=levels, colors="lime", linewidths=1.2, linestyles="solid")
            handles.append(Line2D([], [], color="lime", lw=1.2, label="p' < 0, every 1 hPa"))

    _basic(
        ctx,
        speed,
        SECTION_STYLES["sr_speed"],
        "Storm-relative wind speed (u, w)",
        extra=lows,
        vector_color="w",
    )


@register("yvort_section", SECTION, scope=Scope.RUN, requires=BASE)
def yvort_section(ctx: RunContext) -> None:
    """Horizontal vorticity about y, eta = du/dz - dw/dx (positive: clockwise in this view)."""
    _basic(ctx, lambda s: s.yvort(), SECTION_STYLES["yvort"], r"Horizontal vorticity $\eta = u_z - w_x$")


# --------------------------------------------------------------- buoyancy
@register(
    "buoyancy_section", SECTION, scope=Scope.RUN, requires=[*BASE, "buoyancy"], options={"c_filter_dbz": 10.0}
)
def buoyancy_section(ctx: RunContext) -> None:
    """Buoyancy section (CM1 base state, all hydrometeors) beside cold pool C (Squitieri and Gallus 2022)."""
    style = opaque(FIELD_STYLES["buoyancy"])
    for sec in _sections(ctx):
        fig, axd = _figure()
        window = _plan_reflectivity(axd["plan"], fig, ctx, sec)
        c = diagnostics.cold_pool_c(sec.cm1, sec.place.time, window, ctx.options["c_filter_dbz"])
        _plan_field(
            axd["aux"],
            fig,
            ctx,
            sec,
            window,
            c,
            SECTION_STYLES["cold_pool_c"],
            "Cold pool C (Squitieri and Gallus 2022)",
        )
        ax = axd["sec"]
        b = sec.buoyancy()
        _fill(ax, fig, ctx, sec, b, style)
        handles: list = []
        if np.nanmin(b) < 0.0 < np.nanmax(b):
            ax.contour(sec.x, sec.z, b, levels=[0.0], colors="k", linewidths=2.0, linestyles="solid")
            handles.append(Line2D([], [], color="k", lw=2.0, label="B = 0"))
        _outline_reflectivity(ax, ctx, sec, handles)
        _vectors(ax, ctx, sec)
        twin = ax.twinx()
        twin.plot(sec.x, sec.surface("th"), color="0.25", lw=1.5)
        twin.set_ylabel(r"Lowest-level $\theta$ (K)", fontsize=ctx.fonts["axis_label"], color="0.25")
        twin.tick_params(labelsize=ctx.fonts["tick"])
        handles.append(Line2D([], [], color="0.25", lw=1.5, label=r"Lowest-level $\theta$ (right axis)"))
        _finish_section(ax, ctx, sec, "Buoyancy", handles)
        _suptitle(fig, ctx, sec)
        ctx.save(fig, sec.seconds)


# ------------------------------------------------------ momentum transport
VOLUME_SHORT = {"Leading": "lead", "Trailing": "trail"}
CMT_TERMS = {  # key: (label, color)
    "va_pert": ("VAu'", "C3"),
    "va_base": (r"VA$\bar{u}$", "C0"),
    "ha": ("HAu", "C2"),
    "pga": ("PGA", "C1"),
}


def cmt_terms(sec: Section) -> tuple[dict[str, np.ndarray], str]:
    """Convective momentum transport fields on a section, following Mahoney et al. (2009).

    With u_SR = u - c_x (storm-relative) and u' = u - u0 (u0 the initial wind):

    - ``flux``: u_SR * w (m2 s-2), their Fig. 9a; negative is westerly
      storm-relative momentum carried downward.
    - ``convergence``: -d(u_SR w)/dz, their Fig. 9c, e.
    - budget terms of their Eq. 3, evaluated east-west only as in their
      storm-following volumes: ``va_pert`` = -w du'/dz, ``va_base`` =
      -w du0/dz, ``ha`` = -(u - c_x) du/dx and ``pga`` = -(1/rho) dp'/dx
      (omitted when prspert or density is unavailable).
    - ``u_pert``: band-mean u', for the perturbation wind vectors.

    Rates are in m s-1 h-1. Every product is formed row by row and then
    averaged over the section's band. Returns the fields and the u0 source.
    """
    u0, source = sec.base_u()
    u = sec.u_band().astype(np.float64)
    w = sec.w_band().astype(np.float64)
    c_x = sec.place.storm_u or 0.0
    z_m, x_m = sec.z * 1000.0, sec.x * 1000.0
    u_sr = u - c_x
    flux = u_sr * w
    u_pert = u - u0[:, None, None]
    out = {
        "flux": flux,
        "convergence": -np.gradient(flux, z_m, axis=0) * 3600.0,
        "va_pert": -w * np.gradient(u_pert, z_m, axis=0) * 3600.0,
        "va_base": -w * np.gradient(u0, z_m)[:, None, None] * 3600.0,
        "ha": -u_sr * np.gradient(u, x_m, axis=2) * 3600.0,
        "u_pert": u_pert,
    }
    if sec.cm1.has("prspert") and diagnostics.recipe(sec.cm1, "air_density"):
        dpdx = np.gradient(sec.band("prspert").astype(np.float64), x_m, axis=2)
        out["pga"] = -dpdx / sec.density_band() * 3600.0
    return {k: sec.mean(v) for k, v in out.items()}, source


def _volume_profiles(
    sec: Section, terms: dict[str, np.ndarray], volumes
) -> dict[tuple[str, str], np.ndarray]:
    """Volume-mean profile of each budget term, keyed (volume, term); volumes relative to the gust front."""
    gx = sec.place.gust_x_km
    out = {}
    if gx is None:
        return out
    for name, near, far, depth, _ in volumes:
        cols = (sec.x >= gx - far) & (sec.x <= gx - near)
        rows = sec.z <= depth
        if cols.sum() < 2 or rows.sum() < 2:
            continue
        for key in CMT_TERMS:
            if key in terms:
                prof = np.full(sec.z.size, np.nan)
                prof[rows] = terms[key][np.ix_(rows, cols)].mean(axis=1)
                out[(name, key)] = prof
    return out


@register(
    "cmt_section",
    SECTION,
    scope=Scope.RUN,
    requires=BASE,
    options={
        "lead_extent_km": 40.0,
        "lead_depth_km": 3.0,
        "trail_extent_km": 120.0,
        "trail_depth_km": 6.0,
        "average_s": 1800.0,
        "convergence_levels": [25.0, 50.0, 100.0],
    },
)
def cmt_section(ctx: RunContext) -> None:
    """Storm-relative momentum flux u_SR*w with its convergence and volume-mean budget terms.

    Follows Mahoney et al. (2009): the shading is their Fig. 9a field, the
    contours its convergence (m s-1 h-1, solid convergence, dashed
    divergence), the arrows the ground-relative perturbation wind (u', w),
    and the profiles the east-west budget terms of their Eq. 3 averaged over
    the leading volume (0 to lead_extent_km behind the gust front, below
    lead_depth_km) and the trailing volume (lead_extent_km to trail_extent_km,
    below trail_depth_km). Profiles are also averaged over average_s centered
    on each time, each output in its own storm-following frame. The section is
    lengthened if needed so the trailing volume fits behind the gust front.
    """
    o = ctx.options
    options = dict(o)
    options["window_km"] = max(
        o["window_km"], 1.05 * o["trail_extent_km"] / max(o["trailing_fraction"], 0.05)
    )
    volumes = (
        ("Leading", 0.0, o["lead_extent_km"], o["lead_depth_km"], "-"),
        ("Trailing", o["lead_extent_km"], o["trail_extent_km"], o["trail_depth_km"], "--"),
    )
    places = placements(ctx.run.files, options)
    half = 0.5 * float(o["average_s"] or 0.0)
    wanted = [f.seconds for f in ctx.frames() if ctx.selected(f.seconds)] if ctx.settings.times else None

    # Pass 1: volume profiles at every time that falls inside some figure's averaging window.
    history: list[tuple[float, dict]] = []
    for frame in ctx.frames():
        if wanted is not None and not any(abs(frame.seconds - t) <= half + 0.5 for t in wanted):
            continue
        sec = _section(frame.cm1, places[(frame.cm1.path, frame.time)], options)
        terms, _ = cmt_terms(sec)
        history.append((frame.seconds, _volume_profiles(sec, terms, volumes)))

    # Pass 2: one figure per selected time.
    for sec in _sections(ctx, options):
        terms, source = cmt_terms(sec)
        fig, axd = _figure()
        _plan_reflectivity(axd["plan"], fig, ctx, sec)
        ax = axd["sec"]
        _fill(ax, fig, ctx, sec, terms["flux"], SECTION_STYLES["momentum_flux"])
        handles: list = []
        levels = sorted(float(v) for v in o["convergence_levels"])
        conv = terms["convergence"]
        for lv, ls in (([-v for v in reversed(levels)], "--"), (levels, "-")):
            if np.nanmin(conv) <= max(lv) and np.nanmax(conv) >= min(lv):
                ax.contour(sec.x, sec.z, conv, levels=lv, colors="0.15", linewidths=0.9, linestyles=ls)
        handles.append(
            Line2D(
                [],
                [],
                color="0.15",
                lw=0.9,
                label=f"$-\\partial(u_{{SR}}w)/\\partial z$ = {', '.join(f'{v:g}' for v in levels)}"
                " m s$^{-1}$ h$^{-1}$ (dashed < 0)",
            )
        )
        _outline_cold_pool(ax, ctx, sec, handles)
        _outline_reflectivity(ax, ctx, sec, handles)
        _vectors(ax, ctx, sec, winds=(terms["u_pert"], sec.w(), "ground-relative perturbation wind (u', w)"))

        prof = axd["aux"]
        window = [p for t, p in history if abs(t - sec.seconds) <= half + 0.5]
        drawn = False
        for name, near, far, depth, ls in volumes:
            if sec.place.gust_x_km is not None:
                gx = sec.place.gust_x_km
                ax.add_patch(
                    plt.Rectangle(
                        (gx - far, 0.0),
                        far - near,
                        depth,
                        fill=False,
                        ec="darkgreen",
                        ls=ls,
                        lw=2.2,
                        zorder=6,
                    )
                )
            for key, (label, color) in CMT_TERMS.items():
                series = [p[(name, key)] for p in window if (name, key) in p]
                if not series:
                    continue
                stack = np.stack(series)
                count = np.isfinite(stack).sum(axis=0)
                mean = np.where(count > 0, np.nansum(stack, axis=0) / np.maximum(count, 1), np.nan)
                ok = np.isfinite(mean)
                drawn = True
                prof.plot(
                    mean[ok],
                    sec.z[ok],
                    color=color,
                    ls=ls,
                    lw=1.6,
                    label=f"{label} {VOLUME_SHORT[name]} ({np.nanmean(mean):+.1f})",
                )
        prof.axvline(0.0, color="0.5", lw=0.8)
        prof.set_xlabel(r"Zonal momentum tendency (m s$^{-1}$ h$^{-1}$)", fontsize=ctx.fonts["axis_label"])
        prof.set_ylabel("Height (km)", fontsize=ctx.fonts["axis_label"])
        prof.tick_params(labelsize=ctx.fonts["tick"])
        n = len(window)
        span = f", mean of {n} outputs over {2 * half / 60:g} min" if n > 1 else ""
        prof.set_title(f"Volume-mean budget terms{span}", fontsize=ctx.fonts["title"] - 2)
        if drawn:
            prof.legend(fontsize=ctx.fonts["legend"] - 4, loc="upper right", ncol=2)
        else:
            prof.text(
                0.5, 0.5, "no surface cold pool yet", transform=prof.transAxes, ha="center", color="0.4"
            )
        handles.append(
            Line2D([], [], color="darkgreen", lw=2.2, label="Budget volumes (leading solid, trailing dashed)")
        )
        frame_note = "" if sec.place.storm_u is not None else "; c_x unknown, u_SR = u"
        _finish_section(
            ax, ctx, sec, f"Storm-relative momentum flux $u_{{SR}}w$ (u0: {source}{frame_note})", handles
        )
        _suptitle(fig, ctx, sec)
        ctx.save(fig, sec.seconds)


# ------------------------------------------------------------ hydrometeors
HYDRO_LEVELS = (0.1, 0.5, 1.0, 2.0, 4.0)  # g kg-1


def _species(ctx, cm1: CM1File) -> list[str]:
    wanted = ctx.options["species"]
    return [q for q in (cm1.hydrometeors if wanted is None else wanted) if cm1.has(q)]


def _dbz_background(ax, fig, ctx, sec: Section) -> None:
    if sec.cm1.has("dbz"):
        _fill(ax, fig, ctx, sec, sec.field("dbz"), SECTION_STYLES["dbz_background"])


@register(
    "hydrometeor_section",
    SECTION,
    scope=Scope.RUN,
    requires=[*BASE, "dbz", ("qr", "qs", "qg", "qhl")],
    options={"species": ["qr", "qs", "qg"]},
)
def hydrometeor_section(ctx: RunContext) -> None:
    """Hydrometeor mixing ratio contours over reflectivity, with column maxima in plan view."""
    for sec in _sections(ctx):
        species = _species(ctx, sec.cm1)
        fig, axd = _figure()
        window = _plan_reflectivity(axd["plan"], fig, ctx, sec)
        aux = axd["aux"]
        ys, xs = window
        x, y = sec.cm1.x_km[xs], sec.cm1.y_km[ys]
        handles: list = []
        for q in species:
            color = SPECIES_COLORS.get(q, "gray")
            colmax = 1000.0 * diagnostics.column_reduce(sec.cm1, q, sec.place.time, np.fmax, window)
            if np.nanmax(colmax) >= HYDRO_LEVELS[0]:
                lines = aux.contour(x, y, colmax, levels=HYDRO_LEVELS, colors=[color], linewidths=0.9)
                aux.clabel(lines, fmt="%g", fontsize=ctx.fonts["contour_label"])
            handles.append(Line2D([], [], color=color, lw=1.2, label=f"{SPECIES_NAMES.get(q, q)} ({q})"))
        cref = sec.cm1.read("cref", sec.place.time, y=ys, x=xs)
        aux.contour(x, y, cref, levels=[ctx.options["refl_dbz"]], colors="k", linewidths=1.0)
        _draw_line(aux, sec, 1.5)
        aux.set_aspect("equal")
        km_axes(aux, ctx.fonts)
        aux.set_title("Column-maximum mixing ratio (g kg$^{-1}$)", fontsize=ctx.fonts["title"])

        ax = axd["sec"]
        _dbz_background(ax, fig, ctx, sec)
        for q in species:
            field = 1000.0 * sec.field(q)
            if np.nanmax(field) >= HYDRO_LEVELS[0]:
                lines = ax.contour(
                    sec.x,
                    sec.z,
                    field,
                    levels=HYDRO_LEVELS,
                    colors=[SPECIES_COLORS.get(q, "gray")],
                    linewidths=1.0,
                )
                ax.clabel(lines, fmt="%g", fontsize=ctx.fonts["contour_label"])
        _outline_reflectivity(ax, ctx, sec, handles)
        _finish_section(ax, ctx, sec, "Hydrometeor mixing ratios (g kg$^{-1}$)", handles)
        _suptitle(fig, ctx, sec)
        ctx.save(fig, sec.seconds)


FLUX_SECTION_LEVELS = (1.0, 5.0, 10.0, 25.0, 50.0)  # g m-2 s-1


@register(
    "hydrometeor_flux_section",
    SECTION,
    scope=Scope.RUN,
    requires=[*BASE, "dbz", "air_density", ("qc", "qr", "qi", "qs", "qg", "qhl")],
    options={"species": None},
)
def hydrometeor_flux_section(ctx: RunContext) -> None:
    """Vertical mass flux rho*q*w of each species (solid up, dotted down), with window-mean profiles.

    Transport by the air motion only; sedimentation is not included.
    """
    for sec in _sections(ctx):
        species = _species(ctx, sec.cm1)
        rho_w = 1000.0 * sec.density_band() * sec.w_band()  # products row by row, then averaged
        flux = {q: sec.mean(rho_w * sec.band(q)) for q in species}
        fig, axd = _figure()
        _plan_reflectivity(axd["plan"], fig, ctx, sec)

        prof = axd["aux"]
        total = np.zeros(sec.z.size)
        for q in species:
            p = np.nanmean(flux[q], axis=1)
            total += p
            prof.plot(p, sec.z, color=SPECIES_COLORS.get(q, "gray"), lw=1.5, label=SPECIES_NAMES.get(q, q))
        prof.plot(total, sec.z, color="k", ls="--", lw=1.5, label="Total")
        prof.axvline(0.0, color="0.5", lw=0.8)
        prof.set_xlabel("Section-mean mass flux (g m$^{-2}$ s$^{-1}$)", fontsize=ctx.fonts["axis_label"])
        prof.set_ylabel("Height (km)", fontsize=ctx.fonts["axis_label"])
        prof.tick_params(labelsize=ctx.fonts["tick"])
        prof.legend(fontsize=ctx.fonts["legend"] - 2)
        prof.set_title("Section-mean vertical mass flux", fontsize=ctx.fonts["title"])

        ax = axd["sec"]
        _dbz_background(ax, fig, ctx, sec)
        handles: list = []
        for q in species:
            color = SPECIES_COLORS.get(q, "gray")
            f = flux[q]
            for levels, ls in (
                (FLUX_SECTION_LEVELS, "-"),
                (tuple(-v for v in reversed(FLUX_SECTION_LEVELS)), ":"),
            ):
                if np.nanmin(f) <= max(levels) and np.nanmax(f) >= min(levels):
                    lines = ax.contour(
                        sec.x, sec.z, f, levels=levels, colors=[color], linewidths=1.0, linestyles=ls
                    )
                    ax.clabel(lines, fmt=lambda v: f"{abs(v):g}", fontsize=ctx.fonts["contour_label"])
            handles.append(Line2D([], [], color=color, lw=1.2, label=SPECIES_NAMES.get(q, q)))
        handles.append(Line2D([], [], color="0.3", lw=1.0, ls=":", label="Downward (dotted)"))
        _outline_reflectivity(ax, ctx, sec, handles)
        _finish_section(ax, ctx, sec, "Vertical mass flux (g m$^{-2}$ s$^{-1}$)", handles)
        _suptitle(fig, ctx, sec)
        ctx.save(fig, sec.seconds)


@register(
    "cooling_section",
    SECTION,
    scope=Scope.RUN,
    requires=[*BASE, tuple(COOLING_TERMS)],
    options={"term_levels": [-0.01, -0.003, -0.001]},
)
def cooling_section(ctx: RunContext) -> None:
    """Each cold-pool cooling term (cbud*) contoured at term_levels over reflectivity; thicker is stronger.

    Colors follow the hydrometeor convention (rain green, graupel purple,
    hail red); melting and evaporation are solid, sublimation dashed. The
    plan view shows the column-minimum of the summed terms.
    """
    for sec in _sections(ctx):
        terms = [t for t in COOLING_TERMS if sec.cm1.has(t)]
        fig, axd = _figure()
        window = _plan_reflectivity(axd["plan"], fig, ctx, sec)
        ys, xs = window
        coldest = None
        for k in range(sec.ktop):
            level_total = sum(sec.cm1.read(t, sec.place.time, level=k, y=ys, x=xs) for t in terms)
            coldest = level_total if coldest is None else np.fmin(coldest, level_total)
        _plan_field(
            axd["aux"],
            fig,
            ctx,
            sec,
            window,
            coldest,
            SECTION_STYLES["cooling_total"],
            f"Column-minimum total cooling (below {sec.z[-1]:.0f} km)",
        )

        ax = axd["sec"]
        _dbz_background(ax, fig, ctx, sec)
        handles: list = []
        levels = sorted(float(v) for v in ctx.options["term_levels"])  # most negative first
        widths = np.linspace(3.6, 1.4, len(levels))  # stronger cooling, thicker line
        for t in terms:
            label, color, ls = COOLING_TERMS[t]
            field = sec.field(t)
            drawn = [(lv, lw) for lv, lw in zip(levels, widths, strict=True) if np.nanmin(field) <= lv]
            if drawn:
                ax.contour(
                    sec.x,
                    sec.z,
                    field,
                    levels=[lv for lv, _ in drawn],
                    colors=[color],
                    linewidths=[lw for _, lw in drawn],
                    linestyles=ls,
                )
            handles.append(Line2D([], [], color=color, ls=ls, lw=2.4, label=label))
        handles += [
            Line2D([], [], color="0.3", lw=lw, label=f"{lv:g} K s$^{{-1}}$")
            for lv, lw in zip(levels, widths, strict=True)
        ]
        _finish_section(ax, ctx, sec, "Microphysical cooling", handles)
        _suptitle(fig, ctx, sec)
        ctx.save(fig, sec.seconds)


# ------------------------------------------------------ experiment scope
@register("compare_thpert_section", SECTION, scope=Scope.EXPERIMENT, requires=BASE)
def compare_thpert_section(ctx: ExperimentContext) -> None:
    """Theta' sections of each run (reference first), one row per run, at every common time.

    Each run's section is placed, and its storm motion estimated, from that run alone.
    """
    runs = ctx.all_runs
    places = {run.name: placements(run.files, ctx.options) for run in runs}
    style = opaque(FIELD_STYLES["thpert"])
    for seconds, entries in ctx.common_times(runs):
        if not ctx.selected(seconds):
            continue
        fig, axs = plt.subplots(
            len(runs),
            2,
            figsize=(19, 5.2 * len(runs)),
            layout="constrained",
            width_ratios=[1, 2],
            squeeze=False,
        )
        for r, run in enumerate(runs):
            path, t = entries[run.name]
            with CM1File(path) as cm1:
                place: Placement = places[run.name][(path, t)]
                sec = _section(cm1, place, ctx.options)
                _plan_reflectivity(axs[r, 0], fig, ctx, sec)
                axs[r, 0].set_title(
                    f"{run.display_name}: composite reflectivity", fontsize=ctx.fonts["title"]
                )
                ax = axs[r, 1]
                _fill(ax, fig, ctx, sec, sec.field("thpert"), style)
                handles: list = []
                _outline_cold_pool(ax, ctx, sec, handles)
                _outline_reflectivity(ax, ctx, sec, handles)
                _vectors(ax, ctx, sec)
                _finish_section(ax, ctx, sec, f"{run.display_name}: $\\theta'$", handles)
        fig.suptitle(f"{ctx.tag}, {time_label(seconds)}", fontsize=ctx.fonts["suptitle"])
        ctx.save(fig, seconds)
