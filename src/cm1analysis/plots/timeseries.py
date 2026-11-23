"""Timeseries of domain-reduced and storm-following quantities.

Ported from the legacy ``statistics/derecho_timeseries.py`` (the four
domain extremes), plus three metrics from the momentum-transport papers:

- ``system_speed``: the system speed c_x, placed and fit exactly as for the
  cross sections (:mod:`cm1analysis.sections`, settings from the
  ``cross_section`` config block).
- ``severe_coverage``: grid cells with low-level wind above ``severe_mps``
  (50 kt) as a percentage of the storm area (composite reflectivity above
  ``storm_dbz``), Mahoney and Lackmann (2011, section 4a).
- ``cold_pool_wind``: the mean, over the leading ``lead_extent_km`` behind the
  gust front and across the section band, of each column's maximum zonal wind
  below ``lead_depth_km``; Mahoney et al. (2009, Fig. 7) compare it with the
  system speed. Needs a full 3-D read of u per time.

All metrics are computed in one pass per run, reading each field once, and
shared by every timeseries plot. Each figure is written with a CSV of its
values. Times come from the files' own ``time`` variable. ``extent_km``
restricts the domain reductions (not the storm-following metrics).
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ..io import CM1File
from ..pipeline import ExperimentContext, Frame, RunContext, RunInput, window_for
from ..registry import Category, Scope, category_options, common_options, register, satisfied
from ..sections import PLACEMENT_KEYS, Placement, Section, placements

log = logging.getLogger(__name__)
TIMESERIES = Category.TIMESERIES

DEFAULT_METRICS = [
    "max_sws",
    "max_wind",
    "min_thpert",
    "max_sus",
    "system_speed",
    "severe_coverage",
    "cold_pool_wind",
]
common_options(
    TIMESERIES,
    {
        "extent_km": None,
        "metrics": DEFAULT_METRICS,
        "severe_mps": 25.7,  # 50 kt, the NWS severe criterion used by Mahoney and Lackmann (2011)
        "storm_dbz": 30.0,  # storm area for severe_coverage
        "lead_extent_km": 40.0,  # cold_pool_wind: leading part of the cold pool
        "lead_depth_km": 3.0,
    },
)


@dataclass(frozen=True)
class Metric:
    title: str
    ylabel: str
    requires: tuple  # each a raw field, a derived quantity, or a tuple of alternatives
    compute: Callable[[Frame, Placement | None, Mapping[str, Any]], float]
    placed: bool = False  # needs the storm-following section placement


def _nanmax(a: np.ndarray) -> float:
    return float(np.nanmax(a)) if np.isfinite(a).any() else float("nan")


def _nanmin(a: np.ndarray) -> float:
    return float(np.nanmin(a)) if np.isfinite(a).any() else float("nan")


def _severe_coverage(f: Frame, _place, opt) -> float:
    storm = np.count_nonzero(f.read("cref") > opt["storm_dbz"])
    if storm == 0:
        return float("nan")
    return 100.0 * np.count_nonzero(f.wind().speed > opt["severe_mps"]) / storm


def _system_speed(_f, place: Placement, _opt) -> float:
    return float("nan") if place.storm_u is None else place.storm_u


def _cold_pool_wind(f: Frame, place: Placement, opt) -> float:
    gx = place.gust_x_km
    if gx is None:
        return float("nan")
    sec = Section(f.cm1, place, opt["lead_depth_km"], opt["section"]["average_km"])
    cols = (sec.x >= gx - opt["lead_extent_km"]) & (sec.x <= gx)
    rows = sec.z <= opt["lead_depth_km"]
    if not cols.any() or not rows.any():
        return float("nan")
    u = sec.u_band()[rows][:, :, cols]  # (nz, nrows, ncols)
    return float(np.nanmean(np.nanmax(u, axis=0)))


METRICS: dict[str, Metric] = {
    "max_sws": Metric(
        "Maximum lowest-level wind speed (sws)",
        "Wind speed (m s$^{-1}$)",
        ("sws",),
        lambda f, p, o: _nanmax(f.read("sws")),
    ),
    "max_wind": Metric(
        "Maximum low-level wind speed",
        "Wind speed (m s$^{-1}$)",
        ("low_level_wind",),
        lambda f, p, o: _nanmax(f.wind().speed),
    ),
    "min_thpert": Metric(
        r"Minimum lowest-level $\theta'$",
        r"$\theta'$ (K)",
        ("thpert",),
        lambda f, p, o: _nanmin(f.read("thpert", level=0)),
    ),
    "max_sus": Metric(
        "Maximum vertical velocity at 5 km AGL (sus)",
        "Vertical velocity (m s$^{-1}$)",
        ("sus",),
        lambda f, p, o: _nanmax(f.read("sus")),
    ),
    "system_speed": Metric(
        "System speed $c_x$", "Speed (m s$^{-1}$)", ("thpert", "cref"), _system_speed, placed=True
    ),
    "severe_coverage": Metric(
        "Severe low-level wind coverage",
        "Share of storm area (%)",
        ("low_level_wind", "cref"),
        _severe_coverage,
    ),
    "cold_pool_wind": Metric(
        "Leading cold-pool wind (mean column-max u)",
        "Speed (m s$^{-1}$)",
        ("thpert", "cref", ("uinterp", "u")),
        _cold_pool_wind,
        placed=True,
    ),
}


# ------------------------------------------------------------- computation
@dataclass(frozen=True)
class Series:
    """Metric values for one run: times in seconds and one array per metric."""

    times: np.ndarray
    values: dict[str, np.ndarray]


def metric_options(ctx) -> dict[str, Any]:
    """The plot's options plus the cross-section placement settings (shared with the section plots)."""
    opt = dict(ctx.options)
    opt["section"] = category_options(
        Category.CROSS_SECTION, ctx.settings.plot_config.get("cross_section", {})
    )
    return opt


def series(run: RunInput, metrics: Sequence[str], opt: Mapping[str, Any]) -> Series:
    unknown = [m for m in metrics if m not in METRICS]
    if unknown:
        raise ValueError(f"unknown metric(s) {unknown}; choose from {sorted(METRICS)}")
    extent = None if opt.get("extent_km") is None else tuple(float(v) for v in opt["extent_km"])
    keys = ("severe_mps", "storm_dbz", "lead_extent_km", "lead_depth_km")
    scalars = tuple((k, opt.get(k)) for k in keys)
    section = tuple((k, _freeze(opt.get("section", {}).get(k))) for k in (*PLACEMENT_KEYS, "average_km"))
    return _series(tuple(run.files), tuple(metrics), extent, scalars, section)


def _freeze(value):
    return tuple(value) if isinstance(value, list) else value


@lru_cache(maxsize=32)
def _series(files: tuple[Path, ...], metrics: tuple[str, ...], extent, scalars, section) -> Series:
    opt = dict(scalars)
    opt["section"] = dict(section)
    with CM1File(files[0]) as first:
        usable = [m for m in metrics if _available(first, METRICS[m])]
    for m in sorted(set(metrics) - set(usable)):
        log.warning("metric %s skipped: %s lacks %s", m, files[0].name, METRICS[m].requires)
    places = placements(files, opt["section"]) if any(METRICS[m].placed for m in usable) else {}
    times, values = [], {m: [] for m in usable}
    for path in files:
        with CM1File(path) as cm1:
            window = window_for(cm1, extent)
            for index, seconds in enumerate(cm1.times):
                frame = Frame(cm1, index, float(seconds), window)
                place = places.get((path, index))
                times.append(float(seconds))
                for m in usable:
                    values[m].append(METRICS[m].compute(frame, place, opt))
    order = np.argsort(times, kind="stable")
    return Series(np.asarray(times)[order], {m: np.asarray(v, dtype=float)[order] for m, v in values.items()})


def _available(cm1: CM1File, metric: Metric) -> bool:
    return all(any(satisfied(cm1, n) for n in ((r,) if isinstance(r, str) else r)) for r in metric.requires)


def write_csv(path: Path, columns: dict[str, np.ndarray], times: np.ndarray) -> Path:
    """time_s plus one column per entry, rows in time order (blank where a value is missing)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", *columns])
        for n, t in enumerate(times):
            row = [f"{t:g}"]
            for col in columns.values():
                v = col[n]
                row.append("" if not np.isfinite(v) else f"{v:.6g}")
            writer.writerow(row)
    return path


def _hours(times: np.ndarray) -> np.ndarray:
    return np.asarray(times) / 3600.0


def _style_axes(ax, ctx, ylabel: str, xlabel: bool = True) -> None:
    if xlabel:
        ax.set_xlabel("Time (h)", fontsize=ctx.fonts["axis_label"])
    ax.set_ylabel(ylabel, fontsize=ctx.fonts["axis_label"])
    ax.tick_params(labelsize=ctx.fonts["tick"])
    ax.grid(True, alpha=0.3)


ANY_METRIC = [("sws", "low_level_wind", "thpert", "sus")]


# ------------------------------------------------------------------- plots
@register("timeseries", TIMESERIES, scope=Scope.RUN, requires=ANY_METRIC)
def timeseries(ctx: RunContext) -> None:
    """One panel per metric for a single run, with the values written to CSV."""
    data = series(ctx.run, ctx.options["metrics"], metric_options(ctx))
    names = list(data.values)
    if not names:
        raise ValueError("none of the requested metrics can be computed from this run")
    fig, axs = plt.subplots(
        len(names), 1, figsize=(10, 3.0 * len(names)), sharex=True, layout="constrained", squeeze=False
    )
    for n, (ax, name) in enumerate(zip(axs[:, 0], names, strict=True)):
        metric = METRICS[name]
        ax.plot(_hours(data.times), data.values[name], color=ctx.run.color, lw=1.6)
        ax.set_title(metric.title, fontsize=ctx.fonts["title"])
        _style_axes(ax, ctx, metric.ylabel, xlabel=n == len(names) - 1)
    fig.suptitle(ctx.run.display_name, fontsize=ctx.fonts["suptitle"])
    png = ctx.save(fig)
    write_csv(png.with_suffix(".csv"), data.values, data.times)


@register(
    "speed_vs_cold_pool_wind", TIMESERIES, scope=Scope.RUN, requires=["thpert", "cref", ("uinterp", "u")]
)
def speed_vs_cold_pool_wind(ctx: RunContext) -> None:
    """System speed c_x beside the leading cold-pool wind, as in Mahoney et al. (2009, Fig. 7).

    If the system moves at about the speed of the winds in the forward part of
    its cold pool, the two lines track each other.
    """
    data = series(ctx.run, ["system_speed", "cold_pool_wind"], metric_options(ctx))
    if set(data.values) != {"system_speed", "cold_pool_wind"}:
        raise ValueError("needs both system_speed and cold_pool_wind")
    opt = ctx.options
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    hours = _hours(data.times)
    ax.plot(hours, data.values["system_speed"], color="k", lw=2.2, label="System speed $c_x$")
    ax.plot(
        hours,
        data.values["cold_pool_wind"],
        color="C3",
        lw=1.8,
        ls="--",
        label=f"Mean column-max u, lowest {opt['lead_depth_km']:g} km of the leading "
        f"{opt['lead_extent_km']:g} km of the cold pool",
    )
    both = np.isfinite(data.values["system_speed"]) & np.isfinite(data.values["cold_pool_wind"])
    if both.sum() >= 3:
        a, b = data.values["system_speed"][both], data.values["cold_pool_wind"][both]
        rmse = float(np.sqrt(np.mean((a - b) ** 2)))
        stats = f"RMS difference = {rmse:.1f} m s$^{{-1}}$"
        if np.std(a) > 0 and np.std(b) > 0:
            stats = f"r = {np.corrcoef(a, b)[0, 1]:.2f}, " + stats
        ax.text(
            0.01,
            0.98,
            stats,
            transform=ax.transAxes,
            va="top",
            fontsize=ctx.fonts["legend"],
        )
    ax.set_title(
        f"{ctx.run.display_name}: system speed and leading cold-pool wind", fontsize=ctx.fonts["title"]
    )
    _style_axes(ax, ctx, "Speed (m s$^{-1}$)")
    ax.legend(fontsize=ctx.fonts["legend"], loc="lower right")
    png = ctx.save(fig)
    write_csv(png.with_suffix(".csv"), data.values, data.times)


@register(
    "compare_timeseries", TIMESERIES, scope=Scope.EXPERIMENT, requires=ANY_METRIC, options={"markers": False}
)
def compare_timeseries(ctx: ExperimentContext) -> None:
    """One figure per metric with a line per run (reference first), colored as in the config."""
    runs = ctx.all_runs
    opt = metric_options(ctx)
    data = {run.name: series(run, ctx.options["metrics"], opt) for run in runs}
    all_times = np.unique(np.concatenate([d.times for d in data.values()]))
    for name in ctx.options["metrics"]:
        have = [run for run in runs if name in data[run.name].values]
        if not have:
            log.warning("metric %s: no run can provide it", name)
            continue
        metric = METRICS[name]
        fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
        columns = {}
        for run in have:
            d = data[run.name]
            ax.plot(
                _hours(d.times),
                d.values[name],
                color=run.color,
                lw=1.6,
                label=run.display_name,
                marker=run.marker if ctx.options["markers"] else None,
                markersize=4,
            )
            col = np.full(all_times.size, np.nan)
            col[np.searchsorted(all_times, d.times)] = d.values[name]
            columns[run.name] = col
        ax.set_title(f"{ctx.tag}: {metric.title}", fontsize=ctx.fonts["title"])
        _style_axes(ax, ctx, metric.ylabel)
        ax.legend(fontsize=ctx.fonts["legend"])
        png = ctx.save(fig, suffix=name)
        write_csv(png.with_suffix(".csv"), columns, all_times)
