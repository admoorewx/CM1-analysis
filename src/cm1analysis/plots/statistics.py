"""Distributions of the accumulated wind swath and cold pool, and their sensitivity.

Ported from the legacy ``statistics/sensitivity_analysis.py`` (the main
source), ``statistics/swath_coldpool_stats.py`` and ``swath_comparison.py``
(both two-sample versions of the same comparison, folded in here), and
``derecho_correlations.py`` (wind correlations).

For each run, two fields are accumulated over every output time:

- the **wind swath**: the time maximum of the low-level wind speed at each
  grid point (``swath_field: low_level_wind``, the 10-m wind if written, else
  the lowest model level), or of CM1's own lowest-level swath ``sws``, which
  also catches maxima between outputs (``swath_field: sws``);
- the **cold pool**: the time minimum of the lowest-level theta'.

Each is reduced to a sample of grid points inside the feature (wind at least
``wind_mask_mps``; theta' at most ``cold_pool_mask_k``) so the background
environment does not swamp the distribution. The experiment plots compare
each run's sample with the reference run's: kernel density estimates,
histograms, quantile differences, and a table of effect sizes (Cliff's delta,
Cohen's d, the Kolmogorov-Smirnov statistic D, and quantile differences, see
:mod:`cm1analysis.stats`). Grid points are spatially autocorrelated, so these
are descriptive measures; no p-values are reported.

Footprint areas use each grid cell's own area, so stretched grids are handled.
Times come from the files' ``time`` variable; ``extent_km`` restricts every
field to part of the domain.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from .. import stats
from ..io import CM1File
from ..pipeline import ExperimentContext, Frame, RunContext, RunInput, window_for
from ..registry import Category, Scope, common_options, register, satisfied
from ..style import SEVERE_WIND

log = logging.getLogger(__name__)
STATISTICS = Category.STATISTICS

SAMPLES = ("swath", "cold_pool")
SWATH_FIELDS = ("low_level_wind", "sws")
DECILE_PERCENTILES = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]

common_options(
    STATISTICS,
    {
        "extent_km": None,
        "samples": list(SAMPLES),
        "swath_field": "low_level_wind",
        "wind_mask_mps": 10.0,  # swath sample: points at least this windy
        "wind_area_mps": SEVERE_WIND,  # footprint area of damaging wind
        "cold_pool_mask_k": -1.0,  # cold-pool sample: points at least this cold
        "cold_pool_area_k": -2.0,  # footprint area of a significant cold pool
    },
)


# ---------------------------------------------------------- sample kinds
@dataclass(frozen=True)
class Kind:
    """How one accumulated field becomes a sample, and how to label it."""

    key: str
    title: str
    units: str
    upper_tail: bool  # True: the extreme of interest is the high end (wind)
    mask_key: str
    area_key: str

    def inside(self, field: np.ndarray, threshold: float) -> np.ndarray:
        return field >= threshold if self.upper_tail else field <= threshold


KINDS = {
    "swath": Kind("swath", "Wind swath", "m s$^{-1}$", True, "wind_mask_mps", "wind_area_mps"),
    "cold_pool": Kind(
        "cold_pool", r"Cold pool (minimum lowest-level $\theta'$)", "K", False, "cold_pool_mask_k",
        "cold_pool_area_k",
    ),
}  # fmt: skip


# ----------------------------------------------------------- accumulation
@dataclass(frozen=True)
class Accumulated:
    """Time-accumulated 2-D fields of one run (None where the run lacks the inputs)."""

    x_km: np.ndarray
    y_km: np.ndarray
    cell_area_km2: np.ndarray
    swath: np.ndarray | None
    cold_pool: np.ndarray | None
    swath_label: str
    times: np.ndarray


def accumulate(run: RunInput, opt: Mapping[str, Any]) -> Accumulated:
    """The run's wind swath and cold pool over all of its output times (cached per run)."""
    if opt["swath_field"] not in SWATH_FIELDS:
        raise ValueError(f"swath_field must be one of {SWATH_FIELDS}, got {opt['swath_field']!r}")
    extent = None if opt.get("extent_km") is None else tuple(float(v) for v in opt["extent_km"])
    return _accumulate(tuple(run.files), opt["swath_field"], extent)


@lru_cache(maxsize=64)
def _accumulate(files: tuple[Path, ...], swath_field: str, extent) -> Accumulated:
    with CM1File(files[0]) as first:
        has_swath = satisfied(first, swath_field)
        has_cold = first.has("thpert")
    swath = cold = None
    label = "lowest-level max wind (sws)" if swath_field == "sws" else ""
    times, x, y = [], None, None
    for path in files:
        with CM1File(path) as cm1:
            window = window_for(cm1, extent)
            if x is None:
                x, y = cm1.x_km[window[1]], cm1.y_km[window[0]]
            for index, seconds in enumerate(cm1.times):
                frame = Frame(cm1, index, float(seconds), window)
                times.append(float(seconds))
                if has_swath:
                    if swath_field == "sws":
                        speed = frame.read("sws")
                    else:
                        wind = frame.wind()
                        speed, label = wind.speed, f"{wind.label} wind"
                    swath = speed if swath is None else np.fmax(swath, speed)
                if has_cold:
                    th = frame.read("thpert", level=0)
                    cold = th if cold is None else np.fmin(cold, th)
    area = np.outer(_spacing(y), _spacing(x))
    return Accumulated(x, y, area, swath, cold, label, np.sort(np.asarray(times)))


def _spacing(coord_km: np.ndarray) -> np.ndarray:
    """Width of each cell (km) around its center; uniform or stretched."""
    return np.abs(np.gradient(coord_km)) if coord_km.size > 1 else np.ones(1)


@dataclass(frozen=True)
class Sample:
    """The grid points of one run inside one feature, plus whole-field summary numbers."""

    run: RunInput
    kind: Kind
    values: np.ndarray
    extreme: float  # domain max (swath) or min (cold pool) of the accumulated field
    area_km2: float  # footprint beyond the area threshold

    @property
    def describe(self) -> dict[str, float]:
        return stats.describe(self.values)


def sample(run: RunInput, kind: Kind, opt: Mapping[str, Any]) -> Sample | None:
    """The run's sample of one kind, or None (logged) if it lacks the field or the feature."""
    acc = accumulate(run, opt)
    field = getattr(acc, kind.key)
    if field is None:
        log.warning("%s: no %s (missing input fields)", run.name, kind.key)
        return None
    finite = np.isfinite(field)
    inside = kind.inside(np.where(finite, field, np.nan), opt[kind.mask_key]) & finite
    if not inside.any():
        log.warning("%s: no grid points in the %s (threshold %g)", run.name, kind.key, opt[kind.mask_key])
        return None
    area = float(np.sum(acc.cell_area_km2[kind.inside(field, opt[kind.area_key]) & finite]))
    extreme = float(np.nanmax(field) if kind.upper_tail else np.nanmin(field))
    return Sample(run, kind, field[inside].astype(float), extreme, area)


def _kinds(opt: Mapping[str, Any]) -> list[Kind]:
    unknown = set(opt["samples"]) - set(KINDS)
    if unknown:
        raise ValueError(f"unknown sample(s) {sorted(unknown)}; choose from {list(KINDS)}")
    return [KINDS[k] for k in opt["samples"]]


def _samples(ctx: ExperimentContext, kind: Kind) -> list[Sample]:
    out = [s for s in (sample(run, kind, ctx.options) for run in ctx.all_runs) if s is not None]
    return out


def _require_reference(ctx: ExperimentContext) -> RunInput:
    if ctx.reference is None:
        raise ValueError(
            f"{ctx.spec.name} compares runs with a reference run: set 'reference: group/run' in the "
            "config or pass --reference"
        )
    return ctx.reference


def _xlabel(kind: Kind, acc_label: str) -> str:
    if kind.key == "swath":
        return f"Maximum {acc_label} speed ({kind.units})".replace("  ", " ")
    return r"Minimum lowest-level $\theta'$ (K)"


def _mask_text(kind: Kind, opt) -> str:
    sign = r"\geq" if kind.upper_tail else r"\leq"
    return f"points ${sign}$ {opt[kind.mask_key]:g} {kind.units}"


def _line_style(s: Sample, reference: RunInput | None) -> dict[str, Any]:
    is_ref = reference is not None and s.run.files == reference.files
    return {
        "color": s.run.color,
        "lw": 2.6 if is_ref else 1.6,
        "zorder": 5 if is_ref else 2,
        "label": s.run.display_name + (" (reference)" if is_ref else ""),
    }


def _panels(ctx, kinds: Sequence[Kind]):
    fig, axs = plt.subplots(1, len(kinds), figsize=(9 * len(kinds), 6), layout="constrained", squeeze=False)
    return fig, axs[0]


def _style(ax, ctx, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel, fontsize=ctx.fonts["axis_label"])
    ax.set_ylabel(ylabel, fontsize=ctx.fonts["axis_label"])
    ax.set_title(title, fontsize=ctx.fonts["title"])
    ax.tick_params(labelsize=ctx.fonts["tick"])
    ax.minorticks_on()
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)


def _fmt(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{value:.6g}" if np.isfinite(value) else ""


def write_rows(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> Path:
    """A CSV with one header row; NaN is written blank."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows([_fmt(v) for v in row] for row in rows)
    return path


# ------------------------------------------------------------------ plots
ANY_SAMPLE = [("low_level_wind", "sws", "thpert")]


@register(
    "distributions",
    STATISTICS,
    scope=Scope.EXPERIMENT,
    requires=ANY_SAMPLE,
    options={"kde_points": 400, "kde_sample_cap": 20000, "seed": 0},
)
def distributions(ctx: ExperimentContext) -> None:
    """Gaussian kernel density estimates of each run's swath and cold-pool sample (reference bold).

    Samples larger than ``kde_sample_cap`` points are subsampled (with a fixed
    ``seed`` per run, so every figure is reproducible) to keep the KDE fast.
    """
    opt = ctx.options
    kinds = _kinds(opt)
    fig, axs = _panels(ctx, kinds)
    drawn = False
    for ax, kind in zip(axs, kinds, strict=True):
        samples = _samples(ctx, kind)
        if not samples:
            ax.set_visible(False)
            continue
        lo = min(s.values.min() for s in samples)
        hi = max(s.values.max() for s in samples)
        grid = np.linspace(lo, hi, int(opt["kde_points"]))
        for s in samples:
            values = s.values
            if values.size > opt["kde_sample_cap"]:
                rng = np.random.default_rng(opt["seed"])
                values = rng.choice(values, size=int(opt["kde_sample_cap"]), replace=False)
            if values.size < 2 or np.ptp(values) == 0:
                log.warning("%s: %s sample too small or constant for a KDE", s.run.name, kind.key)
                continue
            ax.plot(grid, gaussian_kde(values)(grid), **_line_style(s, ctx.reference))
            drawn = True
        acc_label = accumulate(samples[0].run, opt).swath_label
        _style(ax, ctx, _xlabel(kind, acc_label), "Density", f"{kind.title}, {_mask_text(kind, opt)}")
        ax.legend(fontsize=ctx.fonts["legend"], ncol=2)
    if not drawn:
        plt.close(fig)
        raise ValueError("no run has a sample to plot")
    fig.suptitle(f"{ctx.tag}: distributions", fontsize=ctx.fonts["suptitle"])
    ctx.save(fig)


@register("histograms", STATISTICS, scope=Scope.EXPERIMENT, requires=ANY_SAMPLE, options={"bins": 60})
def histograms(ctx: ExperimentContext) -> None:
    """Density-normalized histograms on shared bins; the reference filled, the other runs as outlines."""
    opt = ctx.options
    kinds = _kinds(opt)
    fig, axs = _panels(ctx, kinds)
    rows, drawn = [], False
    for ax, kind in zip(axs, kinds, strict=True):
        samples = _samples(ctx, kind)
        if not samples:
            ax.set_visible(False)
            continue
        lo = min(s.values.min() for s in samples)
        hi = max(s.values.max() for s in samples)
        if hi <= lo:
            hi = lo + 1.0
        bins = np.linspace(lo, hi, int(opt["bins"]) + 1)
        for s in samples:
            style = _line_style(s, ctx.reference)
            is_ref = style["zorder"] == 5
            density, _, _ = ax.hist(
                s.values,
                bins=bins,
                density=True,
                histtype="stepfilled" if is_ref else "step",
                facecolor=(0.75, 0.75, 0.75, 0.5) if is_ref else "none",
                edgecolor=style["color"],
                lw=style["lw"],
                zorder=style["zorder"],
                label=style["label"],
            )
            rows += [
                [kind.key, s.run.name, a, b, d] for a, b, d in zip(bins[:-1], bins[1:], density, strict=True)
            ]
            drawn = True
        acc_label = accumulate(samples[0].run, opt).swath_label
        _style(ax, ctx, _xlabel(kind, acc_label), "Density", f"{kind.title}, {_mask_text(kind, opt)}")
        ax.legend(fontsize=ctx.fonts["legend"], ncol=2)
    if not drawn:
        plt.close(fig)
        raise ValueError("no run has a sample to plot")
    fig.suptitle(f"{ctx.tag}: histograms", fontsize=ctx.fonts["suptitle"])
    png = ctx.save(fig)
    write_rows(png.with_suffix(".csv"), ["sample", "run", "bin_low", "bin_high", "density"], rows)


@register(
    "quantile_differences",
    STATISTICS,
    scope=Scope.EXPERIMENT,
    requires=ANY_SAMPLE,
    options={"percentiles": DECILE_PERCENTILES},
)
def quantile_differences(ctx: ExperimentContext) -> None:
    """Q_p(run) - Q_p(reference) against the reference's own Q_p, one line per run.

    A line above zero at the right (left) end means the run's strongest winds
    (warmest cold-pool points) exceed the reference's; for the cold pool the
    left end is the coldest air.
    """
    reference = _require_reference(ctx)
    opt = ctx.options
    percentiles = [float(p) for p in opt["percentiles"]]
    kinds = _kinds(opt)
    fig, axs = _panels(ctx, kinds)
    rows, drawn = [], False
    for ax, kind in zip(axs, kinds, strict=True):
        ref = sample(reference, kind, opt)
        if ref is None:
            ax.set_visible(False)
            continue
        x = np.percentile(ref.values, percentiles)
        for s in _samples(ctx, kind):
            if s.run.files == reference.files:
                continue
            q = stats.quantile_effects(s.values, ref.values, percentiles)
            y = [q[p].diff for p in percentiles]
            ax.plot(x, y, color=s.run.color, marker=s.run.marker, ms=7, lw=1.7, label=s.run.display_name)
            rows += [
                [kind.key, s.run.name, p, q[p].q_ref, q[p].q_run, q[p].diff, q[p].sqd] for p in percentiles
            ]
            drawn = True
        ax.axhline(0.0, color="k", lw=0.8, ls="--", alpha=0.6)
        acc_label = accumulate(reference, opt).swath_label
        _style(
            ax,
            ctx,
            f"Reference quantile: {_xlabel(kind, acc_label)}",
            f"$Q_{{run}} - Q_{{ref}}$ ({kind.units})",
            f"{kind.title}, {_mask_text(kind, opt)}",
        )
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=ctx.fonts["legend"], ncol=2)
    if not drawn:
        plt.close(fig)
        raise ValueError("no run besides the reference has a sample to compare")
    fig.suptitle(
        f"{ctx.tag}: quantile differences from {reference.display_name}", fontsize=ctx.fonts["suptitle"]
    )
    png = ctx.save(fig)
    header = ["sample", "run", "percentile", "q_ref", "q_run", "diff", "sqd"]
    write_rows(png.with_suffix(".csv"), header, rows)


EFFECT_HEADER = [
    "sample", "run", "n", "mean", "median", "std", "iqr", "robust_scale", "p05", "p10", "p90", "p95",
    "extreme", "area_km2", "cliffs_delta", "cliffs_label", "cohens_d", "cohens_label", "ks_d",
]  # fmt: skip


@register(
    "effect_sizes",
    STATISTICS,
    scope=Scope.EXPERIMENT,
    requires=ANY_SAMPLE,
    options={"swath_percentiles": [90, 95, 100], "cold_pool_percentiles": [10, 5, 0]},
)
def effect_sizes(ctx: ExperimentContext) -> None:
    """Effect size of every run against the reference, and the group's sensitivity summary.

    The figure places each run by the K-S statistic D (how different the
    distributions are, in [0, 1]) and Cliff's delta (which way, and how much,
    in [-1, 1]), with the group mean as a black marker. A CSV lists, per run,
    the sample summary (n, mean, percentiles, domain extreme, footprint area),
    Cliff's delta, Cohen's d, D and the quantile differences at the tail
    percentiles; a text file adds the group means of |delta| and |QD| used to
    rank how strongly the experiment's parameter changes the swath and cold
    pool (legacy ``sensitivity_analysis.py``).
    """
    reference = _require_reference(ctx)
    opt = ctx.options
    kinds = _kinds(opt)
    fig, axs = _panels(ctx, kinds)
    rows, summary = [], []
    tails = {"swath": opt["swath_percentiles"], "cold_pool": opt["cold_pool_percentiles"]}
    q_header = sorted({float(p) for k in kinds for p in tails[k.key]})
    for ax, kind in zip(axs, kinds, strict=True):
        ref = sample(reference, kind, opt)
        if ref is None:
            ax.set_visible(False)
            continue
        group = []
        for s in [ref] + [s for s in _samples(ctx, kind) if s.run.files != reference.files]:
            is_ref = s is ref
            delta = stats.cliffs_delta(s.values, ref.values)
            d = stats.cohens_d(s.values, ref.values)
            ks = stats.ks_statistic(s.values, ref.values)
            q = stats.quantile_effects(s.values, ref.values, q_header)
            desc = s.describe
            rows.append(
                [kind.key, s.run.name]
                + [desc[k] for k in EFFECT_HEADER[2:12]]
                + [s.extreme, s.area_km2, delta, stats.cliffs_label(delta), d, stats.cohens_label(d), ks]
                + [q[p].diff if p in map(float, tails[kind.key]) else float("nan") for p in q_header]
            )
            if not is_ref:
                group.append((delta, ks, [abs(q[float(p)].diff) for p in tails[kind.key]]))
                ax.scatter(ks, delta, color=s.run.color, marker=s.run.marker, s=80, label=s.run.display_name)
        if group:
            deltas = np.array([g[0] for g in group])
            ax.scatter(
                np.mean([g[1] for g in group]),
                deltas.mean(),
                color="k",
                marker="X",
                s=140,
                label="group mean",
            )
            qd = np.array([g[2] for g in group])
            spread = np.abs(deltas).std(ddof=1) if len(group) > 1 else 0.0
            qd_means = dict(zip(tails[kind.key], qd.mean(axis=0), strict=True))
            summary.append((kind, len(group), np.abs(deltas).mean(), spread, qd_means))
        ax.axhline(0.0, color="k", lw=0.8, ls="--", alpha=0.6)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlim(0.0, 1.0)
        _style(
            ax,
            ctx,
            "Kolmogorov-Smirnov statistic D",
            "Cliff's $\\delta$ (run vs. reference)",
            f"{kind.title}, {_mask_text(kind, opt)}",
        )
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=ctx.fonts["legend"], ncol=2)
    if not rows:
        plt.close(fig)
        raise ValueError("the reference run has no sample to compare against")
    fig.suptitle(f"{ctx.tag}: effect sizes against {reference.display_name}", fontsize=ctx.fonts["suptitle"])
    png = ctx.save(fig)
    write_rows(png.with_suffix(".csv"), EFFECT_HEADER + [f"qd{p:g}" for p in q_header], rows)
    png.with_suffix(".txt").write_text(_report(ctx, reference, rows, summary, q_header))


def _report(ctx, reference: RunInput, rows, summary, q_header) -> str:
    """Plain-text tables of the CSV plus the group's sensitivity summary."""
    opt = ctx.options
    lines = [
        f"Sensitivity of experiment {ctx.tag!r} against reference {reference.display_name!r}",
        f"Swath sample: wind >= {opt['wind_mask_mps']:g} m/s; damaging area: >= {opt['wind_area_mps']:g} m/s",
        f"Cold-pool sample: theta' <= {opt['cold_pool_mask_k']:g} K; "
        f"significant area: <= {opt['cold_pool_area_k']:g} K",
        "Effect sizes are descriptive: spatial autocorrelation makes p-values meaningless (none given).",
        "Positive delta / d / QD: the run's values are larger (stronger wind; warmer, weaker cold pool).",
        "",
    ]
    head = f"{'run':<16}{'n':>9}{'mean':>9}{'p10':>9}{'p90':>9}{'extreme':>9}{'area km2':>10}"
    head += f"{'delta':>8}{'d':>8}{'D':>7}" + "".join(f"{'QD' + format(p, 'g'):>8}" for p in q_header)
    for kind in KINDS.values():
        kind_rows = [r for r in rows if r[0] == kind.key]
        if not kind_rows:
            continue
        lines += [f"-- {kind.key} ".ljust(len(head), "-"), head]
        for r in kind_rows:
            v = dict(zip(EFFECT_HEADER + [f"qd{p:g}" for p in q_header], r, strict=True))
            line = f"{v['run']:<16}{v['n']:>9d}{v['mean']:>9.2f}{v['p10']:>9.2f}{v['p90']:>9.2f}"
            line += f"{v['extreme']:>9.2f}{v['area_km2']:>10.0f}{v['cliffs_delta']:>+8.3f}"
            line += _num(v["cohens_d"], 8, "+.3f") + _num(v["ks_d"], 7, ".3f")
            line += "".join(_num(v[f"qd{p:g}"], 8, "+.2f") for p in q_header)
            lines.append(line)
        lines.append("")
    lines.append("Group sensitivity (runs other than the reference):")
    for kind, n, dmean, dstd, qd in summary:
        qd_text = ", ".join(f"mean |QD{p:g}| = {v:.2f}" for p, v in qd.items())
        lines.append(f"  {kind.key}: n = {n}, mean |delta| = {dmean:.3f} +/- {dstd:.3f}, {qd_text}")
    return "\n".join(lines) + "\n"


def _num(value: float, width: int, spec: str) -> str:
    return format(value, spec).rjust(width) if np.isfinite(value) else "-".rjust(width)


# ------------------------------------------------------------ correlations
CORRELATION_FIELDS = ["cpc", "cph", "thpert", "cbudevp", "cbudmltg", "cbudmlth", "cbudsubg", "cbudsubh"]


@register(
    "wind_correlations",
    STATISTICS,
    scope=Scope.RUN,
    requires=["low_level_wind", tuple(CORRELATION_FIELDS)],
    options={"fields": CORRELATION_FIELDS, "storm_dbz": None},
)
def wind_correlations(ctx: RunContext) -> None:
    """Pearson r between the low-level wind speed and other fields, per output time.

    Surface fields (``cpc``, ``cph``) are used as written; ``thpert`` at the
    lowest level; 3-D microphysical cooling budgets (``cbud*``, negative for
    cooling) as their column minimum, the strongest cooling above each point.
    The sign is kept: wind that rises with stronger cooling gives r < 0. With
    ``storm_dbz`` set, only points with composite reflectivity above it count.
    """
    opt = ctx.options
    times, values, fields = [], {}, None
    for frame in ctx.frames():
        if fields is None:
            fields = [f for f in opt["fields"] if frame.cm1.has(f)]
            missing = sorted(set(opt["fields"]) - set(fields))
            if missing:
                log.warning("wind_correlations: %s lacks %s", frame.cm1.path.name, missing)
            if not fields:
                raise ValueError(f"none of {opt['fields']} is in {frame.cm1.path.name}")
            values = {f: [] for f in fields}
        speed = frame.wind().speed
        if opt["storm_dbz"] is not None:
            speed = np.where(frame.read("cref") > opt["storm_dbz"], speed, np.nan)
        times.append(frame.seconds)
        for f in fields:
            values[f].append(stats.pearson_r(speed, _correlation_field(frame, f)))
    order = np.argsort(times, kind="stable")
    times = np.asarray(times)[order]
    values = {f: np.asarray(v)[order] for f, v in values.items()}

    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    for n, (f, r) in enumerate(values.items()):
        ax.plot(times / 3600.0, r, color=f"C{n}", lw=1.6, marker="o", ms=3, label=f)
    ax.axhline(0.0, color="k", lw=0.8, ls="--", alpha=0.6)
    ax.set_ylim(-1.0, 1.0)
    where = "" if opt["storm_dbz"] is None else f" (cref > {opt['storm_dbz']:g} dBZ)"
    _style(
        ax, ctx, "Time (h)", "Pearson r", f"{ctx.run.display_name}: correlation with low-level wind{where}"
    )
    ax.legend(fontsize=ctx.fonts["legend"], ncol=2)
    png = ctx.save(fig)
    write_rows(
        png.with_suffix(".csv"),
        ["time_s", *values],
        [[t, *(values[f][i] for f in values)] for i, t in enumerate(times)],
    )


def _correlation_field(frame: Frame, name: str) -> np.ndarray:
    if frame.cm1.location(name) == "surface":
        return frame.read(name)
    if name == "thpert":
        return frame.read(name, level=0)
    return frame.column_min(name)
