"""Running registered plots over runs, files and times.

A plot function receives one context object and saves figures with
``ctx.save``. Which context depends on the plot's scope:

- :class:`TimeContext` (scope TIME): one output time of one run. It *is* a
  :class:`Frame`, so ``ctx.read("cref")`` reads that time.
- :class:`RunContext` (scope RUN): ``ctx.frames()`` walks every time of the run
  in order; ``ctx.selected(t)`` says whether a figure is wanted at time t.
- :class:`ExperimentContext` (scope EXPERIMENT): several runs;
  ``ctx.common_times()`` lists the times every run has.

Figures go to ``<output>/<run or experiment>/<category>/<plot>/``. A plot that
fails is logged and recorded in the :class:`Report`; the rest still run.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import diagnostics, registry
from .diagnostics import Wind, Window
from .io import CM1File, find_output_files
from .registry import PlotSpec, Scope
from .style import DEFAULT_DPI, FONT_DEFAULTS, save_figure

log = logging.getLogger(__name__)

TIME_TOLERANCE_S = 0.5


# ------------------------------------------------------------------ inputs
@dataclass(frozen=True)
class RunInput:
    """One simulation to plot: its files plus how to label and color it."""

    name: str
    files: tuple[Path, ...]
    label: str | None = None
    color: str = "k"
    marker: str = "o"

    @property
    def display_name(self) -> str:
        return self.label or self.name

    @classmethod
    def from_paths(cls, paths: Sequence[str | Path], pattern: str, name: str | None = None) -> RunInput:
        """A run from one directory, or from an explicit list of files."""
        paths = [Path(p) for p in paths]
        if len(paths) == 1 and paths[0].is_dir():
            files = find_output_files(paths[0], pattern)
            default_name = paths[0].resolve().name
        else:
            missing = [str(p) for p in paths if not p.is_file()]
            if missing:
                raise FileNotFoundError(f"not found: {', '.join(missing)}")
            files = sorted(paths)
            default_name = files[0].resolve().parent.name
        return cls(name=name or default_name, files=tuple(files))


@dataclass(frozen=True)
class Settings:
    output_dir: Path
    dpi: int = DEFAULT_DPI
    fonts: Mapping[str, int] = field(default_factory=lambda: dict(FONT_DEFAULTS))
    plot_config: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)  # per category
    times: tuple[float, ...] | None = None  # only these output times (s); None = all


@dataclass
class Report:
    written: list[Path] = field(default_factory=list)
    failures: list[tuple[str, str, str]] = field(default_factory=list)  # (plot, where, error)
    skipped: dict[str, list[str]] = field(default_factory=dict)  # plot -> unmet requirements

    def merge(self, other: Report) -> None:
        self.written += other.written
        self.failures += other.failures
        for name, unmet in other.skipped.items():
            self.skipped.setdefault(name, unmet)


# ------------------------------------------------------------------ frames
def window_for(cm1: CM1File, extent_km: Sequence[float] | None) -> Window:
    """(y, x) slices covering extent_km = [xmin, xmax, ymin, ymax], or the full domain."""
    if extent_km is None:
        return diagnostics.FULL
    if len(extent_km) != 4:
        raise ValueError(f"extent_km must be [xmin, xmax, ymin, ymax], got {extent_km}")
    x0, x1, y0, y1 = (float(v) for v in extent_km)
    xs = slice(int(np.searchsorted(cm1.x_km, x0)), int(np.searchsorted(cm1.x_km, x1, side="right")))
    ys = slice(int(np.searchsorted(cm1.y_km, y0)), int(np.searchsorted(cm1.y_km, y1, side="right")))
    if xs.stop - xs.start < 2 or ys.stop - ys.start < 2:
        raise ValueError(f"extent_km {list(extent_km)} selects fewer than 2 points in x or y")
    return ys, xs


@dataclass
class Frame:
    """One output time of one file, restricted to a horizontal window."""

    cm1: CM1File
    time: int
    seconds: float
    window: Window = diagnostics.FULL

    @property
    def x(self) -> np.ndarray:
        return self.cm1.x_km[self.window[1]]

    @property
    def y(self) -> np.ndarray:
        return self.cm1.y_km[self.window[0]]

    def level(self, height_km: float | None, staggered: bool = False) -> int:
        """Level index for a height; None means the lowest level."""
        return 0 if height_km is None else self.cm1.level_index(height_km, staggered)

    def read(self, name: str, level: int | None = None, dtype=None) -> np.ndarray:
        ys, xs = self.window
        return self.cm1.read(name, self.time, level=level, y=ys, x=xs, dtype=dtype)

    def wind(self) -> Wind:
        return diagnostics.low_level_wind(self.cm1, self.time, self.window)

    def column_max(self, name: str, top_km: float | None = None) -> np.ndarray:
        return diagnostics.column_reduce(self.cm1, name, self.time, np.fmax, self.window, top_km)

    def column_min(self, name: str, top_km: float | None = None) -> np.ndarray:
        return diagnostics.column_reduce(self.cm1, name, self.time, np.fmin, self.window, top_km)


# ---------------------------------------------------------------- contexts
@dataclass(kw_only=True)
class _Output:
    spec: PlotSpec
    options: dict[str, Any]
    outdir: Path
    tag: str  # run or experiment name, in file names
    settings: Settings
    written: list[Path] = field(default_factory=list)

    @property
    def fonts(self) -> Mapping[str, int]:
        return self.settings.fonts

    def save(self, fig, seconds: float | None = None, suffix: str | None = None) -> Path:
        """Save a figure as <plot>_<tag>[_<suffix>][_<seconds>s].png and close it."""
        parts = [self.spec.name, self.tag]
        if suffix:
            parts.append(suffix)
        if seconds is not None:
            parts.append(f"{round(seconds):06d}s")
        path = save_figure(fig, self.outdir / ("_".join(parts) + ".png"), dpi=self.settings.dpi)
        self.written.append(path)
        log.info("wrote %s", path)
        return path

    def selected(self, seconds: float) -> bool:
        times = self.settings.times
        return times is None or any(abs(seconds - t) <= TIME_TOLERANCE_S for t in times)


@dataclass(kw_only=True)
class TimeContext(Frame, _Output):
    run: RunInput | None = None


@dataclass(kw_only=True)
class RunContext(_Output):
    run: RunInput | None = None

    def frames(self) -> Iterator[Frame]:
        """Every output time of the run in order (each file is open while its frames are used)."""
        window = None
        for path in self.run.files:
            with CM1File(path) as cm1:
                if window is None:
                    window = window_for(cm1, self.options.get("extent_km"))
                for index, seconds in enumerate(cm1.times):
                    yield Frame(cm1, index, float(seconds), window)


@dataclass(kw_only=True)
class ExperimentContext(_Output):
    runs: tuple[RunInput, ...] = ()
    reference: RunInput | None = None

    @property
    def all_runs(self) -> tuple[RunInput, ...]:
        """Reference first (if set), then the experiment's runs."""
        return ((self.reference,) if self.reference else ()) + self.runs

    def common_times(
        self, runs: Sequence[RunInput] | None = None
    ) -> list[tuple[float, dict[str, tuple[Path, int]]]]:
        """Times present in every run, each with {run name: (file, time index)}, in order."""
        runs = self.all_runs if runs is None else runs
        tables = [_time_table(run) for run in runs]
        common = []
        for seconds, entry in tables[0].items():
            matches = {runs[0].name: entry}
            for run, table in zip(runs[1:], tables[1:], strict=True):
                hit = _lookup(table, seconds)
                if hit is None:
                    break
                matches[run.name] = hit
            else:
                common.append((seconds, matches))
        return common

    def open_frame(self, entry: tuple[Path, int], cm1: CM1File) -> Frame:
        return Frame(
            cm1, entry[1], float(cm1.times[entry[1]]), window_for(cm1, self.options.get("extent_km"))
        )


def _time_table(run: RunInput) -> dict[float, tuple[Path, int]]:
    table = {}
    for path in run.files:
        with CM1File(path) as cm1:
            for index, seconds in enumerate(cm1.times):
                table[float(seconds)] = (path, index)
    return table


def _lookup(table: Mapping[float, tuple[Path, int]], seconds: float) -> tuple[Path, int] | None:
    for t, entry in table.items():
        if abs(t - seconds) <= TIME_TOLERANCE_S:
            return entry
    return None


# ------------------------------------------------------------------ runner
def _options(spec: PlotSpec, settings: Settings) -> dict[str, Any]:
    return registry.resolve_options(spec, settings.plot_config.get(str(spec.category), {}))


def _call(spec: PlotSpec, ctx: _Output, where: str, report: Report) -> None:
    try:
        spec.func(ctx)
    except Exception as exc:  # one plot failing must not stop the batch
        log.error("%s failed on %s: %s", spec.name, where, exc, exc_info=log.isEnabledFor(logging.DEBUG))
        report.failures.append((spec.name, where, f"{type(exc).__name__}: {exc}"))
    report.written += ctx.written


def _split_runnable(specs: Sequence[PlotSpec], first_file: Path, report: Report) -> list[PlotSpec]:
    with CM1File(first_file) as cm1:
        runnable, skipped = registry.check(cm1, specs=specs)
    for name, unmet in skipped.items():
        log.warning("skipping %s: %s lacks %s", name, first_file.name, ", ".join(unmet))
    report.skipped.update(skipped)
    return runnable


def run_plots(run: RunInput, specs: Sequence[PlotSpec], settings: Settings) -> Report:
    """Run TIME- and RUN-scope plots on one run."""
    report = Report()
    specs = [s for s in specs if s.scope in (Scope.TIME, Scope.RUN)]
    if not specs:
        return report
    specs = _split_runnable(specs, run.files[0], report)
    options = {s.name: _options(s, settings) for s in specs}  # validate before plotting anything

    def outdir(spec: PlotSpec) -> Path:
        return settings.output_dir / run.name / str(spec.category) / spec.name

    time_specs = [s for s in specs if s.scope == Scope.TIME]
    if time_specs:
        for path in run.files:
            with CM1File(path) as cm1:
                windows = {s.name: window_for(cm1, options[s.name].get("extent_km")) for s in time_specs}
                for index, seconds in enumerate(cm1.times):
                    seconds = float(seconds)
                    for spec in time_specs:
                        ctx = TimeContext(
                            cm1=cm1,
                            time=index,
                            seconds=seconds,
                            window=windows[spec.name],
                            spec=spec,
                            options=options[spec.name],
                            outdir=outdir(spec),
                            tag=run.name,
                            settings=settings,
                            run=run,
                        )
                        if ctx.selected(seconds):
                            _call(spec, ctx, f"{path.name} t={seconds:g} s", report)

    for spec in (s for s in specs if s.scope == Scope.RUN):
        ctx = RunContext(
            spec=spec,
            options=options[spec.name],
            outdir=outdir(spec),
            tag=run.name,
            settings=settings,
            run=run,
        )
        _call(spec, ctx, f"run {run.name}", report)
    return report


def run_experiment(
    name: str,
    runs: Sequence[RunInput],
    specs: Sequence[PlotSpec],
    settings: Settings,
    reference: RunInput | None = None,
) -> Report:
    """Run EXPERIMENT-scope plots comparing several runs (and an optional reference)."""
    report = Report()
    specs = [s for s in specs if s.scope == Scope.EXPERIMENT]
    if not specs:
        return report
    specs = _split_runnable(specs, (reference or runs[0]).files[0], report)
    for spec in specs:
        ctx = ExperimentContext(
            spec=spec,
            options=_options(spec, settings),
            outdir=settings.output_dir / name / str(spec.category) / spec.name,
            tag=name,
            settings=settings,
            runs=tuple(runs),
            reference=reference,
        )
        _call(spec, ctx, f"experiment {name}", report)
    return report
