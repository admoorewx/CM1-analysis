"""Command-line interface: ``cm1plot <command> ...``."""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # off-screen; must precede any pyplot import

from . import plots as _plots  # noqa: E402,F401  (registers the plots)
from . import registry  # noqa: E402
from .config import Config, ConfigError, load_config  # noqa: E402
from .io import DEFAULT_PATTERN, CM1File  # noqa: E402
from .pipeline import Report, RunInput, Settings, run_experiment, run_plots  # noqa: E402
from .style import DEFAULT_DPI, apply_fonts  # noqa: E402

log = logging.getLogger("cm1analysis")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cm1plot", description="Analysis plots for CM1 output.")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging with tracebacks")
    sub = parser.add_subparsers(dest="command", required=True)

    inv = sub.add_parser("inventory", help="show a file's grid and fields, and which plots it supports")
    inv.add_argument("file", help="CM1 netCDF output file")
    inv.add_argument("--fields", action="store_true", help="list every field with its grid location")
    inv.set_defaults(func=_inventory)

    lst = sub.add_parser("list", help="list registered plots")
    lst.add_argument("--category", choices=[str(c) for c in registry.Category])
    lst.set_defaults(func=_list)

    run = sub.add_parser(
        "run",
        help="make the plots of one category",
        description="Make plots for one run (a directory or files) or for an experiment group from "
        "the config. Per-time and per-run plots are made for each run; comparison plots are "
        "made when an experiment is given.",
    )
    run.add_argument("category", choices=[str(c) for c in registry.Category])
    run.add_argument("paths", nargs="*", type=Path, help="run directory, or CM1 files of one run")
    run.add_argument("-c", "--config", type=Path, help="YAML config (paths, experiments, plot options)")
    run.add_argument("-e", "--experiment", help="experiment group from the config to plot")
    run.add_argument("--reference", help="reference run 'group/run' for comparisons (overrides config)")
    run.add_argument("-o", "--output", type=Path, help="output directory (overrides config)")
    run.add_argument("--name", help="run name for output paths (default: the directory name)")
    run.add_argument("--plots", help="comma-separated plot names (default: all in the category)")
    run.add_argument(
        "--scope",
        default="time,run,experiment",
        help="comma-separated scopes to make: time, run, experiment (default: all)",
    )
    run.add_argument("--times", help="comma-separated output times in seconds (default: all)")
    run.add_argument("--dpi", type=int, help="figure resolution (overrides config; default 200)")
    run.set_defaults(func=_run)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    try:
        return args.func(args)
    except (OSError, ValueError, KeyError, ConfigError) as exc:
        print(f"cm1plot: error: {exc}", file=sys.stderr)
        return 1


# ------------------------------------------------------------------ run
def _run(args: argparse.Namespace) -> int:
    config: Config | None = load_config(args.config) if args.config else None
    if args.paths and args.experiment:
        raise ValueError("give either run paths or --experiment, not both")
    if not args.paths and not args.experiment:
        raise ValueError("give a run directory or files, or --experiment with --config")
    if args.experiment and config is None:
        raise ValueError("--experiment needs --config")
    output = args.output or (config.output_dir if config else None)
    if output is None:
        raise ValueError("give --output or a config with output_dir")

    specs = _select(args.category, args.plots, args.scope)
    fonts = apply_fonts(config.fonts if config else None)
    settings = Settings(
        output_dir=output,
        dpi=args.dpi or (config.dpi if config else DEFAULT_DPI),
        fonts=fonts,
        plot_config=config.plots if config else {},
        times=tuple(float(t) for t in args.times.split(",")) if args.times else None,
    )
    pattern = config.file_pattern if config else DEFAULT_PATTERN

    report = Report()
    if args.paths:
        run = RunInput.from_paths(args.paths, pattern, name=args.name)
        log.info("run %s: %d file(s)", run.name, len(run.files))
        report.merge(run_plots(run, specs, settings))
    else:
        runs = [_run_input(name, spec, pattern) for name, spec in config.experiment(args.experiment).items()]
        ref_name = args.reference or config.reference
        reference = None
        if ref_name:
            group, _, name = ref_name.partition("/")
            reference = _run_input(name, config.run(ref_name), pattern)
            runs = [r for r in runs if r.files != reference.files]
        for run in runs:
            log.info("run %s: %d file(s)", run.name, len(run.files))
            report.merge(run_plots(run, specs, settings))
        report.merge(run_experiment(args.experiment, runs, specs, settings, reference=reference))

    return _summarize(report)


def _run_input(name, run, pattern: str) -> RunInput:
    from .io import find_output_files

    return RunInput(
        name=name,
        files=tuple(find_output_files(run.path, pattern)),
        label=run.label,
        color=run.color,
        marker=run.marker,
    )


def _select(category: str, names: str | None, scopes: str) -> list[registry.PlotSpec]:
    wanted_scopes = {registry.Scope(s.strip()) for s in scopes.split(",") if s.strip()}
    specs = registry.plots(category)
    if names:
        chosen = [n.strip() for n in names.split(",") if n.strip()]
        known = {s.name for s in specs}
        unknown = [n for n in chosen if n not in known]
        if unknown:
            raise ValueError(f"no {category} plot(s) {unknown}; available: {sorted(known)}")
        specs = [s for s in specs if s.name in chosen]
    specs = [s for s in specs if s.scope in wanted_scopes]
    if not specs:
        raise ValueError(f"no {category} plots match the selection")
    return specs


def _summarize(report: Report) -> int:
    print(f"\n{len(report.written)} figure(s) written", end="")
    print(f", {len(report.skipped)} plot(s) skipped, {len(report.failures)} failure(s)")
    for name, unmet in report.skipped.items():
        print(f"  skipped {name}: needs {', '.join(unmet)}")
    for name, where, error in report.failures:
        print(f"  FAILED  {name} ({where}): {error}")
    return 1 if report.failures else 0


# ------------------------------------------------------------ inventory
def _inventory(args: argparse.Namespace) -> int:
    with CM1File(args.file) as cm1:
        info = cm1.summary()
        nz, ny, nx = info["shape_zyx"]
        times = info["times_s"]
        print(f"file:         {info['path']}")
        print(f"grid (z,y,x): {nz} x {ny} x {nx}")
        print(f"spacing:      dx={_fmt_m(info['dx_m'])}, dy={_fmt_m(info['dy_m'])}", end="")
        print(f", top={info['ztop_km']:g} km" if info["ztop_km"] is not None else "")
        print(f"times:        {len(times)} ({_fmt_times(times)})")
        print(f"hydrometeors: {', '.join(info['hydrometeors']) or 'none'}")
        counts = Counter(info["fields"].values())
        print("fields:       " + ", ".join(f"{n} {loc}" for loc, n in sorted(counts.items())))
        if args.fields:
            for name, loc in sorted(info["fields"].items()):
                print(f"  {name:12s} {loc:8s} {cm1.units(name):12s} {cm1.long_name(name)}")

        runnable, skipped = registry.check(cm1)
        print(f"\nplots: {len(runnable)} runnable, {len(skipped)} skipped")
        for spec in runnable:
            print(f"  ok    {spec.category}/{spec.name} ({spec.scope})")
        for name, unmet in skipped.items():
            spec = registry.get(name)
            print(f"  skip  {spec.category}/{name}: needs {', '.join(unmet)}")
    return 0


def _list(args: argparse.Namespace) -> int:
    specs = registry.plots(args.category)
    if not specs:
        print("no plots registered" + (f" in {args.category}" if args.category else ""))
    for spec in specs:
        needs = ", ".join(r if isinstance(r, str) else " or ".join(r) for r in spec.requires)
        opts = ", ".join(f"{k}={v!r}" for k, v in spec.options.items())
        print(f"{spec.category}/{spec.name} [{spec.scope}]: {spec.description}")
        print(f"    needs: {needs or 'nothing'}" + (f"; options: {opts}" if opts else ""))
    return 0


def _fmt_m(value: float | None) -> str:
    return "stretched" if value is None else f"{value:g} m"


def _fmt_times(times: list[float]) -> str:
    if len(times) == 1:
        return f"{times[0]:g} s"
    return f"{times[0]:g}-{times[-1]:g} s"


if __name__ == "__main__":
    sys.exit(main())
