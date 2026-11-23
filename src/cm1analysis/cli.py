"""Command-line interface: ``cm1plot <command> ...``."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Sequence

import matplotlib

matplotlib.use("Agg")  # off-screen; must precede any pyplot import

from . import plots as _plots  # noqa: E402,F401  (registers the plots)
from . import registry  # noqa: E402
from .io import CM1File  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cm1plot", description="Analysis plots for CM1 output.")
    sub = parser.add_subparsers(dest="command", required=True)

    inv = sub.add_parser("inventory", help="show a file's grid and fields, and which plots it supports")
    inv.add_argument("file", help="CM1 netCDF output file")
    inv.add_argument("--fields", action="store_true", help="list every field with its grid location")
    inv.set_defaults(func=_inventory)

    lst = sub.add_parser("list", help="list registered plots")
    lst.add_argument("--category", choices=[str(c) for c in registry.Category])
    lst.set_defaults(func=_list)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (OSError, ValueError, KeyError) as exc:
        print(f"cm1plot: error: {exc}", file=sys.stderr)
        return 1


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
            print(f"  ok    {spec.category}/{spec.name}")
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
        print(f"{spec.category}/{spec.name}: {spec.description} [needs: {needs or 'nothing'}]")
    return 0


def _fmt_m(value: float | None) -> str:
    return "stretched" if value is None else f"{value:g} m"


def _fmt_times(times: list[float]) -> str:
    if len(times) == 1:
        return f"{times[0]:g} s"
    return f"{times[0]:g}-{times[-1]:g} s"


if __name__ == "__main__":
    sys.exit(main())
