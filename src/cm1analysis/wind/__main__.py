"""Partition CM1 winds on one level into rotational, divergent and harmonic parts.

Writes one output file per input file, so batch jobs can process times separately::

    python -m cm1analysis.wind "/path/to/cm1out_*.nc" --output /path/to/wind_decomposition
    python -m cm1analysis.wind "/path/to/cm1out_*.nc" --output /path/to/wind_decomposition --height-km 5

The exit status is 1 if any validation check failed (see the ``diagnostics``
group of the output), else 0.
"""

from __future__ import annotations

import argparse
import glob
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from .compute import DEFAULT_ROI_DBZ, VARIANTS, partition_file


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m cm1analysis.wind", description=__doc__.splitlines()[0])
    parser.add_argument("input", nargs="+", help="input file(s) or quoted glob(s)")
    parser.add_argument("--output", required=True, type=Path, help="output directory")
    level = parser.add_mutually_exclusive_group()
    level.add_argument("--level", type=int, default=0, help="level index (default: lowest)")
    level.add_argument(
        "--height-km",
        type=float,
        help="height above ground in km; uses the nearest model level (terrain-free runs only)",
    )
    parser.add_argument(
        "--time-range",
        nargs=2,
        type=float,
        metavar=("START", "END"),
        help="only process model times in this inclusive range, in seconds",
    )
    parser.add_argument(
        "--variant",
        choices=VARIANTS,
        default="A",
        help="B and C also write u_rot, v_rot, u_div and v_div (default: A)",
    )
    parser.add_argument(
        "--roi",
        nargs=4,
        type=float,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
        help="fixed region of interest in km (default: automatic reflectivity box)",
    )
    parser.add_argument(
        "--roi-dbz",
        type=float,
        default=DEFAULT_ROI_DBZ,
        help=f"cref threshold for the automatic box (default: {DEFAULT_ROI_DBZ:g})",
    )
    parser.add_argument("--overwrite", action="store_true", help="replace existing outputs")
    parser.add_argument(
        "--workers", type=int, default=-1, help="scipy.fft workers (-1: every CPU of the job's allocation)"
    )
    return parser.parse_args(argv)


def expand_inputs(patterns: Sequence[str]) -> list[str]:
    files = sorted({path for pattern in patterns for path in glob.glob(pattern)})
    if not files:
        raise SystemExit(f"No input files match: {list(patterns)}")
    return files


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.output.mkdir(parents=True, exist_ok=True)
    failures = 0
    for path in expand_inputs(args.input):
        try:
            _, failed = partition_file(
                path,
                args.output,
                time_range=args.time_range,
                overwrite=args.overwrite,
                level=args.level,
                height_km=args.height_km,
                variant=args.variant,
                roi_km=args.roi,
                roi_dbz=args.roi_dbz,
                workers=args.workers,
            )
        except FileExistsError as exc:
            raise SystemExit(str(exc)) from None
        failures += failed
    if failures:
        print(f"{failures} validation check(s) failed; see diagnostics in the output.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
