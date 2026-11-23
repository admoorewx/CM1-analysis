"""Three-part Helmholtz partition of the horizontal wind on CM1's C-grid (Lynch 1989).

The wind on one level is split into a rotational part ``V_psi = k x grad psi``,
a divergent part ``V_chi = grad chi`` and a harmonic residual ``V_h`` (both
divergence- and vorticity-free), with ``psi = chi = 0`` on the boundary. Both
Poisson problems are solved exactly (to round-off) with discrete sine
transforms. Variants B and C fold the harmonic part into the rotational or the
divergent part.

Modules:

- :mod:`.ops`: C-grid divergence, vorticity and recovery operators.
- :mod:`.solve`: DST Poisson solvers and sparse direct references.
- :mod:`.partition`: :func:`partition_wind` and :func:`apply_variant`.
- :mod:`.diagnostics`: validation table, region of interest, sensitivity tests.
- :mod:`.source`: one checked level of a CM1 file (:class:`WindLevel`).
- :mod:`.output`: the partition netCDF writer.
- :mod:`.compute`: :func:`write_partition` and :func:`partition_file`.

Command line: ``python -m cm1analysis.wind --help``.
"""

from .compute import partition_file, write_partition
from .partition import Partition, apply_variant, partition_wind
from .source import WindLevel

__all__ = [
    "Partition",
    "WindLevel",
    "apply_variant",
    "partition_file",
    "partition_wind",
    "write_partition",
]
